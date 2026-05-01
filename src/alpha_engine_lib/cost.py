"""
LLM cost-pricing primitive for the Alpha Engine cost-telemetry stream.

This module is the price-table side of the P1 "Per-run LLM cost telemetry as
code artifact" workstream. The capture wrapper in
:mod:`alpha_engine_lib.decision_capture` records token counts on every LLM
call; this module translates token counts × model × wall-clock time into a
USD cost figure.

**Design rule — tokens are immutable, dollars are derived.** Per the
roadmap entry's scope, dollar amounts are NEVER persisted as the load-bearing
analytics column. Every captured artifact stores token counts; cost is
recomputed from the active rate card at query time. That way, if Anthropic
changes pricing or a rate-card entry was wrong when it was first written,
historical numbers can be repriced without rewriting captured data.

``ModelMetadata.cost_usd`` exists as a derived convenience (handy for
emails + dashboards that don't want to load the rate card on every read);
it is overwritable by :func:`recompute_cost` and must not be treated as
canonical.

**Effective dates.** Each ``PriceCard`` carries an ``effective_from``
date. :meth:`PriceTable.get` returns the card whose ``effective_from`` is
the latest ≤ the query date — so a January call gets January rates even
when the YAML has been updated for April rates. Cards for the same model
must be ordered by ``effective_from`` ascending; the loader hard-fails on
overlap or unsorted input per ``feedback_no_silent_fails``.

**Public surface:**

- :class:`PriceCard` — one (model_name, effective_from) → per-1M-token rate row.
- :class:`PriceTable` — wraps a list of cards with effective-date lookup.
- :func:`load_pricing` — reads ``model_pricing.yaml`` into a ``PriceTable``.
- :func:`compute_cost` — pure math from token counts + price card.
- :func:`recompute_cost` — recompute and overwrite ``cost_usd`` on a
  ``ModelMetadata`` from a ``PriceTable`` and a query date.
- :exc:`PriceCardLookupError` — raised when no card matches a (model, date)
  query (do not swallow).

Workstream design: ``alpha-engine-docs/private/ROADMAP.md`` line ~1708
("Per-run LLM cost telemetry as code artifact").
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from alpha_engine_lib.decision_capture import ModelMetadata


# ── Price card ────────────────────────────────────────────────────────────


class PriceCard(BaseModel):
    """One row of the price table — per-model, per-effective-date rate.

    All four prices are USD per 1,000,000 tokens. Cache-write and cache-
    read prices follow Anthropic's prompt-caching semantics: cache-write
    tokens are billed at ~1.25× the input price, cache-read tokens at
    ~0.10× the input price. The fields are stored explicitly rather than
    derived from a multiplier so that future provider changes (or the
    addition of non-Anthropic providers) don't require a math change.

    A card applies to its model from ``effective_from`` until the next
    card for the same model, exclusive on the new card's ``effective_from``
    date.
    """

    model_config = ConfigDict(extra="forbid")

    model_name: str
    effective_from: date
    input_per_1m: float = Field(ge=0.0)
    output_per_1m: float = Field(ge=0.0)
    cache_read_per_1m: float = Field(ge=0.0)
    cache_create_per_1m: float = Field(ge=0.0)


# ── Errors ────────────────────────────────────────────────────────────────


class PriceCardLookupError(LookupError):
    """Raised when :meth:`PriceTable.get` finds no card matching the query.

    Per ``feedback_no_silent_fails``, the cost path does not silently
    return zero on missing models or out-of-range dates — that would
    bury cost regressions. Callers may catch this if they want a
    best-effort price (e.g. dashboard fallback), but the default is
    hard-fail.
    """


class PriceTableLoadError(RuntimeError):
    """Raised when ``model_pricing.yaml`` is malformed.

    Structural validation: missing top-level key, unknown fields, or
    cards for the same model not sorted ascending by ``effective_from``.
    """


# ── Price table ───────────────────────────────────────────────────────────


class PriceTable(BaseModel):
    """Ordered collection of :class:`PriceCard` rows with effective-date lookup.

    Construction-time invariants (enforced by ``model_validator``):

    1. Cards for the same model are sorted ascending by ``effective_from``.
    2. No two cards for the same model share an ``effective_from`` date.

    Lookups via :meth:`get` return the latest card whose ``effective_from``
    is ≤ the query date; if no such card exists for the model, raises
    :exc:`PriceCardLookupError`.
    """

    model_config = ConfigDict(extra="forbid")

    cards: list[PriceCard]

    @model_validator(mode="after")
    def _validate_card_ordering(self) -> "PriceTable":
        seen_dates: dict[str, list[date]] = {}
        for card in self.cards:
            seen_dates.setdefault(card.model_name, []).append(card.effective_from)
        for model_name, dates in seen_dates.items():
            if len(set(dates)) != len(dates):
                raise PriceTableLoadError(
                    f"PriceTable: duplicate effective_from date for model "
                    f"{model_name!r}: {dates}"
                )
            if dates != sorted(dates):
                raise PriceTableLoadError(
                    f"PriceTable: cards for model {model_name!r} are not "
                    f"sorted ascending by effective_from: {dates}"
                )
        return self

    def get(self, model_name: str, at: datetime | date) -> PriceCard:
        """Return the active :class:`PriceCard` for ``model_name`` at ``at``.

        ``at`` may be a ``datetime`` (UTC offsets accepted; only the date
        component is used for lookup) or a ``date``. The returned card is
        the one whose ``effective_from`` is the latest among cards ≤ ``at``.

        Raises :exc:`PriceCardLookupError` if the model has no cards or
        every card's ``effective_from`` is later than ``at``.
        """
        query_date = at.date() if isinstance(at, datetime) else at
        candidates = [
            c for c in self.cards
            if c.model_name == model_name and c.effective_from <= query_date
        ]
        if not candidates:
            raise PriceCardLookupError(
                f"No price card for model {model_name!r} active on {query_date}"
            )
        # cards are validated sorted ascending; latest active = last match.
        return max(candidates, key=lambda c: c.effective_from)


# ── YAML loader ───────────────────────────────────────────────────────────


def load_pricing(path: Path | str) -> PriceTable:
    """Load ``model_pricing.yaml`` from ``path`` into a :class:`PriceTable`.

    Expected YAML shape::

        # version: 1
        cards:
          - model_name: claude-haiku-4-5
            effective_from: 2026-01-01
            input_per_1m: 1.00
            output_per_1m: 5.00
            cache_read_per_1m: 0.10
            cache_create_per_1m: 1.25
          - model_name: claude-sonnet-4-6
            effective_from: 2026-01-01
            input_per_1m: 3.00
            ...

    Validation:

    1. File must exist; missing file → :exc:`FileNotFoundError`.
    2. Top-level must contain ``cards: [...]``.
    3. Each card validated via :class:`PriceCard` (extra fields rejected).
    4. Cards-per-model sorted ascending by ``effective_from`` (validator).

    Returns the loaded :class:`PriceTable`. Hard-fails on any malformation
    per ``feedback_no_silent_fails``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"model_pricing.yaml not found at {path}")

    with path.open() as fh:
        raw: Any = yaml.safe_load(fh)

    if not isinstance(raw, dict) or "cards" not in raw:
        raise PriceTableLoadError(
            f"{path}: expected top-level mapping with 'cards' key; "
            f"got {type(raw).__name__}"
        )
    if not isinstance(raw["cards"], list):
        raise PriceTableLoadError(
            f"{path}: 'cards' must be a list; got {type(raw['cards']).__name__}"
        )

    cards = [PriceCard.model_validate(entry) for entry in raw["cards"]]
    return PriceTable(cards=cards)


# ── Cost math ─────────────────────────────────────────────────────────────


_TOKENS_PER_PRICE_UNIT = 1_000_000


def compute_cost(
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int,
    cache_create_tokens: int,
    card: PriceCard,
) -> float:
    """Compute USD cost from token counts and a single :class:`PriceCard`.

    Pure math; no I/O. Caller is responsible for selecting the correct
    card via :meth:`PriceTable.get` (which knows about effective dates).

    Each token class is billed at its per-1M-token rate, summed.
    """
    return (
        input_tokens * card.input_per_1m
        + output_tokens * card.output_per_1m
        + cache_read_tokens * card.cache_read_per_1m
        + cache_create_tokens * card.cache_create_per_1m
    ) / _TOKENS_PER_PRICE_UNIT


def recompute_cost(
    metadata: ModelMetadata,
    table: PriceTable,
    *,
    at: datetime | date | None = None,
    overwrite: bool = True,
) -> float:
    """Recompute ``cost_usd`` for ``metadata`` against ``table``.

    Returns the freshly computed USD cost. By default also overwrites
    ``metadata.cost_usd`` in place (the field is treated as a derived
    convenience — see module docstring).

    Parameters
    ----------
    metadata
        The :class:`ModelMetadata` whose tokens are priced.
    table
        Active price table.
    at
        Wall-clock date for price-card lookup. Defaults to ``datetime.
        now(timezone.utc)`` — appropriate for live recompute paths.
        Historical recompute (replay against a different rate card)
        passes the original capture timestamp.
    overwrite
        If ``True`` (default), assigns the result to ``metadata.cost_usd``
        before returning. Set to ``False`` for read-only repricing.

    Raises
    ------
    PriceCardLookupError
        If ``table`` has no card for ``metadata.model_name`` active at
        ``at``. Per ``feedback_no_silent_fails`` — silent zero-pricing
        on a missing model would bury cost regressions.
    """
    when = at if at is not None else datetime.now(timezone.utc)
    card = table.get(metadata.model_name, when)
    cost = compute_cost(
        input_tokens=metadata.input_tokens,
        output_tokens=metadata.output_tokens,
        cache_read_tokens=metadata.cache_read_tokens,
        cache_create_tokens=metadata.cache_create_tokens,
        card=card,
    )
    if overwrite:
        metadata.cost_usd = cost
    return cost
