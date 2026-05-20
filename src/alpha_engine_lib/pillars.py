"""Pillar-decomposed attractiveness scoring — canonical 6-pillar
Pydantic shapes for the institutional / SOTA refactor of research-module
composite scoring.

Origin: 2026-05-20 Brian's "moat is one factor among many — what are the
SOTA institutional factors for stock attractiveness?" arc. Plan doc at
``alpha-engine-docs/private/attractiveness-pillars-260520.md``; ROADMAP
entry under Research (P1) in ``alpha-engine-config/private-docs/ROADMAP.md``.

Why these live in the shared lib (not in alpha-engine-research):

  Same isomorphism rationale as ``agent_schemas`` — the Qual Analyst in
  alpha-engine-research emits ``QualitativePillarAssessment`` via tool-use
  forced output; the replay harness in alpha-engine-backtester and the
  per-pillar attribution analyser need to call ``with_structured_output(
  QualitativePillarAssessment)`` using the EXACT same Pydantic schema the
  production agent used. Without a shared lib, backtester would either
  need a cross-repo dep on research or vendor a drifting local copy.

What's here:

  - ``PILLARS`` tuple — canonical ordering of the 6 pillars.
  - ``PillarLiteral`` — Literal type used in all pillar-keyed fields.
  - ``MoatType`` — the 6 Morningstar / Porter moat archetypes plus
    ``"none"`` for the absent case.
  - ``MoatWidth`` — wide / narrow / none, the Morningstar economic-moat
    rating vocabulary.
  - ``MoatTrend`` — widening / stable / eroding; moat trend captures the
    *time derivative* that a one-shot score loses.
  - ``MoatAssessment`` — the qualitative core of the Quality pillar.
  - ``PillarSubscore`` — per-pillar 0-100 score with confidence + optional
    quant_component / qual_component traceability + evidence list.
  - ``QualitativePillarAssessment`` — the full 6-pillar emission shape the
    Qual Analyst produces via tool-use forced output.

What's NOT here (intentionally):

  - The ``CompositeBreakdown`` shape (Phase 4 of the arc) — lives in
    alpha-engine-research's scoring layer because it's coupled to the
    research-internal regime/macro state.
  - The quant-pillar substrate (factor-substrate composites for Growth +
    Stewardship) — those live in alpha-engine-data's factor profile JSON
    and are consumed by research's ``score_aggregator`` as floats.
  - Stance derivation — Phase 5 of the arc; ``scoring/stance_deriver.py``
    in alpha-engine-research, fed BY pillar subscores but not part of the
    schema layer.

Schema-validation discipline mirrors ``agent_schemas``:

  - ``model_config = ConfigDict(extra="allow")`` on every class because LLM
    outputs may include additional fields (forward-compatible drift).
  - Hard validators only where they defend an observed or anticipated
    LLM failure mode (e.g. moat ``durability_years`` upper bound, score
    range clamps).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ── Pillar vocabulary ────────────────────────────────────────────────────


PILLARS: tuple[str, ...] = (
    "quality",
    "value",
    "momentum",
    "growth",
    "stewardship",
    "defensiveness",
)
"""Canonical ordering of the 6 attractiveness pillars.

Source of truth for both the ``PillarLiteral`` type AND iteration order
in any downstream consumer (composite scoring, dashboard radar charts,
backtester per-pillar attribution). Pinning the tuple lets tests assert
equality rather than set-equality, surfacing unintended reordering.

Why these 6:

- **quality** — durable profitability + capital efficiency (Asness "QMJ"
  + Morningstar Economic Moat + Piotroski F-score). Moat is the
  qualitative core (see ``MoatAssessment``).
- **value** — multi-metric, industry-relative price-to-fundamentals
  (Fama-French HML + Greenblatt earnings yield).
- **momentum** — 12-1m price momentum (Jegadeesh-Titman / Carhart MOM)
  plus earnings-revision momentum + SUE.
- **growth** — *sustainable* growth: reinvestment rate × ROIC, not raw
  revenue CAGR. Compounder vs cash-cow distinction.
- **stewardship** — capital allocation discipline (Morningstar
  Stewardship rating + Buffett/Munger lens): buyback timing, insider
  alignment, M&A track record.
- **defensiveness** — low-vol / beta / drawdown profile (Frazzini-Pedersen
  "Betting Against Beta" + Asness defensive).

Catalyst is preserved as the orthogonal stance-survivor (see
``QualitativePillarAssessment.catalyst_horizon_modulation``) rather than
a 7th pillar weight, because catalyst is a *horizon* modulation rather
than an attractiveness pillar — a compounder thesis can be attractive
across pillars without any near-term catalyst."""


PillarLiteral = Literal[
    "quality",
    "value",
    "momentum",
    "growth",
    "stewardship",
    "defensiveness",
]
"""Pillar name as a Literal type. Used by ``PillarSubscore.pillar`` and any
downstream pillar-keyed field. Kept in sync with ``PILLARS`` by
``test_pillar_literal_matches_pillars_tuple``."""


# ── Moat sub-rubric (qualitative core of Quality pillar) ─────────────────


MoatType = Literal[
    "network_effects",
    "switching_costs",
    "cost_advantage",
    "intangibles",
    "efficient_scale",
    "process_power",
    "none",
]
"""The 6 canonical Morningstar / Porter moat archetypes plus ``"none"``.

- **network_effects** — value to each user grows with total users (Visa,
  Meta, exchanges).
- **switching_costs** — high cost / friction to switch providers
  (Microsoft enterprise, SAP, Oracle, mission-critical SaaS).
- **cost_advantage** — structurally lower unit costs from scale, location,
  or process (Costco, Walmart, Waste Management).
- **intangibles** — brand, patents, regulatory licenses, IP (LVMH,
  pharma patent portfolios, ASML EUV patents).
- **efficient_scale** — small market served well by few players;
  attractive economics + structural deterrent to new entrants (regional
  utilities, pipelines, railroads, Waste Management in many markets).
- **process_power** — proprietary process / know-how others can't
  replicate at scale (TSMC leading-edge nodes, ASML EUV manufacturing,
  certain advanced materials).
- **none** — no identifiable moat. Most stocks fall here; this is the
  honest default rather than fabricating a moat to fill the field."""


MoatWidth = Literal["wide", "narrow", "none"]
"""Morningstar economic-moat width vocabulary.

- **wide** — moat expected to persist 20+ years.
- **narrow** — moat expected to persist 10-20 years.
- **none** — no durable moat / moat under near-term threat.

Most public companies are ``"none"``; a meaningful minority are
``"narrow"``; ``"wide"`` is reserved for the small set of truly durable
franchises (Microsoft / Visa / Costco / Moody's / ASML class)."""


MoatTrend = Literal["widening", "stable", "eroding"]
"""Moat trend — the *time derivative* that a one-shot moat score loses.

A wide-but-eroding moat is structurally different from a narrow-but-
widening moat even if their current widths are 2 ranks apart. Trend is
how the agent expresses competitive dynamics: new entrants emerging,
disruption pressure, regulatory shifts, technology cycles."""


class MoatAssessment(BaseModel):
    """Structured moat assessment — qualitative core of the Quality pillar.

    Emitted by the Qual Analyst as part of ``QualitativePillarAssessment``
    via tool-use forced output. Persisted per ticker as a time-series in
    ``archive/universe/{TICKER}/moat_profile.json`` for trend tracking
    (moats decay slowly; the *time series* is the real signal, not any
    single weekly score).

    Permissive (``extra="allow"``) for forward-compatible LLM drift.
    Hard validators only on the structural bounds.
    """

    model_config = ConfigDict(extra="allow")

    primary_type: MoatType = Field(
        description=(
            "Dominant moat archetype. ``'none'`` is the honest default; do "
            "not fabricate a moat type to fill the field."
        )
    )
    secondary_types: list[MoatType] = Field(
        default_factory=list,
        description=(
            "Additional moat archetypes present but subordinate to primary. "
            "Empty list is fine; many strong moats are single-archetype."
        ),
    )
    width: MoatWidth = Field(
        description=(
            "Morningstar economic-moat width: wide (20y+), narrow (10-20y), "
            "or none. Defaults to ``'none'`` when primary_type is ``'none'``."
        )
    )
    durability_years: int = Field(
        ge=0,
        le=50,
        description=(
            "Estimated years the moat persists. Soft heuristic — useful for "
            "trend tracking + dashboard narrative, not a hard composite "
            "input. Upper-bound at 50y is a sanity cap; real moats rarely "
            "outlast that horizon predictably."
        ),
    )
    trend: MoatTrend = Field(
        description=(
            "Time-derivative of moat strength: widening (improving), stable, "
            "or eroding. Wide-but-eroding is meaningfully different from "
            "narrow-but-widening at the same current width."
        )
    )
    evidence: list[str] = Field(
        default_factory=list,
        description=(
            "Citations from 10-K / 10-Q / 8-K / earnings transcripts in RAG "
            "supporting the assessment. Empty is permitted (LLM may not "
            "always cite) but reviewers should expect ≥1 evidence string "
            "for any non-'none' moat. Free-form strings, not structured "
            "citation objects, for forward compatibility."
        ),
    )

    @field_validator("secondary_types")
    @classmethod
    def _unique_secondary(cls, v: list[MoatType]) -> list[MoatType]:
        """Secondary moat types must be unique within the list."""
        if len(set(v)) != len(v):
            raise ValueError(
                f"moat.secondary_types must be unique; got {v}"
            )
        return v

    @field_validator("evidence")
    @classmethod
    def _trim_evidence_strings(cls, v: list[str]) -> list[str]:
        """Strip whitespace + drop empty strings. LLM outputs sometimes
        produce ``["", " ...some text..."]`` from format-token confusion."""
        return [s.strip() for s in v if s and s.strip()]

    @model_validator(mode="after")
    def _primary_not_in_secondary(self) -> MoatAssessment:
        """Primary moat archetype must not appear in secondary list.

        Anticipated LLM failure mode: agents sometimes restate the primary
        archetype in secondary for emphasis. This drops semantic clarity —
        secondary means *additional, subordinate* archetypes.
        """
        if self.primary_type in self.secondary_types:
            raise ValueError(
                f"moat.primary_type ({self.primary_type!r}) must not "
                f"appear in secondary_types ({self.secondary_types!r})"
            )
        return self


# ── Pillar subscore (per-pillar 0-100 with traceability) ─────────────────


class PillarSubscore(BaseModel):
    """Per-pillar attractiveness subscore — 0-100 with optional
    quant/qual decomposition for traceability.

    The 6 ``PillarSubscore`` instances in ``QualitativePillarAssessment``
    are the qualitative side; ``quant_component`` is populated downstream
    by ``scoring/composite.py`` after the factor-substrate quantitative
    subscore is read. Storing both surfaces (quant + qual + the blended
    score) lets the dashboard render the decomposition and the backtester
    decompose realized alpha into quant-pillar vs qual-pillar contribution.

    At LLM emission time (the qual-analyst tool-use call), only ``pillar``
    + ``score`` + ``confidence`` + ``qual_component`` + ``evidence`` are
    populated; ``quant_component`` and the blended ``score`` may be
    rewritten by the composite scoring layer.

    Permissive (``extra="allow"``) for forward-compatible LLM drift.
    """

    model_config = ConfigDict(extra="allow")

    pillar: PillarLiteral = Field(
        description="Which of the 6 pillars this subscore covers."
    )
    score: int = Field(
        ge=0,
        le=100,
        description=(
            "0-100 blended attractiveness score on this pillar. At LLM "
            "emission time this is the qualitative score; downstream "
            "composite scoring may rewrite to a quant+qual blend."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description=(
            "Agent's confidence in this pillar's assessment. Used by "
            "downstream consumers to weight the within-pillar quant/qual "
            "blend (low-confidence qual → lean on quant)."
        )
    )
    quant_component: float | None = Field(
        default=None,
        description=(
            "Optional quantitative subscore from the factor substrate "
            "(``factors/profiles/latest.json``). Populated by the composite "
            "scoring layer post-LLM-emission. ``None`` for pillars without "
            "quant coverage (e.g. stewardship has thin quant signal)."
        ),
    )
    qual_component: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description=(
            "Optional qualitative-only component the agent emits before "
            "any quant blend. When present, ``score`` may differ from "
            "``qual_component`` post-blend; both are persisted for "
            "traceability."
        ),
    )
    evidence: list[str] = Field(
        default_factory=list,
        description=(
            "Citations / observations the agent used to score this pillar. "
            "Free-form strings for forward compatibility; expect 1-5 entries "
            "per non-trivial pillar score."
        ),
    )

    @field_validator("evidence")
    @classmethod
    def _trim_evidence_strings(cls, v: list[str]) -> list[str]:
        return [s.strip() for s in v if s and s.strip()]


# ── Full qualitative pillar emission (Qual Analyst tool-use output) ──────


class QualitativePillarAssessment(BaseModel):
    """Structured 6-pillar assessment emitted by the Qual Analyst via
    tool-use forced output.

    This is the SOTA structured-output replacement for the current
    opaque-scalar ``qual_score: int 0-100`` emission. Each pillar carries
    its own subscore + evidence, plus the Quality pillar carries a moat
    assessment, plus a catalyst horizon-modulation field captures the
    orthogonal "near-term catalyst shifts effective composite by ±N"
    signal that survives from the legacy stance-taxonomy framing.

    Consumer flow:

      1. ``alpha-engine-research/agents/sector_teams/qual_analyst.py``
         emits this via ``ChatAnthropic.with_structured_output(
         QualitativePillarAssessment)`` as its terminal step.
      2. ``alpha-engine-research/scoring/composite.py`` consumes the 6
         ``PillarSubscore`` fields, blends in quantitative subscores from
         ``factors/profiles/latest.json``, and produces a
         ``CompositeBreakdown`` (composite_score + per-pillar breakdown +
         catalyst modulation + macro shift).
      3. The moat assessment is persisted to
         ``archive/universe/{TICKER}/moat_profile.json`` for trend
         tracking.
      4. ``alpha-engine-research/scoring/stance_deriver.py`` (Phase 5 of
         the arc) reads pillar subscores + catalyst_horizon_modulation
         and emits the derived stance label.
      5. ``alpha-engine-dashboard/pages/2_Signals_and_Research.py``
         renders the 6-axis pillar radar + moat block.

    Permissive (``extra="allow"``) for forward-compatible LLM drift.

    Backward-compatibility translation: when the legacy composite is
    needed (Phase 2 flag-gated soak before Phase 4 cutover), use
    ``derive_legacy_qual_score()`` below.
    """

    model_config = ConfigDict(extra="allow")

    quality: PillarSubscore = Field(
        description="Quality pillar subscore — durable profitability + capital efficiency."
    )
    quality_moat: MoatAssessment = Field(
        description=(
            "Structured moat assessment — qualitative core of the Quality "
            "pillar. Persisted as a time-series per ticker for trend "
            "tracking; the time derivative is the real signal."
        )
    )
    value: PillarSubscore = Field(
        description="Value pillar subscore — multi-metric industry-relative."
    )
    momentum: PillarSubscore = Field(
        description="Momentum pillar subscore — price + earnings momentum."
    )
    growth: PillarSubscore = Field(
        description="Growth pillar subscore — sustainable (reinvestment × ROIC), not raw CAGR."
    )
    stewardship: PillarSubscore = Field(
        description="Stewardship pillar subscore — capital allocation discipline."
    )
    defensiveness: PillarSubscore = Field(
        description="Defensiveness pillar subscore — low-vol / beta / drawdown profile."
    )
    catalyst_horizon_modulation: int = Field(
        default=0,
        ge=-20,
        le=20,
        description=(
            "Near-term catalyst horizon shift on effective composite, ±20. "
            "Positive = imminent catalyst raises near-term attractiveness "
            "(earnings beat expected, FDA approval pending, etc.); negative "
            "= imminent risk lowers it (litigation, guide-down). Orthogonal "
            "to the 6 pillars: a compounder can be attractive across "
            "pillars with ``catalyst_horizon_modulation=0``."
        ),
    )

    def pillar_subscores(self) -> dict[str, PillarSubscore]:
        """Return the 6 pillar subscores as a dict keyed by pillar name.

        Iteration follows ``PILLARS`` ordering. Convenience for downstream
        consumers (composite scoring, stance derivation, dashboard radar)
        that want pillar-keyed access without listing each field by name.
        """
        return {
            "quality": self.quality,
            "value": self.value,
            "momentum": self.momentum,
            "growth": self.growth,
            "stewardship": self.stewardship,
            "defensiveness": self.defensiveness,
        }

    def derive_legacy_qual_score(self) -> int:
        """Translation layer — derive the legacy ``qual_score: int 0-100``
        scalar from the per-pillar subscores.

        Used during Phase 2 soak (``EMIT_PILLAR_ASSESSMENT`` flag-gated)
        when the new shape is emitted but the legacy composite must
        consume a scalar to preserve behavior. Equal-weight mean across
        the 6 pillar scores; catalyst_horizon_modulation NOT folded in
        because the legacy composite already had catalyst handling via
        the stance taxonomy.

        Returns int (rounded) in [0, 100]. The field bounds on each
        pillar's ``score`` (0-100) plus Python's ``round`` make the
        return type-safe; no clamp needed.
        """
        scores = [
            self.quality.score,
            self.value.score,
            self.momentum.score,
            self.growth.score,
            self.stewardship.score,
            self.defensiveness.score,
        ]
        return round(sum(scores) / len(scores))
