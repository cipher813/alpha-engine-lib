"""
dates.py — canonical "current date" attribution for trade-related artifacts.

Implements the dual-tracking convention from
``alpha-engine-docs/private/DATE_CONVENTIONS.md``: every trade-related
artifact records both a ``calendar_date`` (wall-clock UTC) and a
``trading_day`` (last completed NYSE session). The ``trading_day``
attribution is strictly backward-looking — never ahead of "now" — so
artifacts produced on weekends, holidays, or pre-open weekday mornings
attribute to the most recent session that has actually closed.

Use this at every artifact-write site::

    from alpha_engine_lib.dates import now_dual

    dd = now_dual()
    record = {
        "calendar_date": dd.calendar_date,
        "trading_day": dd.trading_day,
        ...
    }

For backfilling historical rows that only have a wall-clock timestamp::

    from alpha_engine_lib.dates import session_for_timestamp

    trading_day = session_for_timestamp(row["created_at"])

This module is a thin wrapper over
``alpha_engine_lib.trading_calendar.last_closed_trading_day`` — its purpose
is to standardize the *output shape* (DualDate) and provide a single
canonical entry point so every consumer sees the same semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from .trading_calendar import last_closed_trading_day


@dataclass(frozen=True)
class DualDate:
    """Calendar + trading_day attribution for a moment in time.

    Both fields are ISO ``yyyy-mm-dd`` strings — easy to serialize across
    JSON / SQLite / parquet boundaries with no timezone ambiguity.

    Attributes:
        calendar_date: wall-clock UTC date when the artifact was produced.
            Audit trail. Same on holidays/weekends as any other day; reflects
            *when* the process ran, not which session the data is about.
        trading_day: last NYSE trading session whose 4:00 PM ET close has
            occurred at or before the given moment. Strictly backward-looking;
            equals ``last_closed_trading_day(now)``. Never ahead of "now".

    Example::

        >>> from datetime import datetime, timezone
        >>> from alpha_engine_lib.dates import now_dual
        >>> sat = datetime(2026, 4, 25, 0, 0, tzinfo=timezone.utc)
        >>> dd = now_dual(now=sat)
        >>> dd.calendar_date
        '2026-04-25'
        >>> dd.trading_day  # Sat isn't a session; walk back to Fri
        '2026-04-24'
    """

    calendar_date: str
    trading_day: str


def now_dual(*, now: datetime | None = None) -> DualDate:
    """Canonical current-date attribution for the alpha-engine system.

    Every artifact-write site should call this to populate the
    ``calendar_date`` and ``trading_day`` columns/fields, rather than
    reaching for ``date.today()`` or ``datetime.now().date()``. Calling
    here ensures consistent semantics across modules and prevents the
    drift that motivated the convention (see
    ``alpha-engine-docs/private/DATE_CONVENTIONS.md``).

    Args:
        now: timezone-aware datetime. Defaults to current UTC time.
            Naive datetimes are interpreted as UTC for ``calendar_date``
            and forwarded to ``last_closed_trading_day`` (which itself
            assumes NYSE-local for naive inputs — this is intentional;
            the helper hides the conversion).

    Returns:
        DualDate where ``calendar_date`` is the UTC date of ``now`` and
        ``trading_day`` is the last NYSE session that has fully closed at
        or before ``now``.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    cal_utc = now.astimezone(timezone.utc).date()
    td = last_closed_trading_day(now)

    return DualDate(
        calendar_date=cal_utc.isoformat(),
        trading_day=td.isoformat(),
    )


def session_for_timestamp(ts: datetime) -> str:
    """Trading day a timestamp belongs to under the dual-tracking convention.

    Backward-looking: returns the most recent NYSE trading session whose
    4:00 PM ET close has occurred at or before ``ts``. Used to backfill the
    ``trading_day`` column on historical rows that only have a wall-clock
    timestamp (``created_at``, ``fill_time``, etc.).

    Args:
        ts: timezone-aware datetime. Naive timestamps are assumed UTC.

    Returns:
        ISO ``yyyy-mm-dd`` string of the trading day.

    Example::

        >>> from datetime import datetime
        >>> from zoneinfo import ZoneInfo
        >>> # Mon 9 AM ET — session not yet closed
        >>> ts = datetime(2026, 4, 27, 9, 0, tzinfo=ZoneInfo("America/New_York"))
        >>> session_for_timestamp(ts)
        '2026-04-24'
    """
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return last_closed_trading_day(ts).isoformat()
