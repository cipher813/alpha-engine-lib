"""Unit tests for alpha_engine_lib.trading_calendar."""
from datetime import date

from alpha_engine_lib.trading_calendar import (
    NYSE_HOLIDAYS,
    is_trading_day,
    next_trading_day,
)


class TestIsTradingDay:
    def test_regular_weekday(self):
        assert is_trading_day(date(2026, 4, 16)) is True  # Thursday

    def test_weekend_saturday(self):
        assert is_trading_day(date(2026, 4, 18)) is False

    def test_weekend_sunday(self):
        assert is_trading_day(date(2026, 4, 19)) is False

    def test_new_years_day(self):
        assert is_trading_day(date(2026, 1, 1)) is False

    def test_good_friday_2026(self):
        assert is_trading_day(date(2026, 4, 3)) is False

    def test_independence_day_observed_2026(self):
        """2026 July 4 is a Saturday; observed on Friday July 3."""
        assert is_trading_day(date(2026, 7, 3)) is False
        assert is_trading_day(date(2026, 7, 2)) is True  # Thursday before


class TestNextTradingDay:
    def test_skips_weekend(self):
        assert next_trading_day(date(2026, 4, 17)) == date(2026, 4, 20)  # Fri → Mon

    def test_skips_holiday(self):
        # Apr 2 (Thu) → next is Apr 6 (Mon) because Apr 3 is Good Friday + weekend
        assert next_trading_day(date(2026, 4, 2)) == date(2026, 4, 6)

    def test_consecutive_trading_days(self):
        assert next_trading_day(date(2026, 4, 15)) == date(2026, 4, 16)


class TestHolidayCoverage:
    def test_holidays_cover_through_2030(self):
        years_covered = {d.year for d in NYSE_HOLIDAYS}
        assert years_covered >= {2025, 2026, 2027, 2028, 2029, 2030}

    def test_every_year_has_ten_holidays(self):
        """NYSE has 9-10 observed holidays per year; 2028 has 9 (NYD on Sat not observed)."""
        by_year: dict[int, int] = {}
        for d in NYSE_HOLIDAYS:
            by_year[d.year] = by_year.get(d.year, 0) + 1
        for year, count in by_year.items():
            assert 9 <= count <= 10, f"{year} has {count} holidays (expected 9-10)"
