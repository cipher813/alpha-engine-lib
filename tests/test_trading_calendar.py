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
        assert is_trading_day(date(2026, 7, 2)) is True


class TestNextTradingDay:
    def test_skips_weekend(self):
        assert next_trading_day(date(2026, 4, 17)) == date(2026, 4, 20)  # Fri → Mon

    def test_skips_holiday(self):
        assert next_trading_day(date(2026, 4, 2)) == date(2026, 4, 6)

    def test_consecutive_trading_days(self):
        assert next_trading_day(date(2026, 4, 15)) == date(2026, 4, 16)


class TestHolidayCoverage:
    def test_covers_through_2030(self):
        assert {d.year for d in NYSE_HOLIDAYS} >= {2025, 2026, 2027, 2028, 2029, 2030}
