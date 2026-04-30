"""Unit tests for BasePreflight primitives."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest

from alpha_engine_lib.preflight import BasePreflight


class _Concrete(BasePreflight):
    """Minimal concrete subclass for testing primitives directly."""
    def run(self) -> None:
        self.check_env_vars("FAKE_VAR")


# ── Constructor ──────────────────────────────────────────────────────────


def test_missing_bucket_raises():
    with pytest.raises(ValueError, match="bucket is required"):
        _Concrete("")


def test_region_defaults_from_env(monkeypatch):
    monkeypatch.setenv("AWS_REGION", "eu-west-1")
    p = _Concrete("bkt")
    assert p.region == "eu-west-1"


def test_region_defaults_to_us_east_1(monkeypatch):
    monkeypatch.delenv("AWS_REGION", raising=False)
    p = _Concrete("bkt")
    assert p.region == "us-east-1"


# ── run() override enforcement ───────────────────────────────────────────


def test_base_run_raises_not_implemented():
    p = BasePreflight("bkt")
    with pytest.raises(NotImplementedError, match="must override run"):
        p.run()


# ── check_env_vars ───────────────────────────────────────────────────────


def test_check_env_vars_all_set(monkeypatch):
    monkeypatch.setenv("FOO", "1")
    monkeypatch.setenv("BAR", "1")
    p = _Concrete("bkt")
    p.check_env_vars("FOO", "BAR")  # should not raise


def test_check_env_vars_one_missing(monkeypatch):
    monkeypatch.setenv("FOO", "1")
    monkeypatch.delenv("BAR", raising=False)
    p = _Concrete("bkt")
    with pytest.raises(RuntimeError, match=r"\['BAR'\]"):
        p.check_env_vars("FOO", "BAR")


def test_check_env_vars_empty_value_counts_as_missing(monkeypatch):
    monkeypatch.setenv("FOO", "")
    p = _Concrete("bkt")
    with pytest.raises(RuntimeError, match=r"\['FOO'\]"):
        p.check_env_vars("FOO")


# ── check_s3_bucket ──────────────────────────────────────────────────────


def test_check_s3_bucket_success():
    p = _Concrete("bkt")
    mock_client = mock.Mock()
    mock_client.head_bucket.return_value = {}
    with mock.patch("boto3.client", return_value=mock_client):
        p.check_s3_bucket()
    mock_client.head_bucket.assert_called_once_with(Bucket="bkt")


def test_check_s3_bucket_failure_raises():
    p = _Concrete("bkt")
    mock_client = mock.Mock()
    mock_client.head_bucket.side_effect = Exception("AccessDenied")
    with mock.patch("boto3.client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="S3 bucket 'bkt' unreachable"):
            p.check_s3_bucket()


# ── check_s3_key ─────────────────────────────────────────────────────────


def test_check_s3_key_exists():
    p = _Concrete("bkt")
    mock_client = mock.Mock()
    mock_client.head_object.return_value = {
        "LastModified": datetime.now(timezone.utc),
    }
    with mock.patch("boto3.client", return_value=mock_client):
        p.check_s3_key("some/key")


def test_check_s3_key_missing_raises():
    from botocore.exceptions import ClientError
    p = _Concrete("bkt")
    mock_client = mock.Mock()
    mock_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject"
    )
    with mock.patch("boto3.client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="does not exist"):
            p.check_s3_key("some/key")


def test_check_s3_key_stale_raises():
    p = _Concrete("bkt")
    mock_client = mock.Mock()
    mock_client.head_object.return_value = {
        "LastModified": datetime.now(timezone.utc) - timedelta(days=10),
    }
    with mock.patch("boto3.client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="10 days stale"):
            p.check_s3_key("some/key", max_age_days=4)


def test_check_s3_key_fresh_passes():
    p = _Concrete("bkt")
    mock_client = mock.Mock()
    mock_client.head_object.return_value = {
        "LastModified": datetime.now(timezone.utc) - timedelta(days=2),
    }
    with mock.patch("boto3.client", return_value=mock_client):
        p.check_s3_key("some/key", max_age_days=4)


def test_check_s3_key_non_404_error_raises_generic():
    from botocore.exceptions import ClientError
    p = _Concrete("bkt")
    mock_client = mock.Mock()
    mock_client.head_object.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied"}}, "HeadObject"
    )
    with mock.patch("boto3.client", return_value=mock_client):
        with pytest.raises(RuntimeError, match="unreachable"):
            p.check_s3_key("some/key")


# ── check_ib_paper_account ───────────────────────────────────────────────


def test_check_ib_paper_account_valid():
    p = _Concrete("bkt")
    p.check_ib_paper_account("DU1234567")


def test_check_ib_paper_account_live_raises():
    p = _Concrete("bkt")
    with pytest.raises(RuntimeError, match="not a paper account"):
        p.check_ib_paper_account("U1234567")


def test_check_ib_paper_account_empty_raises():
    p = _Concrete("bkt")
    with pytest.raises(RuntimeError, match="account_id is empty"):
        p.check_ib_paper_account("")


# ── check_arcticdb_fresh (stubbed — arcticdb is optional) ────────────────


def test_check_arcticdb_fresh_missing_import_raises():
    """If arcticdb isn't installed, the primitive should raise with a
    clear install hint."""
    p = _Concrete("bkt")
    with mock.patch.dict("sys.modules", {"arcticdb": None}):
        with pytest.raises(RuntimeError, match="arcticdb not importable"):
            p.check_arcticdb_fresh("universe", "SPY", max_stale_days=4)


def test_check_arcticdb_fresh_stale_raises():
    """Full path test: mock arcticdb + pandas and assert the freshness
    check fires when the last date is older than the threshold."""
    pd = pytest.importorskip("pandas")
    try:
        import arcticdb  # noqa: F401
    except ImportError:
        pytest.skip("arcticdb not installed")

    p = _Concrete("bkt")
    old_idx = pd.DatetimeIndex([pd.Timestamp("2026-04-01")])
    stale_df = pd.DataFrame({"Close": [100.0]}, index=old_idx)

    mock_versioned_item = mock.Mock()
    mock_versioned_item.data = stale_df
    mock_lib = mock.Mock()
    mock_lib.read.return_value = mock_versioned_item
    mock_arctic = mock.Mock()
    mock_arctic.get_library.return_value = mock_lib

    with mock.patch("arcticdb.Arctic", return_value=mock_arctic):
        # Set "now" implicitly — stale_df is 2026-04-01, today is 2026-04-14+
        with pytest.raises(RuntimeError, match="days stale"):
            p.check_arcticdb_fresh("universe", "SPY", max_stale_days=4)


# ── check_arcticdb_universe_fresh (per-symbol scan) ──────────────────────


def _build_mock_lib(symbol_to_last_date: dict, fail_symbols: tuple = ()):
    """Helper: construct a mocked ArcticDB library where ``tail(sym, n=1)``
    returns a frame with ``last_date`` for each symbol, or raises for
    symbols in ``fail_symbols``."""
    pd = pytest.importorskip("pandas")
    mock_lib = mock.Mock()
    mock_lib.list_symbols.return_value = list(symbol_to_last_date.keys())

    def _tail(sym, n=1):
        if sym in fail_symbols:
            raise RuntimeError(f"simulated read failure for {sym}")
        last = symbol_to_last_date[sym]
        df = pd.DataFrame(
            {"Close": [1.0]},
            index=pd.DatetimeIndex([pd.Timestamp(last)]),
        )
        item = mock.Mock()
        item.data = df
        return item

    mock_lib.tail.side_effect = _tail
    return mock_lib


def test_check_arcticdb_universe_fresh_missing_import_raises():
    p = _Concrete("bkt")
    with mock.patch.dict("sys.modules", {"arcticdb": None}):
        with pytest.raises(RuntimeError, match="arcticdb not importable"):
            p.check_arcticdb_universe_fresh("universe", max_stale_days=5)


def test_check_arcticdb_universe_fresh_all_fresh_passes():
    pytest.importorskip("pandas")
    try:
        import arcticdb  # noqa: F401
    except ImportError:
        pytest.skip("arcticdb not installed")

    today = datetime.now(timezone.utc).date()
    fresh = {
        "AAPL": today,
        "MSFT": today - timedelta(days=1),
        "NVDA": today - timedelta(days=2),
    }
    mock_lib = _build_mock_lib(fresh)
    mock_arctic = mock.Mock()
    mock_arctic.get_library.return_value = mock_lib

    p = _Concrete("bkt")
    with mock.patch("arcticdb.Arctic", return_value=mock_arctic):
        p.check_arcticdb_universe_fresh("universe", max_stale_days=5)


def test_check_arcticdb_universe_fresh_stale_symbol_raises():
    pytest.importorskip("pandas")
    try:
        import arcticdb  # noqa: F401
    except ImportError:
        pytest.skip("arcticdb not installed")

    today = datetime.now(timezone.utc).date()
    mixed = {
        "AAPL": today,
        "ASGN": today - timedelta(days=30),  # well past 5d threshold
        "MSFT": today - timedelta(days=1),
    }
    mock_lib = _build_mock_lib(mixed)
    mock_arctic = mock.Mock()
    mock_arctic.get_library.return_value = mock_lib

    p = _Concrete("bkt")
    with mock.patch("arcticdb.Arctic", return_value=mock_arctic):
        with pytest.raises(RuntimeError, match=r"have stale data.*ASGN"):
            p.check_arcticdb_universe_fresh("universe", max_stale_days=5)


def test_check_arcticdb_universe_fresh_empty_library_raises():
    pytest.importorskip("pandas")
    try:
        import arcticdb  # noqa: F401
    except ImportError:
        pytest.skip("arcticdb not installed")

    mock_lib = mock.Mock()
    mock_lib.list_symbols.return_value = []
    mock_arctic = mock.Mock()
    mock_arctic.get_library.return_value = mock_lib

    p = _Concrete("bkt")
    with mock.patch("arcticdb.Arctic", return_value=mock_arctic):
        with pytest.raises(RuntimeError, match="zero symbols"):
            p.check_arcticdb_universe_fresh("universe", max_stale_days=5)


def test_check_arcticdb_universe_fresh_read_error_is_fatal():
    """A symbol that can't be read is treated as fatal — silent read
    errors here would mask exactly the write-skip class this scan
    exists to catch."""
    pytest.importorskip("pandas")
    try:
        import arcticdb  # noqa: F401
    except ImportError:
        pytest.skip("arcticdb not installed")

    today = datetime.now(timezone.utc).date()
    symbols = {"AAPL": today, "BROKEN": today, "MSFT": today}
    mock_lib = _build_mock_lib(symbols, fail_symbols=("BROKEN",))
    mock_arctic = mock.Mock()
    mock_arctic.get_library.return_value = mock_lib

    p = _Concrete("bkt")
    with mock.patch("arcticdb.Arctic", return_value=mock_arctic):
        with pytest.raises(RuntimeError, match=r"could not be read.*BROKEN"):
            p.check_arcticdb_universe_fresh("universe", max_stale_days=5)


def test_check_arcticdb_universe_fresh_library_unreachable_raises():
    pytest.importorskip("pandas")
    try:
        import arcticdb  # noqa: F401
    except ImportError:
        pytest.skip("arcticdb not installed")

    mock_arctic = mock.Mock()
    mock_arctic.get_library.side_effect = RuntimeError("S3 timeout")

    p = _Concrete("bkt")
    with mock.patch("arcticdb.Arctic", return_value=mock_arctic):
        with pytest.raises(RuntimeError, match="library 'universe' unreachable"):
            p.check_arcticdb_universe_fresh("universe", max_stale_days=5)


def test_check_arcticdb_universe_fresh_stale_list_truncated_at_10():
    """When more than 10 symbols are stale, the error message lists
    the 10 stalest plus a +N more counter."""
    pytest.importorskip("pandas")
    try:
        import arcticdb  # noqa: F401
    except ImportError:
        pytest.skip("arcticdb not installed")

    today = datetime.now(timezone.utc).date()
    # 15 stale symbols; staleness varies so sort-by-stalest can be checked
    stale = {
        f"TKR{i:02d}": today - timedelta(days=10 + i) for i in range(15)
    }
    mock_lib = _build_mock_lib(stale)
    mock_arctic = mock.Mock()
    mock_arctic.get_library.return_value = mock_lib

    p = _Concrete("bkt")
    with mock.patch("arcticdb.Arctic", return_value=mock_arctic):
        with pytest.raises(RuntimeError, match=r"\+5 more"):
            p.check_arcticdb_universe_fresh("universe", max_stale_days=5)
