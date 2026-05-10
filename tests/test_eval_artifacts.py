"""Tests for ``alpha_engine_lib.eval_artifacts``.

Validates the canonical eval-style artifact partition convention:
- ``new_eval_run_id`` returns a YYMMDDHHMM string from a UTC moment
- ``eval_artifact_key`` formats {prefix}/{run_id}.json (default) or
  {prefix}/{run_id}_{basename} (named). Flat layout — the YYMMDDHHMM
  run_id encodes the date itself, so no date sub-partition is needed.
- ``eval_latest_key`` returns {prefix}/latest.json
- Trailing/leading slashes are normalized away so callers don't have
  to think about prefix shape
- Both helpers compose with each other and with ``now_dual`` to produce
  fully-canonical paths in one call site
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from alpha_engine_lib.eval_artifacts import (
    EVAL_LATEST_FILENAME,
    eval_artifact_key,
    eval_latest_key,
    new_eval_run_id,
)


class TestNewEvalRunId:

    def test_format_is_yymmddhhmm(self):
        # Inject a known UTC moment → exact YYMMDDHHMM expected
        moment = datetime(2026, 5, 10, 14, 37, 0, tzinfo=timezone.utc)
        assert new_eval_run_id(now=moment) == "2605101437"

    def test_length_is_ten_chars(self):
        moment = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        rid = new_eval_run_id(now=moment)
        assert len(rid) == 10
        assert rid == "2601010000"

    def test_minute_resolution_distinct_minutes_yield_distinct_ids(self):
        m1 = datetime(2026, 5, 10, 14, 37, 0, tzinfo=timezone.utc)
        m2 = datetime(2026, 5, 10, 14, 38, 0, tzinfo=timezone.utc)
        assert new_eval_run_id(now=m1) != new_eval_run_id(now=m2)

    def test_seconds_within_minute_collide_by_design(self):
        # Two runs within the same UTC minute MUST produce the same
        # run_id — the convention is minute-granularity.
        m1 = datetime(2026, 5, 10, 14, 37, 1, tzinfo=timezone.utc)
        m2 = datetime(2026, 5, 10, 14, 37, 59, tzinfo=timezone.utc)
        assert new_eval_run_id(now=m1) == new_eval_run_id(now=m2)

    def test_naive_datetime_treated_as_utc(self):
        # Mirrors dates.now_dual semantics: naive inputs assumed UTC,
        # callers responsible for their own TZ awareness if not.
        naive = datetime(2026, 5, 10, 14, 37, 0)
        assert new_eval_run_id(now=naive) == "2605101437"

    def test_lexicographic_sort_yields_chronological(self):
        ids = [
            new_eval_run_id(now=datetime(2026, 5, 9, 23, 59, 0, tzinfo=timezone.utc)),
            new_eval_run_id(now=datetime(2026, 5, 10, 0, 0, 0, tzinfo=timezone.utc)),
            new_eval_run_id(now=datetime(2026, 5, 10, 14, 37, 0, tzinfo=timezone.utc)),
            new_eval_run_id(now=datetime(2026, 5, 10, 14, 38, 0, tzinfo=timezone.utc)),
            new_eval_run_id(now=datetime(2026, 12, 31, 23, 59, 0, tzinfo=timezone.utc)),
        ]
        assert sorted(ids) == ids, (
            f"YYMMDDHHMM should sort lexicographically into chronological "
            f"order; got {sorted(ids)} != expected {ids}"
        )

    def test_default_uses_now_utc(self):
        # No injected datetime → uses datetime.now(timezone.utc). Smoke
        # check that the result is parseable as YYMMDDHHMM and falls
        # within a reasonable window.
        rid = new_eval_run_id()
        assert len(rid) == 10
        assert rid.isdigit()
        # Year-prefix should be in [25, 99] for any realistic now() call
        # within the project's lifetime
        year_prefix = int(rid[:2])
        assert 25 <= year_prefix <= 99


class TestEvalArtifactKey:

    def test_default_basename_simplifies_to_run_id_dot_json(self):
        key = eval_artifact_key(
            prefix="predictor/variant_gates/triple_barrier",
            run_id="2605101437",
        )
        assert key == "predictor/variant_gates/triple_barrier/2605101437.json"

    def test_custom_basename_keeps_run_id_prefix(self):
        # Multi-file-per-run pipelines (eval-judge): per-file basename
        # gets the run_id prefix so files for one run group together
        # in path listings.
        key = eval_artifact_key(
            prefix="decision_artifacts/_eval",
            run_id="2605101437",
            basename="haiku_eval.json",
        )
        assert key == "decision_artifacts/_eval/2605101437_haiku_eval.json"

    def test_prefix_trailing_slash_normalized(self):
        key = eval_artifact_key(
            prefix="predictor/variant_gates/triple_barrier/",
            run_id="2605101437",
        )
        assert key == "predictor/variant_gates/triple_barrier/2605101437.json"

    def test_prefix_leading_slash_normalized(self):
        key = eval_artifact_key(
            prefix="/predictor/variant_gates/triple_barrier",
            run_id="2605101437",
        )
        assert key == "predictor/variant_gates/triple_barrier/2605101437.json"

    def test_composes_with_now_eval_run_id(self):
        moment = datetime(2026, 5, 10, 14, 37, 0, tzinfo=timezone.utc)
        rid = new_eval_run_id(now=moment)
        key = eval_artifact_key(
            prefix="predictor/variant_gates/triple_barrier",
            run_id=rid,
        )
        assert key == "predictor/variant_gates/triple_barrier/2605101437.json"

    def test_no_date_partition_in_path(self):
        # The flat layout is the institutional canonical form per
        # 2026-05-10 design discussion: YYMMDDHHMM run_id encodes the
        # date, so a {calendar_date}/ sub-partition would be pure
        # redundancy. This test pins the flat shape against future
        # well-meaning re-introduction of date partitioning.
        key = eval_artifact_key(prefix="x/y", run_id="2605101437")
        # Path is exactly two components after the prefix: x/y/run_id.json
        # (no x/y/2026-05-10/run_id.json shape)
        assert key.count("/") == 2
        assert "/2026-" not in key  # no ISO date sub-partition


class TestEvalLatestKey:

    def test_basic(self):
        assert (
            eval_latest_key("predictor/variant_gates/triple_barrier")
            == "predictor/variant_gates/triple_barrier/latest.json"
        )

    def test_trailing_slash_normalized(self):
        assert (
            eval_latest_key("predictor/variant_gates/triple_barrier/")
            == "predictor/variant_gates/triple_barrier/latest.json"
        )

    def test_filename_constant_exposed(self):
        # Constant is part of the public API so dashboards / scripts
        # can hard-code the filename without re-inventing it.
        assert EVAL_LATEST_FILENAME == "latest.json"
