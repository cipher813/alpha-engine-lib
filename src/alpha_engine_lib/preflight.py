"""
Preflight: fast fail-fast connectivity + freshness checks.

``BasePreflight`` provides the shared primitives; consumer modules
subclass it and override ``run()`` to compose a module-specific check
sequence. The base raises ``RuntimeError`` on any failure — consumers
catch nothing, so the raise propagates up through ``main()`` → non-zero
exit → the orchestration layer's failure handler.

Design context (2026-04-14): the alpha-engine-data DailyData step
silently ran against a stale ArcticDB universe library for two
weekdays because an ``ImportError`` on ``arcticdb`` was caught at debug
level. A freshness check on SPY would have flagged the outage in ~1s.
Preflight exists to catch that class of failure *before* spending 30
minutes on real work.

Scope is deliberately narrow: **external-world handshakes only** (env
vars, S3 reachability, ArcticDB symbol freshness). Data-correctness
hard-fails still live in the hardened collectors themselves.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

log = logging.getLogger(__name__)


class BasePreflight:
    """Shared preflight primitives.

    Subclass and override :meth:`run` to compose a module-specific
    check sequence. Each primitive raises :class:`RuntimeError` on
    failure with an explanatory message that includes what was checked
    and what went wrong.
    """

    def __init__(self, bucket: str, region: str | None = None):
        if not bucket:
            raise ValueError("BasePreflight: bucket is required")
        self.bucket = bucket
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")

    # ── Composition entry point ──────────────────────────────────────────

    def run(self) -> None:
        """Execute the preflight check sequence.

        Subclasses override this to compose primitives. The default
        raises to prevent a misuse where a subclass forgets to override
        and silently passes.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override run() to compose preflight checks"
        )

    # ── Primitives ───────────────────────────────────────────────────────

    def check_env_vars(self, *names: str) -> None:
        """Raise if any of the given env vars are unset or empty."""
        missing = [n for n in names if not os.environ.get(n)]
        if missing:
            raise RuntimeError(f"Pre-flight: required env vars missing: {missing}")

    def check_s3_bucket(self) -> None:
        """Raise if the configured bucket is not reachable (auth, network, or missing)."""
        import boto3
        try:
            boto3.client("s3").head_bucket(Bucket=self.bucket)
        except Exception as exc:
            raise RuntimeError(
                f"Pre-flight: S3 bucket {self.bucket!r} unreachable: {exc}"
            ) from exc

    def check_s3_key(self, key: str, max_age_days: int | None = None) -> None:
        """Raise if ``s3://{bucket}/{key}`` is missing or older than ``max_age_days``.

        ``max_age_days=None`` disables the freshness check — existence only.
        """
        import boto3
        from botocore.exceptions import ClientError
        try:
            head = boto3.client("s3").head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            err_code = exc.response.get("Error", {}).get("Code")
            if err_code in ("404", "NoSuchKey"):
                raise RuntimeError(
                    f"Pre-flight: S3 key s3://{self.bucket}/{key} does not exist"
                ) from exc
            raise RuntimeError(
                f"Pre-flight: S3 key s3://{self.bucket}/{key} unreachable: {exc}"
            ) from exc
        if max_age_days is not None:
            last_modified = head["LastModified"]
            age_days = (datetime.now(timezone.utc) - last_modified).days
            if age_days > max_age_days:
                raise RuntimeError(
                    f"Pre-flight: S3 key s3://{self.bucket}/{key} is "
                    f"{age_days} days stale (threshold {max_age_days})"
                )

    def check_arcticdb_fresh(
        self,
        library: str,
        symbol: str,
        max_stale_days: int,
    ) -> None:
        """Raise if ``arcticdb`` is unavailable, the library/symbol is
        unreadable, or the last date in ``symbol`` is older than
        ``max_stale_days`` calendar days from today (UTC).

        Requires the ``arcticdb`` optional extra
        (``alpha-engine-lib[arcticdb]``).
        """
        try:
            import arcticdb as adb
            import pandas as pd
        except ImportError as exc:
            raise RuntimeError(
                "Pre-flight: arcticdb not importable — install "
                "alpha-engine-lib[arcticdb] or add arcticdb to the deploy image: "
                f"{exc}"
            ) from exc

        uri = (
            f"s3s://s3.{self.region}.amazonaws.com:{self.bucket}"
            "?path_prefix=arcticdb&aws_auth=true"
        )
        try:
            lib = adb.Arctic(uri).get_library(library)
        except Exception as exc:
            raise RuntimeError(
                f"Pre-flight: ArcticDB library {library!r} unreachable "
                f"at {uri}: {exc}"
            ) from exc

        try:
            df = lib.read(symbol).data
        except Exception as exc:
            raise RuntimeError(
                f"Pre-flight: ArcticDB {library}/{symbol} read failed: {exc}"
            ) from exc

        if df.empty:
            raise RuntimeError(
                f"Pre-flight: ArcticDB {library}/{symbol} is empty"
            )

        last_ts = pd.Timestamp(df.index[-1])
        # Normalize to tz-naive date for comparison against today's UTC date.
        if last_ts.tzinfo is not None:
            last_ts = last_ts.tz_convert("UTC").tz_localize(None)
        today = pd.Timestamp(datetime.now(timezone.utc).date())
        age_days = (today - last_ts.normalize()).days
        if age_days > max_stale_days:
            raise RuntimeError(
                f"Pre-flight: ArcticDB {library}/{symbol} last date "
                f"{last_ts.date()} is {age_days} days stale "
                f"(threshold {max_stale_days})"
            )

    def check_ib_paper_account(self, account_id: str) -> None:
        """Raise if ``account_id`` doesn't start with 'D' (IBKR paper prefix).

        Defensive check for the executor — prevents live credentials
        leaking into a paper-trading run (or vice versa).
        """
        if not account_id:
            raise RuntimeError("Pre-flight: IB account_id is empty")
        if not account_id.startswith("D"):
            raise RuntimeError(
                f"Pre-flight: IB account_id {account_id!r} is not a paper "
                "account (paper accounts start with 'D')"
            )
