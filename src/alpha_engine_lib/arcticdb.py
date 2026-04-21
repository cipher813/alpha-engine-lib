"""
ArcticDB helpers: uniform library-open path + common read patterns.

Centralizes the ``adb.Arctic(uri).get_library("...")`` boilerplate that was
duplicated across predictor, research, backtester, data, and executor. Every
site was constructing the same S3 URI string by hand — one escape bug in
that string (path_prefix= query param collapsing under shell double-quote
interpolation) surfaced 2026-04-21 during the SNDK incident.

Using this module guarantees that:

- The S3 URI format stays consistent everywhere (single source of truth).
- Library-open failures raise a uniform ``RuntimeError`` with bucket
  context, so downstream errors have a consistent shape.
- ``arcticdb`` is imported lazily inside each function, so this module
  stays importable on consumers that don't install the ``[arcticdb]``
  optional extra (e.g. lightweight CLI tools that only use the logging
  submodule).

Requires the ``arcticdb`` optional extra
(``alpha-engine-lib[arcticdb]``) to actually call any function here.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from arcticdb.version_store.library import Library

log = logging.getLogger(__name__)

# Library name constants — these match what every alpha-engine module uses.
# Centralized so a rename propagates from one place.
UNIVERSE_LIB = "universe"
MACRO_LIB = "macro"


def arctic_uri(bucket: str, *, region: str | None = None) -> str:
    """Return the canonical ArcticDB S3 URI for ``bucket``.

    Format: ``s3s://s3.{region}.amazonaws.com:{bucket}?path_prefix=arcticdb&aws_auth=true``

    ``region`` defaults to ``AWS_REGION`` env var, then ``us-east-1``.
    """
    region = region or os.environ.get("AWS_REGION", "us-east-1")
    return (
        f"s3s://s3.{region}.amazonaws.com:{bucket}"
        "?path_prefix=arcticdb&aws_auth=true"
    )


def _import_arcticdb():
    """Lazy import helper with a uniform error message."""
    try:
        import arcticdb as adb
    except ImportError as exc:
        raise RuntimeError(
            "arcticdb is not importable — install "
            "alpha-engine-lib[arcticdb] or add arcticdb to the deploy "
            f"image: {exc}"
        ) from exc
    return adb


def open_arctic(bucket: str, *, region: str | None = None):
    """Return an ``arcticdb.Arctic`` instance pointed at ``bucket``.

    Raises ``RuntimeError`` if ``arcticdb`` is not installed.
    """
    adb = _import_arcticdb()
    return adb.Arctic(arctic_uri(bucket, region=region))


def open_universe_lib(bucket: str, *, region: str | None = None) -> "Library":
    """Open the ``universe`` library on ``bucket``.

    Raises ``RuntimeError`` on any library-open failure, with bucket and
    URI in the message so the operator can see which endpoint is failing.
    """
    arctic = open_arctic(bucket, region=region)
    try:
        return arctic.get_library(UNIVERSE_LIB)
    except Exception as exc:
        raise RuntimeError(
            f"ArcticDB {UNIVERSE_LIB!r} library open failed on bucket "
            f"{bucket!r} (uri={arctic_uri(bucket, region=region)}): {exc}"
        ) from exc


def open_macro_lib(bucket: str, *, region: str | None = None) -> "Library":
    """Open the ``macro`` library on ``bucket``.

    Raises ``RuntimeError`` on any library-open failure.
    """
    arctic = open_arctic(bucket, region=region)
    try:
        return arctic.get_library(MACRO_LIB)
    except Exception as exc:
        raise RuntimeError(
            f"ArcticDB {MACRO_LIB!r} library open failed on bucket "
            f"{bucket!r} (uri={arctic_uri(bucket, region=region)}): {exc}"
        ) from exc


def get_universe_symbols(bucket: str, *, region: str | None = None) -> set[str]:
    """Return the set of symbols currently present in the universe library.

    Common use case: filtering tickers against "what's actually in
    ArcticDB right now" before passing to downstream code that hard-fails
    on missing symbols (e.g. the executor's load_daily_vwap / load_atr_14_pct
    guards, or the backtester's simulate replay of historical signals).

    Raises ``RuntimeError`` on library-open or list failure — an
    ArcticDB health problem is a pipeline-level precondition, not
    something to silently paper over with an empty set.
    """
    lib = open_universe_lib(bucket, region=region)
    try:
        symbols = set(lib.list_symbols())
    except Exception as exc:
        raise RuntimeError(
            f"ArcticDB {UNIVERSE_LIB}.list_symbols() failed on bucket "
            f"{bucket!r}: {exc}"
        ) from exc
    log.info("ArcticDB %s symbols available: %d", UNIVERSE_LIB, len(symbols))
    return symbols
