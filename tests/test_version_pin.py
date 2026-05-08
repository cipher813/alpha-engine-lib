"""Pin ``alpha_engine_lib.__version__`` to ``pyproject.toml::version``.

Doc-string drift: 2026-05-08 we shipped v0.5.6 by bumping
``pyproject.toml::version`` 0.5.5 → 0.5.6 but forgot to bump
``src/alpha_engine_lib/__init__.py::__version__``. The wheel built
from v0.5.6 had package metadata 0.5.6 but the runtime
``__version__`` string lagged at 0.5.5 — confusing for any consumer
that reads the runtime attribute (operator scripts, dashboards,
log lines).

Functional impact: zero (load_inventory + every other code path
reads the YAML or the actual code, not the version string).
But the drift makes "what version is deployed?" harder to answer.

This test pins the two together so future bumps that miss one
side fail at CI.
"""

from __future__ import annotations

import re
from pathlib import Path

import alpha_engine_lib


_PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


def _pyproject_version() -> str:
    """Read ``version = "X.Y.Z"`` from pyproject.toml.

    Avoid a tomllib import (3.11+) — kept stdlib-free + Python-3.9-safe
    via a single regex match. The line we want is the only top-level
    ``version = "..."`` in the [project] block; the file ships one of
    these per release.
    """
    text = _PYPROJECT.read_text()
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    assert match is not None, "version not found in pyproject.toml"
    return match.group(1)


def test_init_version_matches_pyproject():
    """Lockstep pin: bumping one without the other fails here.

    If you're updating this test to a new version, you almost
    certainly want to bump BOTH ``pyproject.toml`` and
    ``src/alpha_engine_lib/__init__.py`` in the same commit. Search
    for ``__version__`` and ``version =`` to find them.
    """
    assert alpha_engine_lib.__version__ == _pyproject_version(), (
        f"alpha_engine_lib.__version__={alpha_engine_lib.__version__!r} "
        f"!= pyproject.toml::version={_pyproject_version()!r} — bump both "
        f"in lockstep or this test fails. Doc-string drift cause was a "
        f"pyproject-only bump on 2026-05-08 (v0.5.6) that left __init__.py "
        f"saying 0.5.5 for ~one day."
    )
