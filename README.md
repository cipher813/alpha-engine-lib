# alpha-engine-lib

> Part of [**Nous Ergon**](https://nousergon.ai) — Autonomous Multi-Agent Trading System. Repo and S3 names use the underlying project name `alpha-engine`.

[![Part of Nous Ergon](https://img.shields.io/badge/Part_of-Nous_Ergon-1a73e8?style=flat-square)](https://nousergon.ai)
[![Python](https://img.shields.io/badge/python-3.11+-blue?style=flat-square)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)
[![Phase 2 · Reliability](https://img.shields.io/badge/Phase_2-Reliability-e9c46a?style=flat-square)](https://github.com/cipher813/alpha-engine-docs#phase-trajectory)

Shared utility library used by all 6 modules of Nous Ergon. Cross-cutting concerns only — logging, freshness checks, trading-calendar arithmetic, ArcticDB helpers, agent-decision capture, LLM cost tracking. No proprietary trading logic, no model weights, no agent prompts.

The lib's job is to keep the same code from being maintained six times.

---

## Install

```
# requirements.txt
alpha-engine-lib @ git+https://github.com/cipher813/alpha-engine-lib@v0.2.4
```

Tagged releases: `v0.1.0`, `v0.2.0`, `v0.2.4`, etc. Consumers pin to a specific tag. Breaking changes bump the minor version while Alpha Engine is in pre-1.0.

```bash
# With optional extras
pip install "alpha-engine-lib[arcticdb] @ git+https://github.com/cipher813/alpha-engine-lib@v0.2.4"
```

| Extra | Pulls in | When you need it |
|---|---|---|
| `[arcticdb]` | `arcticdb`, `pandas` | Anything that calls `check_arcticdb_fresh` or the ArcticDB read/write helpers |
| `[flow_doctor]` | `flow-doctor` | Logging integration that escalates ERROR-level events to flow-doctor |
| `[dev]` | `pytest`, lint tooling | Local development |

## Modules

### `logging` — structured logging + flow-doctor attach

Replaces the near-identical `log_config.py` copies that used to live in alpha-engine-data and alpha-engine-executor. Consumers call `setup_logging` once at process startup:

```python
from alpha_engine_lib.logging import setup_logging

setup_logging("data-collector", flow_doctor_yaml="/path/to/flow-doctor.yaml")
```

- Text mode by default; JSON via `ALPHA_ENGINE_JSON_LOGS=1`
- Flow-doctor attaches as an ERROR-level handler when `FLOW_DOCTOR_ENABLED=1` (requires `[flow_doctor]` extra)

### `preflight` — fail-fast connectivity + freshness checks

Runs at the top of every entrypoint, before any real work starts. Primitives live on `BasePreflight`; each consumer subclasses and overrides `run()`:

```python
from alpha_engine_lib.preflight import BasePreflight

class DataPreflight(BasePreflight):
    def __init__(self, bucket, mode):
        super().__init__(bucket)
        self.mode = mode

    def run(self):
        self.check_env_vars("AWS_REGION")
        if self.mode == "phase1":
            self.check_env_vars("FRED_API_KEY", "POLYGON_API_KEY")
        self.check_s3_bucket()
        if self.mode == "daily":
            self.check_arcticdb_fresh("universe", "SPY", max_stale_days=4)
```

Failed checks raise `RuntimeError` with an explanatory message. Consumers catch nothing — the raise propagates up through `main()` → non-zero exit → Step Function `HandleFailure` → flow-doctor notification. The point is to fail before paying for any LLM calls or downstream work.

### `arcticdb` — read/write helpers + symbol enumeration

Wrappers around the ArcticDB Python client. Standardizes the URI format, library naming, and read paths so each consumer doesn't reinvent the connection logic.

### `dates` — trading-day arithmetic

`now_dual()` returns a `(calendar_date, trading_day)` pair following the rule `trading_day = last_closed_trading_day(now)`. Strictly backward-looking; never ahead. `session_for_timestamp(ts)` resolves any timestamp to its trading session. Used at every artifact-write site to prevent calendar/trading-day drift between modules.

### `trading_calendar` — NYSE holiday detection

Pure-Python NYSE calendar through 2030. No `pandas-market-calendars` dependency.

### `decision_capture` — agent decision audit logger

Captures every agent decision as a structured artifact: prompt metadata (id + version), input snapshot, agent output, and cost. Each decision becomes replayable, auditable, and attributable to a specific prompt revision. Backbone of the Phase 2 measurement substrate.

### `cost` — LLM cost tracking

Token-aware cost computation following Anthropic's prompt-caching semantics (cache-write vs cache-read pricing). Used by every LLM call site to attach a `cost_usd` to its output.

## How it's used

All six Nous Ergon module repos depend on this lib:

| Module | Repo | What it imports from here |
|---|---|---|
| Data | [`alpha-engine-data`](https://github.com/cipher813/alpha-engine-data) | `logging`, `preflight`, `arcticdb`, `dates`, `trading_calendar` |
| Research | [`alpha-engine-research`](https://github.com/cipher813/alpha-engine-research) | `logging`, `decision_capture`, `cost`, `dates` |
| Predictor | [`alpha-engine-predictor`](https://github.com/cipher813/alpha-engine-predictor) | `logging`, `preflight`, `arcticdb`, `dates` |
| Executor | [`alpha-engine`](https://github.com/cipher813/alpha-engine) | `logging`, `preflight`, `arcticdb`, `dates`, `trading_calendar` |
| Backtester | [`alpha-engine-backtester`](https://github.com/cipher813/alpha-engine-backtester) | `logging`, `preflight`, `arcticdb`, `dates` |
| Dashboard | [`alpha-engine-dashboard`](https://github.com/cipher813/alpha-engine-dashboard) | `logging`, `arcticdb`, `dates` |

## Development

```bash
git clone https://github.com/cipher813/alpha-engine-lib.git
cd alpha-engine-lib
pip install -e ".[dev,arcticdb,flow_doctor]"
pytest
```

## Scope discipline

This repo is intentionally narrow. Code lands here when **at least two consumers would otherwise maintain their own copy**. New modules land as their own minor release with per-consumer adoption — no lockstep updates.

Code that **does not** belong here:
- Anything tunable (scoring weights, risk thresholds, sizing parameters) → `alpha-engine-config` (private)
- Agent prompts → `alpha-engine-config` (private)
- Module-specific business logic → that module's repo

## License

MIT — see [LICENSE](LICENSE).
