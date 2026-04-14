# alpha-engine-lib

Shared utilities for Alpha Engine modules.

Private repo — consumers install via pip + git URL:

```
# requirements.txt
alpha-engine-lib @ git+https://github.com/cipher813/alpha-engine-lib@v0.1.0
```

Git credentials flow through the existing `~/.netrc` PAT used by other private Alpha Engine repos on EC2 / Docker / Lambda.

## Modules

### `logging` — structured logging + flow-doctor attach

Replaces the near-identical `log_config.py` copies in alpha-engine-data and alpha-engine/executor. Consumers call `setup_logging` once at process startup:

```python
from alpha_engine_lib.logging import setup_logging

setup_logging("data-collector", flow_doctor_yaml="/path/to/flow-doctor.yaml")
```

- Text mode by default; JSON via `ALPHA_ENGINE_JSON_LOGS=1`.
- Flow Doctor attaches as an ERROR-level handler when `FLOW_DOCTOR_ENABLED=1`. Requires `alpha-engine-lib[flow_doctor]`.

### `preflight` — fast fail-fast connectivity + freshness checks

Intended to run at the top of every entrypoint, before any real work starts. Primitives live on `BasePreflight`; each consumer subclasses and overrides `run()`:

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

The base class raises `RuntimeError` with an explanatory message on any failed check. Consumers catch nothing — the raise propagates up through `main()` → non-zero exit → Step Function `HandleFailure` → flow-doctor notification.

### Optional extras

`alpha-engine-lib[arcticdb]` pulls in `arcticdb` + `pandas` for `check_arcticdb_fresh`. Consumers that don't use ArcticDB (e.g., researchers who only need S3 + env checks) can install the base package without the extra.

## Development

```bash
pip install -e .[dev,arcticdb]
pytest
```

## Versioning

Tagged releases: `v0.1.0`, `v0.2.0`, etc. Consumer pins via `@v0.1.0` in their `requirements.txt`. Breaking changes bump the minor version while Alpha Engine is in pre-1.0.

## Scope

This repo is intentionally narrow — it only holds code that is genuinely duplicated across multiple Alpha Engine consumer repos. New modules land here when at least two consumers would otherwise maintain their own copy.

Candidates tracked for future accretion:
- `log_config` — JSONFormatter + `setup_logging` + flow-doctor attach (duplicated in data + executor today).
- `ssm_secrets` — SSM Parameter Store secret loader.
- `config_loader` — search-path YAML loader (alpha-engine-config → local → example).

Each lands as its own minor release with per-consumer adoption — no lockstep updates.
