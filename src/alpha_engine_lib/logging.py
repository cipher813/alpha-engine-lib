"""
Shared structured logging + Flow Doctor integration.

Replaces near-identical copies of ``log_config.py`` in alpha-engine-data
and alpha-engine/executor. Consumers call :func:`setup_logging` once at
process startup; subsequent call sites retrieve the Flow Doctor instance
via :func:`get_flow_doctor`.

Modes:

- Text (default): human-readable single-line log format.
- JSON: activated by ``ALPHA_ENGINE_JSON_LOGS=1``. Emits one JSON object
  per log record, including tracebacks for errors.

Flow Doctor activates only when ``FLOW_DOCTOR_ENABLED=1`` and a
``flow_doctor_yaml`` path is provided. ERROR-level records (including
``logger.exception``) fire the FlowDoctorHandler, which dispatches per
the yaml config (email + GitHub issue with dedup + rate limits).

Requires the ``flow_doctor`` optional extra when FLOW_DOCTOR_ENABLED=1
(``alpha-engine-lib[flow_doctor]``).
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

# Singleton populated by setup_logging() when FLOW_DOCTOR_ENABLED=1.
# ``Optional[object]`` typing avoids forcing a flow_doctor import here.
_fd_instance: Optional[object] = None


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "func": record.funcName,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exc"] = self.formatException(record.exc_info)
        if hasattr(record, "ctx"):
            entry["ctx"] = record.ctx
        return json.dumps(entry, default=str)


def get_flow_doctor():
    """Return the shared flow-doctor instance, or None if not initialized."""
    return _fd_instance


def _attach_flow_doctor(
    yaml_path: str,
    exclude_patterns: list[str] | None = None,
) -> None:
    """Initialize the shared flow-doctor instance and attach a log handler.

    ``exclude_patterns`` is a list of regex strings forwarded to
    ``FlowDoctorHandler(exclude_patterns=...)``. Log records whose
    rendered message matches any pattern are dropped before entering
    the flow-doctor dispatch pipeline (email / GitHub issue). Use for
    benign ERROR-level noise that would otherwise dedup-spam on-call.
    """
    global _fd_instance
    try:
        import flow_doctor
    except ImportError as exc:
        raise RuntimeError(
            "FLOW_DOCTOR_ENABLED=1 but flow-doctor is not installed. Install "
            "via alpha-engine-lib[flow_doctor] or add flow-doctor[diagnosis] "
            f"to requirements: {exc}"
        ) from exc

    if not os.path.exists(yaml_path):
        raise RuntimeError(
            f"FLOW_DOCTOR_ENABLED=1 but flow-doctor config not found at {yaml_path}"
        )

    _fd_instance = flow_doctor.init(config_path=yaml_path)
    handler_kwargs: dict = {"level": logging.ERROR}
    if exclude_patterns:
        handler_kwargs["exclude_patterns"] = exclude_patterns
    handler = flow_doctor.FlowDoctorHandler(_fd_instance, **handler_kwargs)
    logging.getLogger().addHandler(handler)


def setup_logging(
    name: str,
    flow_doctor_yaml: str | None = None,
    exclude_patterns: list[str] | None = None,
) -> None:
    """Configure the root logger for an Alpha Engine entrypoint.

    :param name: Logger name shown in the text-mode prefix
        (``"%(asctime)s %(levelname)s [{name}] %(message)s"``). Typically
        the module name (``"data-collector"``, ``"executor"``, etc.).
    :param flow_doctor_yaml: Absolute or CWD-relative path to the
        flow-doctor yaml config. Required if ``FLOW_DOCTOR_ENABLED=1``;
        ignored otherwise.
    :param exclude_patterns: Optional list of regex strings. When
        ``FLOW_DOCTOR_ENABLED=1``, these are forwarded to
        ``FlowDoctorHandler`` so matching ERROR-level records are
        dropped before the flow-doctor dispatch pipeline. Use sparingly
        — this silences *alerts*, not logs. The records still appear in
        stdout / JSON logs; only flow-doctor's email + GitHub issue
        routing is suppressed. Example: the executor passes
        ``[r"Error 10197"]`` to suppress benign IB Gateway noise when
        the iOS app steals the live-data session.

    Env vars consulted:

    - ``ALPHA_ENGINE_JSON_LOGS`` — ``"1"`` enables JSON formatter.
    - ``FLOW_DOCTOR_ENABLED`` — ``"1"`` attaches FlowDoctorHandler.
    """
    json_mode = os.environ.get("ALPHA_ENGINE_JSON_LOGS", "0") == "1"

    handler = logging.StreamHandler()
    if json_mode:
        handler.setFormatter(JSONFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            f"%(asctime)s %(levelname)s [{name}] %(message)s"
        ))

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    if os.environ.get("FLOW_DOCTOR_ENABLED", "0") == "1":
        if not flow_doctor_yaml:
            raise RuntimeError(
                "FLOW_DOCTOR_ENABLED=1 but setup_logging() was not given a "
                "flow_doctor_yaml path"
            )
        _attach_flow_doctor(flow_doctor_yaml, exclude_patterns=exclude_patterns)
