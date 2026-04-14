"""Unit tests for shared logging setup."""

from __future__ import annotations

import json
import logging
import os
from unittest import mock

import pytest

from alpha_engine_lib.logging import (
    JSONFormatter,
    get_flow_doctor,
    setup_logging,
)


@pytest.fixture(autouse=True)
def _reset_root_logger():
    """Every test starts with a clean root logger."""
    yield
    logging.getLogger().handlers.clear()
    # Reset the module-level singleton so tests don't bleed state.
    import alpha_engine_lib.logging as m
    m._fd_instance = None


# ── JSONFormatter ────────────────────────────────────────────────────────


def test_json_formatter_basic():
    fmt = JSONFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__,
        lineno=1, msg="hello %s", args=("world",), exc_info=None,
    )
    out = json.loads(fmt.format(record))
    assert out["level"] == "INFO"
    assert out["msg"] == "hello world"
    assert "ts" in out


def test_json_formatter_with_exception():
    fmt = JSONFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys
        record = logging.LogRecord(
            name="test", level=logging.ERROR, pathname=__file__,
            lineno=1, msg="failed", args=(), exc_info=sys.exc_info(),
        )
    out = json.loads(fmt.format(record))
    assert "exc" in out
    assert "ValueError" in out["exc"]


def test_json_formatter_with_ctx():
    fmt = JSONFormatter()
    record = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__,
        lineno=1, msg="hi", args=(), exc_info=None,
    )
    record.ctx = {"ticker": "SPY", "qty": 10}
    out = json.loads(fmt.format(record))
    assert out["ctx"] == {"ticker": "SPY", "qty": 10}


# ── setup_logging ────────────────────────────────────────────────────────


def test_setup_logging_text_mode_default(monkeypatch):
    monkeypatch.delenv("ALPHA_ENGINE_JSON_LOGS", raising=False)
    monkeypatch.delenv("FLOW_DOCTOR_ENABLED", raising=False)
    setup_logging("test")
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert not isinstance(root.handlers[0].formatter, JSONFormatter)


def test_setup_logging_json_mode(monkeypatch):
    monkeypatch.setenv("ALPHA_ENGINE_JSON_LOGS", "1")
    monkeypatch.delenv("FLOW_DOCTOR_ENABLED", raising=False)
    setup_logging("test")
    root = logging.getLogger()
    assert isinstance(root.handlers[0].formatter, JSONFormatter)


def test_setup_logging_clears_existing_handlers(monkeypatch):
    """setup_logging replaces whatever handlers were previously attached
    (pytest's LogCaptureHandler, other loggers, etc.) with exactly one
    fresh StreamHandler."""
    monkeypatch.delenv("FLOW_DOCTOR_ENABLED", raising=False)
    root = logging.getLogger()
    root.addHandler(logging.StreamHandler())
    root.addHandler(logging.StreamHandler())
    setup_logging("test")
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.StreamHandler)


# ── Flow Doctor attach ───────────────────────────────────────────────────


def test_flow_doctor_disabled_by_default(monkeypatch):
    monkeypatch.delenv("FLOW_DOCTOR_ENABLED", raising=False)
    setup_logging("test")
    assert get_flow_doctor() is None


def test_flow_doctor_enabled_without_yaml_raises(monkeypatch):
    monkeypatch.setenv("FLOW_DOCTOR_ENABLED", "1")
    with pytest.raises(RuntimeError, match="not given a flow_doctor_yaml path"):
        setup_logging("test")


def test_flow_doctor_enabled_missing_yaml_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOW_DOCTOR_ENABLED", "1")
    missing = tmp_path / "nope.yaml"
    # Even if flow_doctor is installed, the missing yaml must raise before
    # we try to init. Test that path.
    fake_module = mock.Mock()
    fake_module.init = mock.Mock(return_value=mock.Mock())
    fake_module.FlowDoctorHandler = mock.Mock(return_value=logging.NullHandler())
    with mock.patch.dict("sys.modules", {"flow_doctor": fake_module}):
        with pytest.raises(RuntimeError, match="flow-doctor config not found"):
            setup_logging("test", flow_doctor_yaml=str(missing))


def test_flow_doctor_enabled_missing_package_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOW_DOCTOR_ENABLED", "1")
    yaml_path = tmp_path / "flow-doctor.yaml"
    yaml_path.write_text("flow_name: test\n")
    # Simulate flow_doctor not installed.
    with mock.patch.dict("sys.modules", {"flow_doctor": None}):
        with pytest.raises(RuntimeError, match="flow-doctor is not installed"):
            setup_logging("test", flow_doctor_yaml=str(yaml_path))


def test_flow_doctor_enabled_happy_path(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOW_DOCTOR_ENABLED", "1")
    yaml_path = tmp_path / "flow-doctor.yaml"
    yaml_path.write_text("flow_name: test\n")

    fake_instance = mock.Mock()
    fake_module = mock.Mock()
    fake_module.init = mock.Mock(return_value=fake_instance)
    fake_module.FlowDoctorHandler = mock.Mock(return_value=logging.NullHandler())

    with mock.patch.dict("sys.modules", {"flow_doctor": fake_module}):
        setup_logging("test", flow_doctor_yaml=str(yaml_path))

    assert get_flow_doctor() is fake_instance
    fake_module.init.assert_called_once_with(config_path=str(yaml_path))
    # No exclude_patterns passed → kwarg must be absent so we don't
    # silently override the FlowDoctorHandler default.
    _, kwargs = fake_module.FlowDoctorHandler.call_args
    assert "exclude_patterns" not in kwargs


def test_flow_doctor_exclude_patterns_forwarded(monkeypatch, tmp_path):
    """exclude_patterns reaches FlowDoctorHandler when provided.

    Context (2026-04-14): executor suppresses benign IB Error 10197
    alerts via this path. The kwarg must pass through unchanged.
    """
    monkeypatch.setenv("FLOW_DOCTOR_ENABLED", "1")
    yaml_path = tmp_path / "flow-doctor.yaml"
    yaml_path.write_text("flow_name: test\n")

    fake_module = mock.Mock()
    fake_module.init = mock.Mock(return_value=mock.Mock())
    fake_module.FlowDoctorHandler = mock.Mock(return_value=logging.NullHandler())

    patterns = [r"Error 10197", r"benign warning"]
    with mock.patch.dict("sys.modules", {"flow_doctor": fake_module}):
        setup_logging("test", flow_doctor_yaml=str(yaml_path), exclude_patterns=patterns)

    _, kwargs = fake_module.FlowDoctorHandler.call_args
    assert kwargs.get("exclude_patterns") == patterns


def test_flow_doctor_empty_exclude_patterns_not_forwarded(monkeypatch, tmp_path):
    """Empty/None exclude_patterns must not set the kwarg — preserves
    the FlowDoctorHandler default rather than forcing it to ``[]``."""
    monkeypatch.setenv("FLOW_DOCTOR_ENABLED", "1")
    yaml_path = tmp_path / "flow-doctor.yaml"
    yaml_path.write_text("flow_name: test\n")

    fake_module = mock.Mock()
    fake_module.init = mock.Mock(return_value=mock.Mock())
    fake_module.FlowDoctorHandler = mock.Mock(return_value=logging.NullHandler())

    with mock.patch.dict("sys.modules", {"flow_doctor": fake_module}):
        setup_logging("test", flow_doctor_yaml=str(yaml_path), exclude_patterns=[])

    _, kwargs = fake_module.FlowDoctorHandler.call_args
    assert "exclude_patterns" not in kwargs
