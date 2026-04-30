"""lyra_core.common.logging."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import structlog

from lyra_core.common import logging as log_mod
from lyra_core.common.logging import configure_logging, get_logger


def test_get_logger_returns_bound_logger() -> None:
    logger = get_logger("test")
    assert hasattr(logger, "info")
    assert hasattr(logger, "error")
    assert hasattr(logger, "warning")


def test_configure_logging_invokes_basicconfig_with_resolved_level(monkeypatch) -> None:
    captured = {}

    def fake_basic(**kw):
        captured.update(kw)

    monkeypatch.setattr(log_mod.logging, "basicConfig", fake_basic)
    monkeypatch.setattr(log_mod.structlog, "configure", lambda **kw: None)

    configure_logging(level="DEBUG", json_logs=False)
    assert captured["level"] == logging.DEBUG

    configure_logging(level="WARNING", json_logs=True)
    assert captured["level"] == logging.WARNING


def test_configure_logging_json_uses_json_renderer(monkeypatch) -> None:
    seen = {}

    def fake_configure(**kw):
        seen["procs"] = kw["processors"]

    monkeypatch.setattr(log_mod.structlog, "configure", fake_configure)
    monkeypatch.setattr(log_mod.logging, "basicConfig", lambda **kw: None)

    configure_logging(level="INFO", json_logs=True)
    last = seen["procs"][-1]
    assert isinstance(last, structlog.processors.JSONRenderer)


def test_configure_logging_dev_uses_console_renderer(monkeypatch) -> None:
    seen = {}

    def fake_configure(**kw):
        seen["procs"] = kw["processors"]

    monkeypatch.setattr(log_mod.structlog, "configure", fake_configure)
    monkeypatch.setattr(log_mod.logging, "basicConfig", lambda **kw: None)

    configure_logging(level="INFO", json_logs=False)
    last = seen["procs"][-1]
    assert isinstance(last, structlog.dev.ConsoleRenderer)


def test_configure_logging_invalid_level_falls_back_to_info(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(log_mod.logging, "basicConfig", lambda **kw: captured.update(kw))
    monkeypatch.setattr(log_mod.structlog, "configure", lambda **kw: None)
    configure_logging(level="NONSENSE", json_logs=False)
    assert captured["level"] == logging.INFO


def test_get_logger_with_no_name() -> None:
    assert get_logger() is not None
