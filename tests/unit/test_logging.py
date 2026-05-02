"""lyra_core.common.logging."""

from __future__ import annotations

import asyncio

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


# -----------------------------------------------------------------------------
# phase() / phase_sync() / context binding
# -----------------------------------------------------------------------------


def test_phase_emits_start_and_end_with_duration_ms(monkeypatch) -> None:
    """`phase()` must log a phase.start, then phase.end with duration_ms."""
    events: list[tuple[str, dict]] = []

    fake_logger = MagicMock()
    fake_logger.info = lambda evt, **fields: events.append((evt, fields))
    fake_logger.warning = lambda evt, **fields: events.append((evt, fields))

    monkeypatch.setattr(log_mod, "_phase_log", fake_logger)

    async def run():
        async with log_mod.phase("test.op", foo="bar"):
            await asyncio.sleep(0)

    asyncio.run(run())

    assert [e[0] for e in events] == ["phase.start", "phase.end"]
    start_fields = events[0][1]
    end_fields = events[1][1]
    assert start_fields["phase"] == "test.op"
    assert start_fields["foo"] == "bar"
    assert end_fields["phase"] == "test.op"
    assert end_fields["phase_ok"] is True
    assert isinstance(end_fields["duration_ms"], int)
    assert end_fields["duration_ms"] >= 0


def test_phase_records_failure_and_reraises(monkeypatch) -> None:
    events: list[tuple[str, dict]] = []
    fake_logger = MagicMock()
    fake_logger.info = lambda evt, **fields: events.append((evt, fields))
    fake_logger.warning = lambda evt, **fields: events.append((evt, fields))

    monkeypatch.setattr(log_mod, "_phase_log", fake_logger)

    async def run():
        async with log_mod.phase("test.crash"):
            raise ValueError("boom")

    try:
        asyncio.run(run())
    except ValueError as exc:
        assert str(exc) == "boom"
    else:
        raise AssertionError("phase did not re-raise")

    end_evt, end_fields = events[-1]
    assert end_evt == "phase.end"
    assert end_fields["phase_ok"] is False
    assert end_fields["error"] == "ValueError"


def test_bind_and_clear_job_context_use_contextvars(monkeypatch) -> None:
    """`bind_job_context` should set fields on structlog's contextvars
    bag; `clear_job_context` should reset them."""
    log_mod.clear_job_context()
    log_mod.bind_job_context(job_id="j-1", thread_id="t-1", custom="x")
    bag = structlog.contextvars.get_contextvars()
    assert bag.get("job_id") == "j-1"
    assert bag.get("thread_id") == "t-1"
    assert bag.get("custom") == "x"

    log_mod.clear_job_context()
    bag = structlog.contextvars.get_contextvars()
    assert "job_id" not in bag
    assert "thread_id" not in bag


def test_bind_job_context_drops_none_values() -> None:
    log_mod.clear_job_context()
    log_mod.bind_job_context(job_id="j-1", thread_id=None, tenant_id=None)
    bag = structlog.contextvars.get_contextvars()
    assert bag.get("job_id") == "j-1"
    assert "thread_id" not in bag
    assert "tenant_id" not in bag
    log_mod.clear_job_context()
