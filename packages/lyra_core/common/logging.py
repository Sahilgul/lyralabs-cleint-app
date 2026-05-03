"""Structured JSON logging via structlog.

Production: JSON to stdout (Cloud Run / ACA picks it up).
Development: Pretty colored output.

Also exposes `phase()` -- a tiny context manager for stamping
phase.start/phase.end records with a duration_ms, so a single
end-to-end task can be reconstructed from one greppable `job_id`.

Use it like:

    async with phase("worker.tenant_lookup"):
        tenant = await lookup(...)

Output (pretty mode, when LOG_LEVEL allows debug):
    phase.start  name=worker.tenant_lookup
    phase.end    name=worker.tenant_lookup duration_ms=187
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from typing import Any

import structlog


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=log_level)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_logs:
        processors = [*shared_processors, structlog.processors.JSONRenderer()]
    else:
        # ConsoleRenderer formats exc_info itself; remove `format_exc_info` to
        # silence structlog's pretty-exceptions warning.
        dev_procs = [p for p in shared_processors if p is not structlog.processors.format_exc_info]
        processors = [*dev_procs, structlog.dev.ConsoleRenderer(colors=True)]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


# ---------------------------------------------------------------------------
# Phase timing
# ---------------------------------------------------------------------------

_phase_log = structlog.get_logger("phase")


@asynccontextmanager
async def phase(name: str, **fields: Any):
    """Async context manager that emits start/end records with duration_ms.

    Pair with `bind_job_context()` so every line for a job carries the
    job_id / thread_id / tenant_id automatically; that way a single
    `grep job_id=<uuid>` reconstructs the full timeline.

    On exception we still emit phase.end with `error` and `duration_ms`
    so a crashed phase is observable, then re-raise.

    The success flag on phase.end is named `phase_ok` (not `ok`) so it
    doesn't collide if the caller passes their own `ok` field as
    domain context (e.g. tool execution success).
    """
    started = time.perf_counter()
    _phase_log.info("phase.start", phase=name, **fields)
    try:
        yield
    except BaseException as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        _phase_log.warning(
            "phase.end",
            phase=name,
            duration_ms=elapsed_ms,
            phase_ok=False,
            error=type(exc).__name__,
            **fields,
        )
        raise
    else:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        _phase_log.info(
            "phase.end",
            phase=name,
            duration_ms=elapsed_ms,
            phase_ok=True,
            **fields,
        )


@contextmanager
def phase_sync(name: str, **fields: Any):
    """Synchronous twin of `phase()` for non-async code paths."""
    started = time.perf_counter()
    _phase_log.info("phase.start", phase=name, **fields)
    try:
        yield
    except BaseException as exc:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        _phase_log.warning(
            "phase.end",
            phase=name,
            duration_ms=elapsed_ms,
            phase_ok=False,
            error=type(exc).__name__,
            **fields,
        )
        raise
    else:
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        _phase_log.info(
            "phase.end",
            phase=name,
            duration_ms=elapsed_ms,
            phase_ok=True,
            **fields,
        )


def bind_job_context(
    *,
    job_id: str | None = None,
    thread_id: str | None = None,
    tenant_id: str | None = None,
    **extra: Any,
) -> None:
    """Bind per-task fields onto every structlog record on this loop.

    Call once at task entry; every subsequent `phase()` and any
    `log.info(...)` from any module will pick up these fields without
    callers having to pass them through.

    Wraps `structlog.contextvars.bind_contextvars` -- you must clear them
    yourself at task exit (or rely on the prefork worker recycling the
    context). The ContextVar is async-safe across the asyncio loop the
    Celery task runs on.
    """
    pairs = {
        k: v
        for k, v in {
            "job_id": job_id,
            "thread_id": thread_id,
            "tenant_id": tenant_id,
            **extra,
        }.items()
        if v is not None
    }
    structlog.contextvars.bind_contextvars(**pairs)


def clear_job_context() -> None:
    """Reset the contextvars bound by `bind_job_context()`.

    Call from a task's `finally:` so logs from later tasks don't inherit
    the previous task's job_id (especially under prefork worker concurrency).
    """
    structlog.contextvars.clear_contextvars()
