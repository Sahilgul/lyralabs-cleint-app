"""Structured JSON logging via structlog.

Production: JSON to stdout (Cloud Run / ACA picks it up).
Development: Pretty colored output.
"""

from __future__ import annotations

import logging
import sys
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
