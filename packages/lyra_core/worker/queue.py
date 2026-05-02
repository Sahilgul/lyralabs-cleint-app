"""arq enqueue helpers for channel adapters.

Lazily opens an arq pool on first use and keeps it alive for the
life of the process (Cloud Run API container). Channel adapters call
`enqueue_run_agent` / `enqueue_resume_agent` instead of the old
`run_agent.delay()` / `resume_agent.delay()` Celery pattern.

The pool is process-scoped: opening it is cheap (<5ms for a Redis
connection vs. ~200ms for Postgres TLS), but still worth sharing.
"""

from __future__ import annotations

import asyncio

from arq import ArqRedis, create_pool
from arq.connections import RedisSettings

from ..common.config import get_settings
from ..common.logging import get_logger

log = get_logger(__name__)

_pool: ArqRedis | None = None
_pool_lock: asyncio.Lock | None = None


async def _get_pool() -> ArqRedis:
    global _pool, _pool_lock
    if _pool_lock is None:
        _pool_lock = asyncio.Lock()
    if _pool is not None:
        return _pool
    async with _pool_lock:
        if _pool is not None:
            return _pool
        settings = get_settings()
        _pool = await create_pool(RedisSettings.from_dsn(settings.celery_broker_url))
        log.info("arq.pool.opened")
        return _pool


async def enqueue_run_agent(message_json: str, *, event_ts: str | None = None) -> None:
    """Enqueue a run_agent job. Pass event_ts for Slack dedup (idempotency key)."""
    pool = await _get_pool()
    await pool.enqueue_job("run_agent", message_json, _job_id=event_ts)


async def enqueue_resume_agent(*, job_id: str, decision: str, user_id: str) -> None:
    """Enqueue a resume_agent job after an approval action."""
    pool = await _get_pool()
    await pool.enqueue_job(
        "resume_agent",
        _kwargs={"job_id": job_id, "decision": decision, "user_id": user_id},
    )
