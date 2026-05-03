"""Postgres-backed LangGraph checkpointer.

Uses langgraph-checkpoint-postgres. The checkpointer persists graph state
keyed by `thread_id`, so we can `interrupt()` a graph for human approval
and resume hours later.

The connection pool is process-scoped: opened lazily on first use, kept
warm for the lifetime of the worker process. Per-task pool open/close
costs ~1-1.5s of TLS handshake to Supabase Tokyo from us-east1, so a
shared pool is the difference between a 5s direct reply and a 7s one.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.errors import UniqueViolation
from psycopg_pool import AsyncConnectionPool

from ..common.config import get_settings
from ..common.logging import get_logger

log = get_logger(__name__)

_pool: AsyncConnectionPool | None = None
_pool_lock = asyncio.Lock()
_SETUP_DONE = False


async def _get_pool() -> AsyncConnectionPool:
    """Return the process-scoped connection pool, opening it on first call.

    `min_size=0` so worker startup doesn't block on a Postgres TLS handshake.
    Connections are created lazily on first checkout; the pool stays warm
    after that. Combined with arq's Redis pool initializing right before
    this, the previous `min_size=1, wait=True` opener occasionally hit the
    30s default timeout — fatal at startup, but transient (a TLS handshake
    blip to Supabase). Lazy creation makes startup deterministic and pushes
    any real connectivity failure to the first job, where it surfaces as a
    normal phase error instead of crashing the whole worker.
    """
    global _pool
    if _pool is not None:
        return _pool
    async with _pool_lock:
        if _pool is not None:
            return _pool
        settings = get_settings()
        pg_url = settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://")
        pool = AsyncConnectionPool(
            conninfo=pg_url,
            min_size=0,
            max_size=4,
            kwargs={"autocommit": True, "prepare_threshold": 0},
            open=False,
        )
        # `wait=False` returns as soon as the pool object is set up; it does
        # NOT block on creating min_size connections (we have min_size=0
        # anyway). Generous timeout for safety against future config changes.
        await pool.open(wait=False, timeout=60)
        _pool = pool
        return _pool


@asynccontextmanager
async def checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    """Yield a connected AsyncPostgresSaver backed by the shared pool."""
    global _SETUP_DONE
    pool = await _get_pool()
    saver = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
    if not _SETUP_DONE:
        # langgraph's setup() does CREATE TABLE IF NOT EXISTS for the
        # checkpoint tables, then INSERTs into checkpoint_migrations.
        # That INSERT has no ON CONFLICT, so concurrent workers racing on
        # a fresh DB hit UniqueViolation. Treat that as benign — it just
        # means a sibling worker won the race and the migration is already
        # applied.
        try:
            await saver.setup()
        except UniqueViolation:
            log.info("checkpointer.setup.race_lost")
        _SETUP_DONE = True
    yield saver
