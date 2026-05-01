"""Postgres-backed LangGraph checkpointer.

Uses langgraph-checkpoint-postgres. The checkpointer persists graph state
keyed by `thread_id`, so we can `interrupt()` a graph for human approval
and resume hours later.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.errors import UniqueViolation
from psycopg_pool import AsyncConnectionPool

from ..common.config import get_settings
from ..common.logging import get_logger

log = get_logger(__name__)

# Per-process flag: once we've successfully run setup() in this worker
# process, skip it on subsequent calls. Migrations are global to the DB,
# so re-running them every task wastes ~1-2s and races with sibling
# Celery prefork workers (see UniqueViolation handling below).
_SETUP_DONE = False


@asynccontextmanager
async def checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    """Yield a connected AsyncPostgresSaver."""
    global _SETUP_DONE
    settings = get_settings()

    # langgraph-checkpoint-postgres needs the sync-style URL, no +asyncpg/+psycopg suffix
    pg_url = settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://")

    async with AsyncConnectionPool(
        conninfo=pg_url,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    ) as pool:
        saver = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
        if not _SETUP_DONE:
            # langgraph's setup() does CREATE TABLE IF NOT EXISTS for the
            # checkpoint tables, then INSERTs into checkpoint_migrations.
            # That INSERT has no ON CONFLICT, so concurrent prefork workers
            # racing on a fresh DB hit UniqueViolation. Treat that as
            # benign -- it just means a sibling worker won the race and
            # the migration is already applied.
            try:
                await saver.setup()
            except UniqueViolation:
                log.info("checkpointer.setup.race_lost")
            _SETUP_DONE = True
        yield saver
