"""Postgres-backed LangGraph checkpointer.

Uses langgraph-checkpoint-postgres. The checkpointer persists graph state
keyed by `thread_id`, so we can `interrupt()` a graph for human approval
and resume hours later.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool

from ..common.config import get_settings


@asynccontextmanager
async def checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    """Yield a connected AsyncPostgresSaver."""
    settings = get_settings()

    # langgraph-checkpoint-postgres needs the sync-style URL, no +asyncpg/+psycopg suffix
    pg_url = settings.database_url_sync.replace("postgresql+psycopg://", "postgresql://")

    async with AsyncConnectionPool(
        conninfo=pg_url,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    ) as pool:
        saver = AsyncPostgresSaver(pool)  # type: ignore[arg-type]
        await saver.setup()
        yield saver
