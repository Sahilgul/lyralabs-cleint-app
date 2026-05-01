"""Async SQLAlchemy session management."""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from ..common.config import get_settings

_settings = get_settings()

# NOTE: `statement_cache_size=0` is required for Supabase's transaction-mode
# pooler (port 6543) and any other pgbouncer setup that recycles backend
# connections across clients. Without this, asyncpg tries to prepare statements
# under predictable names (`__asyncpg_stmt_1__`, ...), they leak between
# clients via the pooler, and you get DuplicatePreparedStatementError on
# random queries. Switch to session-mode pooler (port 5432) if you ever need
# prepared-statement performance back.
#
# `poolclass=NullPool`: required because the Celery worker creates a fresh
# asyncio event loop per task (`asyncio.run(_run(...))` in run_agent.py).
# asyncpg connections are bound to the loop they were opened on -- a pooled
# connection from a previous, now-closed loop will raise
# "Event loop is closed" / "Future attached to a different loop" the next
# time SQLAlchemy hands it out. NullPool opens one connection per checkout
# and closes it on return, so each task gets a connection scoped to its
# own loop. The API process (single uvicorn loop) gives up some pooling
# perf, but Supabase's pgbouncer (port 6543) is pooling underneath us
# anyway, so the real cost is a few ms per request.
engine = create_async_engine(
    _settings.database_url,
    poolclass=NullPool,
    echo=False,
    connect_args={"statement_cache_size": 0},
)

async_session: async_sessionmaker[AsyncSession] = async_sessionmaker(
    bind=engine,
    expire_on_commit=False,
    class_=AsyncSession,
)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency. Wraps each request in a transaction."""
    async with async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
