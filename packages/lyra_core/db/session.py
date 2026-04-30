"""Async SQLAlchemy session management."""

from __future__ import annotations

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..common.config import get_settings

_settings = get_settings()

# NOTE: `statement_cache_size=0` is required for Supabase's transaction-mode
# pooler (port 6543) and any other pgbouncer setup that recycles backend
# connections across clients. Without this, asyncpg tries to prepare statements
# under predictable names (`__asyncpg_stmt_1__`, ...), they leak between
# clients via the pooler, and you get DuplicatePreparedStatementError on
# random queries. Switch to session-mode pooler (port 5432) if you ever need
# prepared-statement performance back.
engine = create_async_engine(
    _settings.database_url,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
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
