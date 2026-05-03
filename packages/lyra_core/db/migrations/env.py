"""Alembic env. Uses sync DATABASE_URL_SYNC for migrations."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from lyra_core.common.config import get_settings
from lyra_core.db.models import Base
from sqlalchemy import create_engine, pool

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Read the URL straight from settings — never push it through Alembic's
# ConfigParser. ConfigParser treats `%` as interpolation syntax and chokes on
# URL-encoded passwords (e.g. `%26` for `&`).
DATABASE_URL_SYNC = get_settings().database_url_sync

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    context.configure(
        url=DATABASE_URL_SYNC,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(DATABASE_URL_SYNC, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
