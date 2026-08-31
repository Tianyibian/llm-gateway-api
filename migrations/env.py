from __future__ import annotations

import asyncio
from logging.config import fileConfig

from alembic import context
from pgvector.sqlalchemy import Vector
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool

from app.core.config import get_settings
from app.db import models  # noqa: F401 - registers ORM metadata
from app.db.base import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def compare_column_types(
    migration_context,
    inspected_column,
    metadata_column,
    inspected_type,
    metadata_type,
):
    """Ignore SQLite's lossy reflection of pgvector's portable test type."""
    del inspected_column, metadata_column, inspected_type
    if migration_context.dialect.name == "sqlite" and isinstance(metadata_type, Vector):
        return False
    return None


def get_database_url() -> str:
    """Read the same environment-aware database URL used by the application."""
    return get_settings().database_url


def run_migrations_offline() -> None:
    """Generate SQL without opening a database connection."""
    database_url = get_database_url()
    context.configure(
        url=database_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=compare_column_types,
        render_as_batch=database_url.startswith("sqlite"),
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
        compare_type=compare_column_types,
        render_as_batch=connection.dialect.name == "sqlite",
    )

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = create_async_engine(get_database_url(), poolclass=NullPool)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations through the configured async database driver."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
