from __future__ import annotations

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings
from app.db.base import Base


def build_database(
    database_url: str,
    *,
    echo: bool = False,
    pool_size: int = 5,
    max_overflow: int = 10,
    pool_timeout: float = 30.0,
    pool_recycle: int = 1800,
) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    """Build an async engine and reusable session factory."""
    engine = create_async_engine(
        database_url,
        echo=echo,
        pool_pre_ping=True,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_timeout=pool_timeout,
        pool_recycle=pool_recycle,
    )
    if engine.url.get_backend_name() == "sqlite":
        event.listen(engine.sync_engine, "connect", _enable_sqlite_foreign_keys)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, session_factory


def _enable_sqlite_foreign_keys(dbapi_connection, connection_record) -> None:
    """Enable SQLite foreign-key enforcement for every new connection."""
    del connection_record
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


settings = get_settings()
engine, async_session_factory = build_database(
    settings.database_url,
    echo=settings.database_echo,
    pool_size=settings.database_pool_size,
    max_overflow=settings.database_max_overflow,
    pool_timeout=settings.database_pool_timeout_seconds,
    pool_recycle=settings.database_pool_recycle_seconds,
)


async def create_tables(database_engine: AsyncEngine = engine) -> None:
    """Create missing tables directly for isolated test databases."""
    # Import models here so their table metadata is registered before create_all.
    from app.db import models  # noqa: F401

    async with database_engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
