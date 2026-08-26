from datetime import datetime, timezone

import pytest

from scripts.migrate_sqlite_to_postgres import (
    _normalize_timestamps,
    _validate_urls,
)


def test_migration_urls_require_sqlite_source_and_postgresql_target() -> None:
    source, target = _validate_urls(
        "sqlite+aiosqlite:///./source.db",
        "postgresql+asyncpg://user:password@localhost/database",
    )

    assert source.get_backend_name() == "sqlite"
    assert target.get_backend_name() == "postgresql"

    with pytest.raises(ValueError, match="SQLite"):
        _validate_urls(
            "postgresql+asyncpg://user:password@localhost/source",
            "postgresql+asyncpg://user:password@localhost/target",
        )

    with pytest.raises(ValueError, match="PostgreSQL"):
        _validate_urls(
            "sqlite+aiosqlite:///./source.db",
            "sqlite+aiosqlite:///./target.db",
        )


def test_migration_normalizes_naive_sqlite_timestamps_to_utc() -> None:
    naive = datetime(2026, 8, 24, 12, 30)
    aware = datetime(2026, 8, 24, 12, 30, tzinfo=timezone.utc)

    normalized = _normalize_timestamps(
        {"created_at": naive, "updated_at": aware},
        "created_at",
        "updated_at",
    )

    assert normalized["created_at"].tzinfo == timezone.utc
    assert normalized["updated_at"] is aware
