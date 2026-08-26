from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any

from sqlalchemy import func, insert, select, text
from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app.core.config import get_settings
from app.db.models import Conversation, ConversationMessage

DEFAULT_SOURCE_URL = "sqlite+aiosqlite:///./llm_gateway.db"


def _display_url(url: str) -> str:
    return make_url(url).render_as_string(hide_password=True)


def _validate_urls(source_url: str, target_url: str) -> tuple[URL, URL]:
    source = make_url(source_url)
    target = make_url(target_url)
    if source.get_backend_name() != "sqlite":
        raise ValueError("SQLITE_SOURCE_URL must use a SQLite SQLAlchemy URL.")
    if target.get_backend_name() != "postgresql":
        raise ValueError("DATABASE_URL must use a PostgreSQL SQLAlchemy URL.")
    return source, target


def _require_source_file(source: URL) -> None:
    database = source.database
    if not database or database == ":memory:":
        raise ValueError("The source must be a file-backed SQLite database.")
    if not Path(database).expanduser().resolve().is_file():
        raise FileNotFoundError(f"SQLite source file was not found: {database}")


def _normalize_timestamps(row: dict[str, Any], *fields: str) -> dict[str, Any]:
    normalized = dict(row)
    for field in fields:
        value = normalized.get(field)
        if isinstance(value, datetime) and value.tzinfo is None:
            normalized[field] = value.replace(tzinfo=timezone.utc)
    return normalized


async def _read_source(
    source_engine: AsyncEngine,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    async with source_engine.connect() as connection:
        conversations_result = await connection.execute(
            select(Conversation.__table__).order_by(
                Conversation.created_at,
                Conversation.id,
            )
        )
        messages_result = await connection.execute(
            select(ConversationMessage.__table__).order_by(ConversationMessage.id)
        )

    conversations = [
        _normalize_timestamps(dict(row), "created_at", "updated_at")
        for row in conversations_result.mappings()
    ]
    messages = [
        _normalize_timestamps(dict(row), "created_at")
        for row in messages_result.mappings()
    ]
    return conversations, messages


async def migrate(
    source_url: str,
    target_url: str,
    *,
    dry_run: bool = False,
) -> tuple[int, int]:
    source, _ = _validate_urls(source_url, target_url)
    _require_source_file(source)

    source_engine = create_async_engine(source_url)
    target_engine = create_async_engine(target_url, pool_pre_ping=True)
    try:
        conversations, messages = await _read_source(source_engine)

        try:
            async with target_engine.begin() as connection:
                target_conversations = await connection.scalar(
                    select(func.count()).select_from(Conversation)
                )
                target_messages = await connection.scalar(
                    select(func.count()).select_from(ConversationMessage)
                )
                if target_conversations or target_messages:
                    raise RuntimeError(
                        "PostgreSQL target is not empty. Migration stopped to prevent "
                        "duplicate or conflicting data."
                    )

                if not dry_run:
                    if conversations:
                        await connection.execute(
                            insert(Conversation.__table__), conversations
                        )
                    if messages:
                        await connection.execute(
                            insert(ConversationMessage.__table__), messages
                        )
                        await connection.execute(
                            text(
                                "SELECT setval("
                                "pg_get_serial_sequence('messages', 'id'), "
                                ":maximum_id, true)"
                            ),
                            {"maximum_id": max(message["id"] for message in messages)},
                        )
        except SQLAlchemyError as exc:
            raise RuntimeError(
                "PostgreSQL migration failed and was rolled back. Verify the "
                "connection and run 'alembic upgrade head' before retrying."
            ) from exc

        return len(conversations), len(messages)
    finally:
        await source_engine.dispose()
        await target_engine.dispose()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy conversations and messages from SQLite into an empty PostgreSQL "
            "database without deleting or modifying the source file."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate both databases and report row counts without inserting data.",
    )
    return parser.parse_args()


async def _main() -> None:
    args = _parse_args()
    source_url = os.getenv("SQLITE_SOURCE_URL", DEFAULT_SOURCE_URL)
    target_url = get_settings().database_url

    print(f"SQLite source: {_display_url(source_url)}")
    print(f"PostgreSQL target: {_display_url(target_url)}")
    conversations, messages = await migrate(
        source_url,
        target_url,
        dry_run=args.dry_run,
    )
    action = "Validated" if args.dry_run else "Migrated"
    print(f"{action} {conversations} conversations and {messages} messages.")
    if not args.dry_run:
        print("SQLite source was preserved unchanged.")


if __name__ == "__main__":
    asyncio.run(_main())
