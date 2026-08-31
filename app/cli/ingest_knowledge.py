from __future__ import annotations

import asyncio
from dataclasses import asdict
import json

from app.db.session import engine
from app.services.factory import LLMServiceFactory


async def _run() -> None:
    service = LLMServiceFactory().create_knowledge_ingestion_service()
    try:
        report = await service.ingest()
        print(json.dumps(asdict(report), indent=2))
    finally:
        await engine.dispose()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
