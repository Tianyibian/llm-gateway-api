"""Print allowed, observed Neo4j schema metadata without model calls or writes."""
import asyncio

from app.core.config import Settings
from app.services.neo4j_service import Neo4jExecutor


async def inspect():
    schema = await Neo4jExecutor(Settings()).schema()
    print(schema.prompt_text())


def main():
    try:
        asyncio.run(inspect())
    except Exception:
        raise SystemExit("Schema discovery failed. Check Neo4j readiness, permissions and schema compatibility.") from None


if __name__ == "__main__":
    main()
