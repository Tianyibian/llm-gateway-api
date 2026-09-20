"""Generate private local-development credentials, without touching an existing file."""
import os
from pathlib import Path
import secrets


def main():
    path = Path(".env.neo4j")
    password = secrets.token_urlsafe(32)
    content = ("# Local Community development only; run the database read-only after import.\n"
               "NEO4J_ENABLED=true\nNEO4J_URI=bolt://127.0.0.1:7687\nNEO4J_USER=neo4j\n"
               f"NEO4J_PASSWORD={password}\nNEO4J_DATABASE=neo4j\nNEO4J_DATASET=aster-business-v1\n"
               f"NEO4J_LOAD_USER=neo4j\nNEO4J_LOAD_PASSWORD={password}\n")
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        raise SystemExit(".env.neo4j already exists; it was not changed.") from None
    with os.fdopen(fd, "w") as handle:
        handle.write(content)
    print("Created private .env.neo4j (mode 600). No credentials were printed.")


if __name__ == "__main__":
    main()
