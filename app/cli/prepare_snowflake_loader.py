"""Prepare local, separate loading credentials; never execute remote grants."""
from __future__ import annotations

import json
import os
from pathlib import Path
import secrets

from app.core.config import get_settings
from app.services.errors import LLMConfigurationError


def prepare() -> tuple[Path, Path]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    settings = get_settings()
    if (settings.snowflake_database, settings.snowflake_schema, settings.snowflake_warehouse) != (
        "SMART_AI_ANALYTICS", "ECOMMERCE", "SMART_AI_WH"
    ):
        raise LLMConfigurationError("Loader setup expects the documented analytics database, schema, and warehouse.")
    if not settings.snowflake_account:
        raise LLMConfigurationError("Missing SNOWFLAKE_ACCOUNT.")
    directory = Path(".local/snowflake/loader")
    env_path = Path(".env.snowflake-loader")
    if directory.exists() or env_path.exists():
        raise LLMConfigurationError("Loader files already exist. Inspect and reuse them; do not overwrite credentials.")
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    passphrase = secrets.token_urlsafe(48)
    public_key = key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()
    key_body = "".join(public_key.splitlines()[1:-1])
    private_key = key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.BestAvailableEncryption(passphrase.encode()),
    )
    template = Path("snowflake/setup_loader.sql").read_text(encoding="utf-8")
    if template.count("PASTE_LOADER_PUBLIC_KEY_HERE") != 1:
        raise LLMConfigurationError("Unexpected loader SQL template.")
    directory.mkdir(parents=True, mode=0o700)
    directory.chmod(0o700)

    def exclusive_write(path: Path, content: bytes) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as file:
            file.write(content)

    key_path = directory / "rsa_key.p8"
    exclusive_write(key_path, private_key)
    exclusive_write(directory / "rsa_key.pub", public_key.encode())
    sql_path = directory / "setup_loader.sql"
    exclusive_write(sql_path, template.replace("PASTE_LOADER_PUBLIC_KEY_HERE", key_body).encode())
    values = {
        "SNOWFLAKE_ENABLED": "false",
        "SNOWFLAKE_ACCOUNT": settings.snowflake_account,
        "SNOWFLAKE_USER": "SMART_AI_LOADER_APP",
        "SNOWFLAKE_AUTH_METHOD": "key_pair",
        "SNOWFLAKE_PRIVATE_KEY_FILE": str(key_path.resolve()),
        "SNOWFLAKE_PRIVATE_KEY_PASSPHRASE": passphrase,
        "SNOWFLAKE_WAREHOUSE": settings.snowflake_warehouse,
        "SNOWFLAKE_DATABASE": settings.snowflake_database,
        "SNOWFLAKE_SCHEMA": settings.snowflake_schema,
        "SNOWFLAKE_ROLE": "SMART_AI_LOADER",
        "SNOWFLAKE_LOAD_ROLE": "SMART_AI_LOADER",
        "BUSINESS_DATA_DIR": str(Path(settings.business_data_dir).resolve()),
    }
    # dotenv does not decode JSON's Unicode escapes in filesystem paths.
    exclusive_write(env_path, ("\n".join(f"{name}={json.dumps(value, ensure_ascii=False)}" for name, value in values.items()) + "\n").encode())
    return sql_path.resolve(), env_path.resolve()


def main() -> None:
    try:
        sql_path, env_path = prepare()
    except (LLMConfigurationError, OSError) as exc:
        print(f"Preparation stopped: {exc}")
        raise SystemExit(1)
    print(f"Run this public-key-only SQL in an administrator worksheet: {sql_path}")
    print(f"Local private loader settings prepared: {env_path}")
    print("No remote changes, uploads, or grants were performed. Never share the env file or private key.")


if __name__ == "__main__":
    main()
