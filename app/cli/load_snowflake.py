"""Initial full-field CSV load using an isolated, insert-only identity."""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from uuid import uuid4

from app.core.config import Settings
from app.services.errors import LLMConfigurationError
from app.services.snowflake_analytics import create_snowflake_engine, dispose_snowflake_engines


TABLE_FILES = {
    "CATEGORIES": "Categories.csv",
    "SUPPLIERS": "Suppliers.csv",
    "PRODUCTS": "Products.csv",
    "ORDERS": "Orders.csv",
    "ORDER_DETAILS": "_Order_Details.csv",
}
STAGE = "SMART_AI_LOAD_STAGE"


def read_sources(directory: Path) -> dict[str, tuple[Path, int, list[str]]]:
    """Validate files without projecting, editing, or omitting any fields."""
    sources = {}
    for table, filename in TABLE_FILES.items():
        path = (directory / filename).resolve()
        with path.open(encoding="utf-8-sig", newline="") as file:
            reader = csv.reader(file, strict=True)
            headers = next(reader, [])
            if not headers or any(not name.strip() for name in headers):
                raise ValueError(f"{table}: missing CSV headers.")
            normalized = [name.upper() for name in headers]
            if len(set(normalized)) != len(normalized):
                raise ValueError(f"{table}: duplicate CSV headers.")
            count = 0
            for row in reader:
                if len(row) != len(headers):
                    raise ValueError(f"{table}: inconsistent CSV row width.")
                count += 1
            if not count:
                raise ValueError(f"{table}: refusing an empty source file.")
        sources[table] = (path, count, normalized)
    return sources


def loader_settings(env_file: Path) -> Settings:
    if not env_file.is_file():
        raise LLMConfigurationError("Missing dedicated loader env file. Run app.cli.prepare_snowflake_loader first.")
    settings = Settings(_env_file=env_file)
    if (settings.snowflake_user, settings.snowflake_role, settings.snowflake_load_role) != (
        "SMART_AI_LOADER_APP", "SMART_AI_LOADER", "SMART_AI_LOADER"
    ):
        raise LLMConfigurationError("Loading requires the separate SMART_AI_LOADER_APP identity and SMART_AI_LOADER role.")
    if (settings.snowflake_database, settings.snowflake_schema) != ("SMART_AI_ANALYTICS", "ECOMMERCE"):
        raise LLMConfigurationError("Loader target must be SMART_AI_ANALYTICS.ECOMMERCE.")
    if settings.snowflake_auth_method != "key_pair":
        raise LLMConfigurationError("Loading requires explicit key_pair authentication.")
    return settings


def load(*, env_file: Path = Path(".env.snowflake-loader")) -> None:
    settings = loader_settings(env_file)
    sources = read_sources(Path(settings.business_data_dir))
    engine = create_snowflake_engine(settings)
    engine.update_execution_options(autocommit=False)
    prefix = f"initial_{uuid4().hex}"
    try:
        with engine.connect() as connection:
            context = connection.exec_driver_sql(
                "SELECT CURRENT_USER(), CURRENT_ROLE(), CURRENT_DATABASE(), CURRENT_SCHEMA()"
            ).one()
            if tuple(context) != ("SMART_AI_LOADER_APP", "SMART_AI_LOADER", "SMART_AI_ANALYTICS", "ECOMMERCE"):
                raise LLMConfigurationError("Unexpected live loader session context; no files uploaded.")
            # Validate every destination before the first upload.
            for table, (_, _, headers) in sources.items():
                if connection.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").scalar_one() != 0:
                    raise ValueError(f"{table}: target is not empty; no automatic replacement.")
                columns = connection.exec_driver_sql(f"DESC TABLE {table}").mappings().all()
                if {str(row["name"]).upper() for row in columns} != set(headers):
                    raise ValueError(f"{table}: source and target columns differ; no files uploaded.")
            for table, (path, _, _) in sources.items():
                escaped = path.as_posix().replace("'", "''")
                uploaded = connection.exec_driver_sql(
                    f"PUT 'file://{escaped}' @{STAGE}/{prefix}/{table}/ "
                    "AUTO_COMPRESS=TRUE OVERWRITE=FALSE"
                ).mappings().all()
                if len(uploaded) != 1 or str(uploaded[0]["status"]).upper() != "UPLOADED":
                    raise RuntimeError(f"{table}: expected one successful staged upload.")
                if uploaded[0]["target"] != path.name + ".gz":
                    raise RuntimeError(f"{table}: unexpected staged filename.")

        # COPY is DML: commit all tables together, or roll back inserts on failure.
        # Staging uploads remain available for diagnosis, independent of rollback.
        with engine.begin() as connection:
            for table, (path, expected, _) in sources.items():
                if connection.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").scalar_one() != 0:
                    raise ValueError(f"{table}: target changed during upload; aborting.")
                filename = (path.name + ".gz").replace("'", "''")
                results = connection.exec_driver_sql(
                    f"COPY INTO {table} FROM @{STAGE}/{prefix}/{table}/ "
                    f"FILES=('{filename}') FILE_FORMAT=(FORMAT_NAME=SMART_AI_CSV) "
                    "MATCH_BY_COLUMN_NAME=CASE_INSENSITIVE ON_ERROR=ABORT_STATEMENT PURGE=FALSE"
                ).mappings().all()
                if not results or any(str(row["status"]).upper() != "LOADED" for row in results):
                    raise RuntimeError(f"{table}: COPY did not report a complete load.")
                if sum(int(row["rows_loaded"]) for row in results) != expected:
                    raise RuntimeError(f"{table}: COPY row count differs from the local source.")
                actual = connection.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").scalar_one()
                if actual != expected:
                    raise RuntimeError(f"{table}: destination row count mismatch.")
        for table, (_, count, _) in sources.items():
            print(f"{table}: committed {count} rows")
        print(f"All five tables committed. Staged copies remain at @{STAGE}/{prefix}/; local sources are unchanged.")
    finally:
        dispose_snowflake_engines()


def main() -> None:
    parser = argparse.ArgumentParser(description="Load five complete CSV files with separate credentials; refuses nonempty targets.")
    parser.add_argument("--env-file", type=Path, default=Path(".env.snowflake-loader"))
    args = parser.parse_args()
    logging.getLogger("snowflake.connector").setLevel(logging.CRITICAL)
    try:
        load(env_file=args.env_file)
    except (LLMConfigurationError, FileNotFoundError, ValueError) as exc:
        parser.exit(2, f"Load stopped ({type(exc).__name__}). Check local settings, source files, and destination state.\n")
    except Exception as exc:
        original = getattr(exc, "orig", exc)
        parser.exit(1, f"Load failed: {type(original).__name__}; code={getattr(original, 'errno', None)}, sqlstate={getattr(original, 'sqlstate', None)}. No success is claimed.\n")


if __name__ == "__main__":
    main()
