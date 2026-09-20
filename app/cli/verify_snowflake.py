"""Read-only live validation of all five tables and four report templates."""
from __future__ import annotations

import asyncio
from datetime import date
import json
import logging
from pathlib import Path

from app.cli.load_snowflake import read_sources
from app.core.config import Settings
from app.models.schemas import AnalyticsIntent, AnalyticsQueryPlan
from app.services.analytics_comparison import AnalyticsComparisonService
from app.services.csv_analytics import CsvAnalyticsBaseline
from app.services.snowflake_analytics import create_snowflake_engine, dispose_snowflake_engines, SnowflakeAnalyticsService


async def verify():
    settings = Settings()
    # Explicit diagnostic only: does not change the assistant's enablement/routing.
    engine = create_snowflake_engine(settings)
    sources = read_sources(Path(settings.business_data_dir))
    counts = {}
    with engine.connect() as connection:
        for table, (_, expected, _) in sources.items():
            actual = connection.exec_driver_sql(f"SELECT COUNT(*) FROM {table}").scalar_one()
            counts[table] = {"expected": expected, "actual": actual, "match": actual == expected}
    print(json.dumps({"table_counts": counts}), flush=True)
    if not all(row["match"] for row in counts.values()):
        print("Verification stopped: table counts differ. No writes or replacement were performed.")
        return False
    service = SnowflakeAnalyticsService(engine=engine, database=settings.snowflake_database, schema=settings.snowflake_schema)
    comparison = AnalyticsComparisonService(csv_baseline=CsvAnalyticsBaseline(settings.business_data_dir), snowflake=service)
    checks = []
    for intent in AnalyticsIntent:
        for year in (None, 2025, 2026):
            plan = AnalyticsQueryPlan(intent=intent, start_date=date(year,1,1) if year else None,
                end_date=date(year,12,31) if year else None, limit=20, reason="Live source-data correctness verification")
            result = await comparison.compare(plan)
            checks.append(result.results_match)
            print(json.dumps({"intent":intent.value, "year":year, "matches_csv":result.results_match,
                              "rows":len(result.snowflake_rows)}), flush=True)
    print(json.dumps({"passed":sum(checks), "total":len(checks), "writes_performed":False}))
    return all(checks)


def main():
    logging.getLogger("snowflake.connector").setLevel(logging.CRITICAL)
    try:
        success = asyncio.run(verify())
    except Exception as exc:
        original = getattr(exc, "orig", exc)
        print(json.dumps({"verified":False, "error_type":type(original).__name__, "code":getattr(original,"errno",None)}))
        success = False
    finally:
        dispose_snowflake_engines()
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
