from __future__ import annotations

import argparse
import asyncio
from datetime import date
import json

from app.core.config import get_settings
from app.models.schemas import AnalyticsIntent, AnalyticsQueryPlan
from app.services.analytics_comparison import AnalyticsComparisonService
from app.services.csv_analytics import CsvAnalyticsBaseline
from app.services.errors import LLMConfigurationError
from app.services.snowflake_analytics import SnowflakeAnalyticsService


def parse_date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    plan = AnalyticsQueryPlan(
        intent=AnalyticsIntent(args.intent),
        start_date=parse_date(args.start_date),
        end_date=parse_date(args.end_date),
        limit=args.limit,
        reason="Command-line correctness and latency comparison.",
    )
    comparison = AnalyticsComparisonService(
        csv_baseline=CsvAnalyticsBaseline(settings.business_data_dir),
        snowflake=SnowflakeAnalyticsService.from_settings(settings),
    )
    result = await comparison.compare(plan)
    print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0 if result.results_match else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Snowflake analytics with the local CSV baseline."
    )
    parser.add_argument(
        "--intent",
        choices=[intent.value for intent in AnalyticsIntent],
        default=AnalyticsIntent.TOP_PRODUCTS_BY_REVENUE.value,
    )
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--limit", type=int, default=10)
    try:
        exit_code = asyncio.run(run(parser.parse_args()))
    except (LLMConfigurationError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
