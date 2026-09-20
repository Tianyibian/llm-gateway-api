from __future__ import annotations

import asyncio
from datetime import date
from decimal import Decimal
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from app.models.schemas import AnalyticsIntent, AnalyticsQueryPlan
from app.services.analytics_comparison import AnalyticsComparisonService
from app.services.csv_analytics import CsvAnalyticsBaseline
from app.services.snowflake_analytics import SnowflakeAnalyticsService


def test_driver_query_id_is_captured_before_result_consumption():
    engine = MagicMock()
    result = engine.connect.return_value.__enter__.return_value.execute.return_value
    result.cursor.sfqid = "real-driver-query-id"
    def consume():
        result.cursor = None
        return [{"revenue":Decimal("1.25")}]
    result.mappings.return_value.all.side_effect = consume
    service = SnowflakeAnalyticsService(database="ANALYTICS", schema="ECOMMERCE", engine=engine)
    rows, query_id = service._run_with_engine("SELECT 1", {})
    assert rows[0]["revenue"] == Decimal("1.25")
    assert query_id == "real-driver-query-id"


def test_analytics_plan_rejects_reversed_date_range() -> None:
    with pytest.raises(ValidationError, match="end_date must be on or after"):
        AnalyticsQueryPlan(
            intent=AnalyticsIntent.MONTHLY_SALES_TREND,
            start_date=date(2025, 2, 1),
            end_date=date(2025, 1, 1),
            reason="Invalid test range.",
        )


def test_snowflake_service_uses_allowlisted_sql_and_bound_parameters() -> None:
    calls = []

    def runner(sql, parameters):
        calls.append((sql, parameters))
        return ([{"product_id": 1, "revenue": Decimal("42.50")}], "query-123")

    async def scenario() -> None:
        service = SnowflakeAnalyticsService(
            database="ANALYTICS",
            schema="ECOMMERCE",
            query_runner=runner,
        )
        plan = AnalyticsQueryPlan(
            intent=AnalyticsIntent.TOP_PRODUCTS_BY_REVENUE,
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
            limit=5,
            reason="Rank products.",
        )

        result = await service.query(plan)

        assert "JOIN ORDER_DETAILS" in calls[0][0]
        assert ":start_date" in calls[0][0]
        assert calls[0][1] == {
            "start_date": date(2025, 1, 1),
            "end_date": date(2025, 1, 31),
            "limit": 5,
        }
        assert result.query_id == "query-123"
        assert result.source == "snowflake://ANALYTICS.ECOMMERCE"

    asyncio.run(scenario())


def test_csv_baseline_calculates_discounted_product_revenue() -> None:
    plan = AnalyticsQueryPlan(
        intent=AnalyticsIntent.TOP_PRODUCTS_BY_REVENUE,
        limit=1,
        reason="Find the top product.",
    )

    result = CsvAnalyticsBaseline("Business_data").query(plan)

    assert result.rows == [
        {
            "product_id": 72,
            "product_name": "TP-Link Kasa Security System Standard",
            "revenue": 4199589.65,
            "units_sold": 484,
        }
    ]


def test_comparison_normalizes_snowflake_decimal_values() -> None:
    async def scenario() -> None:
        baseline = CsvAnalyticsBaseline("Business_data")
        plan = AnalyticsQueryPlan(
            intent=AnalyticsIntent.TOP_PRODUCTS_BY_REVENUE,
            limit=1,
            reason="Compare engines.",
        )
        expected = baseline.query(plan).rows

        def runner(sql, parameters):
            del sql, parameters
            row = expected[0]
            return (
                [
                    {
                        "PRODUCT_ID": Decimal(str(row["product_id"])),
                        "PRODUCT_NAME": row["product_name"],
                        "REVENUE": Decimal(str(row["revenue"])),
                        "UNITS_SOLD": Decimal(str(row["units_sold"])),
                    }
                ],
                "query-456",
            )

        snowflake = SnowflakeAnalyticsService(
            database="ANALYTICS",
            schema="ECOMMERCE",
            query_runner=runner,
        )
        comparison = await AnalyticsComparisonService(
            csv_baseline=baseline,
            snowflake=snowflake,
        ).compare(plan)

        assert comparison.results_match is True
        assert comparison.mismatch is None

    asyncio.run(scenario())
