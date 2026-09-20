from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Any

from app.models.schemas import AnalyticsQueryPlan
from app.services.csv_analytics import CsvAnalyticsBaseline
from app.services.snowflake_analytics import SnowflakeAnalyticsService


@dataclass(frozen=True, kw_only=True)
class AnalyticsComparison:
    plan: dict[str, Any]
    csv_rows: list[dict[str, Any]]
    snowflake_rows: list[dict[str, Any]]
    csv_elapsed_ms: float
    snowflake_elapsed_ms: float
    results_match: bool
    mismatch: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnalyticsComparisonService:
    """Compare Snowflake output with the deterministic local CSV baseline."""

    def __init__(
        self,
        *,
        csv_baseline: CsvAnalyticsBaseline,
        snowflake: SnowflakeAnalyticsService,
    ) -> None:
        self._csv = csv_baseline
        self._snowflake = snowflake

    @staticmethod
    def _normalize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normalized = []
        for row in rows:
            normalized.append(
                {
                    str(key).lower(): (
                        float(value) if isinstance(value, Decimal) else value
                    )
                    for key, value in row.items()
                }
            )
        return normalized

    async def compare(self, plan: AnalyticsQueryPlan) -> AnalyticsComparison:
        csv_task = asyncio.to_thread(self._csv.query, plan)
        csv_result, snowflake_result = await asyncio.gather(
            csv_task,
            self._snowflake.query(plan),
        )
        csv_rows = self._normalize(csv_result.rows)
        snowflake_rows = self._normalize(snowflake_result.rows)
        matches = csv_rows == snowflake_rows
        return AnalyticsComparison(
            plan=plan.model_dump(mode="json"),
            csv_rows=csv_rows,
            snowflake_rows=snowflake_rows,
            csv_elapsed_ms=csv_result.elapsed_ms,
            snowflake_elapsed_ms=snowflake_result.elapsed_ms,
            results_match=matches,
            mismatch=None if matches else "Rows differ after key and number normalization.",
        )
