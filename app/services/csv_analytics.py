from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from time import perf_counter

from app.models.schemas import AnalyticsIntent, AnalyticsQueryPlan
from app.services.snowflake_analytics import AnalyticsResult


class CsvAnalyticsBaseline:
    """Reference implementation used to validate Snowflake query results."""

    def __init__(self, business_data_dir: str | Path) -> None:
        self._directory = Path(business_data_dir)
        self.source = f"csv://{self._directory.as_posix()}"

    @staticmethod
    def _read(path: Path) -> list[dict[str, str]]:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    @staticmethod
    def _money(value: Decimal) -> float:
        return float(value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

    def query(self, plan: AnalyticsQueryPlan) -> AnalyticsResult:
        started = perf_counter()
        orders = {
            int(row["OrderID"]): row
            for row in self._read(self._directory / "Orders.csv")
        }
        products = {
            int(row["ProductID"]): row
            for row in self._read(self._directory / "Products.csv")
        }
        suppliers = {
            int(row["SupplierID"]): row
            for row in self._read(self._directory / "Suppliers.csv")
        }
        categories = {
            int(row["CategoryID"]): row
            for row in self._read(self._directory / "Categories.csv")
        }
        details = self._read(self._directory / "_Order_Details.csv")

        aggregates: dict[object, dict[str, Decimal | int | str]] = defaultdict(
            lambda: {"revenue": Decimal("0"), "units_sold": 0}
        )
        for detail in details:
            order = orders[int(detail["OrderID"])]
            order_date = datetime.fromisoformat(order["OrderDate"]).date()
            if plan.start_date is not None and order_date < plan.start_date:
                continue
            if plan.end_date is not None and order_date > plan.end_date:
                continue

            product = products[int(detail["ProductID"])]
            revenue = (
                Decimal(detail["UnitPrice"])
                * int(detail["Quantity"])
                * (Decimal("1") - Decimal(detail["Discount"]))
            )
            if plan.intent is AnalyticsIntent.TOP_PRODUCTS_BY_REVENUE:
                key: object = int(product["ProductID"])
                labels = {"product_name": product["ProductName"]}
            elif plan.intent is AnalyticsIntent.MONTHLY_SALES_TREND:
                key = order_date.strftime("%Y-%m")
                labels = {}
            elif plan.intent is AnalyticsIntent.SUPPLIER_PERFORMANCE:
                key = int(product["SupplierID"])
                labels = {"supplier_name": suppliers[key]["CompanyName"]}
            else:
                key = int(product["CategoryID"])
                labels = {"category_name": categories[key]["CategoryName"]}

            aggregate = aggregates[key]
            aggregate.update(labels)
            aggregate["revenue"] = Decimal(str(aggregate["revenue"])) + revenue
            aggregate["units_sold"] = int(aggregate["units_sold"]) + int(
                detail["Quantity"]
            )

        rows: list[dict[str, object]] = []
        for key, values in aggregates.items():
            if plan.intent is AnalyticsIntent.TOP_PRODUCTS_BY_REVENUE:
                identity = {"product_id": key, "product_name": values["product_name"]}
            elif plan.intent is AnalyticsIntent.MONTHLY_SALES_TREND:
                identity = {"month": key}
            elif plan.intent is AnalyticsIntent.SUPPLIER_PERFORMANCE:
                identity = {
                    "supplier_id": key,
                    "supplier_name": values["supplier_name"],
                }
            else:
                identity = {
                    "category_id": key,
                    "category_name": values["category_name"],
                }
            rows.append(
                {
                    **identity,
                    "revenue": self._money(Decimal(str(values["revenue"]))),
                    "units_sold": int(values["units_sold"]),
                }
            )

        if plan.intent is AnalyticsIntent.MONTHLY_SALES_TREND:
            rows.sort(key=lambda row: str(row["month"]))
        else:
            identity_key = next(key for key in rows[0] if key.endswith("_id")) if rows else ""
            rows.sort(key=lambda row: (-float(row["revenue"]), row[identity_key]))
        rows = rows[: plan.limit]
        return AnalyticsResult(
            intent=plan.intent,
            rows=rows,
            source=self.source,
            elapsed_ms=round((perf_counter() - started) * 1000, 2),
        )
