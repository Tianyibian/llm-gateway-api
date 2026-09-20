from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
import os
from pathlib import Path
from time import perf_counter
from typing import Any

from sqlalchemy import Engine, text

from app.models.schemas import AnalyticsIntent, AnalyticsQueryPlan
from app.services.errors import LLMConfigurationError


_SNOWFLAKE_ENGINES: dict[tuple[object, ...], Engine] = {}


def _authentication_parameters(settings: Any) -> dict[str, Any]:
    """Validate explicit authentication; never fall back from a broken key."""
    method = settings.snowflake_auth_method
    if method == "password":
        secret = settings.snowflake_password
        if secret is None or not secret.get_secret_value().strip():
            raise LLMConfigurationError("Missing SNOWFLAKE_PASSWORD for password authentication.")
        return {"password": secret.get_secret_value()}
    if method != "key_pair":
        raise LLMConfigurationError("Unsupported SNOWFLAKE_AUTH_METHOD.")
    if not settings.snowflake_private_key_file:
        raise LLMConfigurationError("Missing SNOWFLAKE_PRIVATE_KEY_FILE for key_pair authentication.")
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
    except ImportError:
        raise LLMConfigurationError("Install requirements-snowflake.txt for key-pair authentication.") from None
    path = Path(settings.snowflake_private_key_file).expanduser()
    try:
        with path.open("rb") as key_file:
            if os.name == "posix" and os.fstat(key_file.fileno()).st_mode & 0o077:
                raise LLMConfigurationError("Private key permissions are too broad; use chmod 600 on the key file.")
            pem = key_file.read()
        secret = settings.snowflake_private_key_passphrase
        passphrase = secret.get_secret_value().encode() if secret and secret.get_secret_value() else None
        key = serialization.load_pem_private_key(pem, password=passphrase)
    except (OSError, ValueError, TypeError):
        raise LLMConfigurationError("Unable to load the Snowflake private key. Check the file and passphrase locally.") from None
    if not isinstance(key, rsa.RSAPrivateKey) or key.key_size < 2048:
        raise LLMConfigurationError("Snowflake requires an RSA private key of at least 2048 bits.")
    return {
        "authenticator": "SNOWFLAKE_JWT",
        "private_key": key.private_bytes(
            serialization.Encoding.DER,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
    }


def create_snowflake_engine(settings: Any, *, role: str | None = None) -> Engine:
    """Build a pooled synchronous engine for Snowflake's Python driver."""
    required = {
        "SNOWFLAKE_ACCOUNT": settings.snowflake_account,
        "SNOWFLAKE_USER": settings.snowflake_user,
        "SNOWFLAKE_WAREHOUSE": settings.snowflake_warehouse,
        "SNOWFLAKE_DATABASE": settings.snowflake_database,
        "SNOWFLAKE_SCHEMA": settings.snowflake_schema,
    }
    missing = [name for name, value in required.items() if not value or not value.strip()]
    if missing:
        raise LLMConfigurationError("Missing Snowflake configuration: " + ", ".join(missing))
    authentication = _authentication_parameters(settings)
    try:
        from snowflake.sqlalchemy import URL
        from sqlalchemy import create_engine
    except ImportError as exc:
        raise LLMConfigurationError(
            "Snowflake analytics requires the optional dependency. Run "
            "'python -m pip install -r requirements-snowflake.txt'."
        ) from exc

    # Cache by credential fingerprint, not raw secrets; key rotation gets a new pool.
    credential = authentication.get("private_key") or authentication["password"].encode()
    engine_key = (
        settings.snowflake_account,
        settings.snowflake_user,
        settings.snowflake_auth_method,
        sha256(credential).digest(),
        settings.snowflake_warehouse,
        settings.snowflake_database,
        settings.snowflake_schema,
        role or settings.snowflake_role,
        settings.snowflake_pool_size,
        settings.snowflake_max_overflow,
        settings.snowflake_timeout_seconds,
    )
    if engine_key in _SNOWFLAKE_ENGINES:
        return _SNOWFLAKE_ENGINES[engine_key]

    engine = create_engine(
        URL(
            account=settings.snowflake_account,
            user=settings.snowflake_user,
            warehouse=settings.snowflake_warehouse,
            database=settings.snowflake_database,
            schema=settings.snowflake_schema,
            role=role or settings.snowflake_role,
        ),
        pool_pre_ping=True,
        pool_size=settings.snowflake_pool_size,
        max_overflow=settings.snowflake_max_overflow,
        connect_args={
            **authentication,
            "login_timeout": settings.snowflake_timeout_seconds,
            "network_timeout": settings.snowflake_timeout_seconds,
            "session_parameters": {
                "QUERY_TAG": "smart_ai_support",
                "STATEMENT_TIMEOUT_IN_SECONDS": settings.snowflake_timeout_seconds,
            },
        },
    )
    _SNOWFLAKE_ENGINES[engine_key] = engine
    return engine


def dispose_snowflake_engines() -> None:
    """Close every cached Snowflake connection pool during app shutdown."""
    for engine in _SNOWFLAKE_ENGINES.values():
        engine.dispose()
    _SNOWFLAKE_ENGINES.clear()


@dataclass(frozen=True, kw_only=True)
class AnalyticsResult:
    intent: AnalyticsIntent
    rows: list[dict[str, Any]]
    source: str
    elapsed_ms: float
    query_id: str | None = None


QueryRunner = Callable[
    [str, Mapping[str, object]],
    tuple[list[dict[str, Any]], str | None],
]


class SnowflakeAnalyticsService:
    """Run allowlisted, parameterized analytics queries against Snowflake."""

    QUERY_TEMPLATES = {
        AnalyticsIntent.TOP_PRODUCTS_BY_REVENUE: """
            SELECT
                p.PRODUCTID AS "product_id",
                p.PRODUCTNAME AS "product_name",
                ROUND(SUM(od.UNITPRICE * od.QUANTITY * (1 - od.DISCOUNT)), 2)
                    AS "revenue",
                SUM(od.QUANTITY) AS "units_sold"
            FROM ORDERS o
            JOIN ORDER_DETAILS od ON od.ORDERID = o.ORDERID
            JOIN PRODUCTS p ON p.PRODUCTID = od.PRODUCTID
            WHERE (:start_date IS NULL OR o.ORDERDATE >= :start_date)
              AND (:end_date IS NULL OR o.ORDERDATE < DATEADD(day, 1, :end_date))
            GROUP BY p.PRODUCTID, p.PRODUCTNAME
            ORDER BY "revenue" DESC, "product_id"
            LIMIT :limit
        """,
        AnalyticsIntent.MONTHLY_SALES_TREND: """
            SELECT
                TO_CHAR(DATE_TRUNC('month', o.ORDERDATE), 'YYYY-MM') AS "month",
                ROUND(SUM(od.UNITPRICE * od.QUANTITY * (1 - od.DISCOUNT)), 2)
                    AS "revenue",
                SUM(od.QUANTITY) AS "units_sold"
            FROM ORDERS o
            JOIN ORDER_DETAILS od ON od.ORDERID = o.ORDERID
            WHERE (:start_date IS NULL OR o.ORDERDATE >= :start_date)
              AND (:end_date IS NULL OR o.ORDERDATE < DATEADD(day, 1, :end_date))
            GROUP BY DATE_TRUNC('month', o.ORDERDATE)
            ORDER BY DATE_TRUNC('month', o.ORDERDATE)
            LIMIT :limit
        """,
        AnalyticsIntent.SUPPLIER_PERFORMANCE: """
            SELECT
                s.SUPPLIERID AS "supplier_id",
                s.COMPANYNAME AS "supplier_name",
                ROUND(SUM(od.UNITPRICE * od.QUANTITY * (1 - od.DISCOUNT)), 2)
                    AS "revenue",
                SUM(od.QUANTITY) AS "units_sold"
            FROM ORDERS o
            JOIN ORDER_DETAILS od ON od.ORDERID = o.ORDERID
            JOIN PRODUCTS p ON p.PRODUCTID = od.PRODUCTID
            JOIN SUPPLIERS s ON s.SUPPLIERID = p.SUPPLIERID
            WHERE (:start_date IS NULL OR o.ORDERDATE >= :start_date)
              AND (:end_date IS NULL OR o.ORDERDATE < DATEADD(day, 1, :end_date))
            GROUP BY s.SUPPLIERID, s.COMPANYNAME
            ORDER BY "revenue" DESC, "supplier_id"
            LIMIT :limit
        """,
        AnalyticsIntent.CATEGORY_PERFORMANCE: """
            SELECT
                c.CATEGORYID AS "category_id",
                c.CATEGORYNAME AS "category_name",
                ROUND(SUM(od.UNITPRICE * od.QUANTITY * (1 - od.DISCOUNT)), 2)
                    AS "revenue",
                SUM(od.QUANTITY) AS "units_sold"
            FROM ORDERS o
            JOIN ORDER_DETAILS od ON od.ORDERID = o.ORDERID
            JOIN PRODUCTS p ON p.PRODUCTID = od.PRODUCTID
            JOIN CATEGORIES c ON c.CATEGORYID = p.CATEGORYID
            WHERE (:start_date IS NULL OR o.ORDERDATE >= :start_date)
              AND (:end_date IS NULL OR o.ORDERDATE < DATEADD(day, 1, :end_date))
            GROUP BY c.CATEGORYID, c.CATEGORYNAME
            ORDER BY "revenue" DESC, "category_id"
            LIMIT :limit
        """,
    }

    def __init__(
        self,
        *,
        database: str,
        schema: str,
        engine: Engine | None = None,
        query_runner: QueryRunner | None = None,
    ) -> None:
        if engine is None and query_runner is None:
            raise ValueError("engine or query_runner is required")
        self._engine = engine
        self._query_runner = query_runner or self._run_with_engine
        self.source = f"snowflake://{database}.{schema}"

    @classmethod
    def from_settings(cls, settings: Any) -> SnowflakeAnalyticsService:
        if not settings.snowflake_enabled:
            raise LLMConfigurationError(
                "Snowflake analytics is disabled. Set SNOWFLAKE_ENABLED=true."
            )
        engine = create_snowflake_engine(settings)
        return cls(
            engine=engine,
            database=settings.snowflake_database,
            schema=settings.snowflake_schema,
        )

    @classmethod
    def build_query(
        cls,
        plan: AnalyticsQueryPlan,
    ) -> tuple[str, dict[str, date | int | None]]:
        return cls.QUERY_TEMPLATES[plan.intent], {
            "start_date": plan.start_date,
            "end_date": plan.end_date,
            "limit": plan.limit,
        }

    async def query(self, plan: AnalyticsQueryPlan) -> AnalyticsResult:
        sql, parameters = self.build_query(plan)
        started = perf_counter()
        rows, query_id = await asyncio.to_thread(
            self._query_runner,
            sql,
            parameters,
        )
        return AnalyticsResult(
            intent=plan.intent,
            rows=rows,
            source=self.source,
            elapsed_ms=round((perf_counter() - started) * 1000, 2),
            query_id=query_id,
        )

    def _run_with_engine(
        self,
        sql: str,
        parameters: Mapping[str, object],
    ) -> tuple[list[dict[str, Any]], str | None]:
        if self._engine is None:  # pragma: no cover - constructor prevents this.
            raise RuntimeError("Snowflake engine is not configured")
        with self._engine.connect() as connection:
            result = connection.execute(text(sql), dict(parameters))
            # SQLAlchemy may detach the cursor when all rows are consumed.
            query_id = getattr(getattr(result, "cursor", None), "sfqid", None)
            rows = [dict(row) for row in result.mappings().all()]
            return rows, query_id
