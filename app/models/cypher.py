"""A deliberately small graph query language, not model-authored executable code."""
from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import Field

from app.models.graphrag import GraphContract

NodeKey = Literal["n0", "n1", "n2", "n3", "n4", "n5"]
NodeLabel = Literal["Product", "Supplier", "Category", "Order", "OrderLine"]
EdgeLabel = Literal["SUPPLIED_BY", "BELONGS_TO", "CONTAINS", "OF_PRODUCT"]
TemplateName = Literal["product_supplier", "category_products", "shared_supplier_products", "revenue_by_product", "revenue_by_supplier", "monthly_sales"]


class CypherSelection(GraphContract):
    """Choose an exact template match, not a merely similar query."""
    route: Literal["template", "text_to_cypher", "unsupported"]
    template: TemplateName | None
    name: str | None = Field(max_length=200)
    year: int | None = Field(ge=1900, le=2100)
    limit: int = Field(ge=1, le=20)


class QueryNode(GraphContract):
    key: NodeKey
    label: NodeLabel


class QueryEdge(GraphContract):
    source: NodeKey
    relation: EdgeLabel
    target: NodeKey


class NameFilter(GraphContract):
    node: NodeKey
    operator: Literal["equals", "contains"]
    value: str = Field(min_length=1, max_length=200)


class CypherPlan(GraphContract):
    action: Literal["query", "unsupported"]
    nodes: list[QueryNode] = Field(max_length=6)
    edges: list[QueryEdge] = Field(max_length=5)
    filters: list[NameFilter] = Field(max_length=6)
    target: NodeKey | None = Field(description="Existing node key for every query, including ungrouped totals; null only for unsupported.")
    count_node: NodeKey | None
    metric: Literal["records", "count", "revenue", "units"]
    grouping: Literal["none", "entity", "month"]
    start_date: date | None
    end_date: date | None
    limit: int = Field(ge=1, le=20)
    exclude_same_products: bool


class CypherExecution(GraphContract):
    query_mode: Literal["template", "text_to_cypher"]
    template_id: TemplateName | None
    cypher: str = Field(max_length=12000)
    parameters: dict[str, str | int]
    row_count: int = Field(ge=0, le=20)
    truncated: bool
    checks: list[str] = Field(default_factory=list)


class CypherReview(GraphContract):
    matches_question: bool
    preserves_all_constraints: bool
    reason_code: Literal["approved", "wrong_metric", "missing_filter", "wrong_relationship", "wrong_dates", "incomplete_request", "unsupported_request"]


class CypherResult(GraphContract):
    status: Literal["complete", "unsupported", "unavailable", "rejected"]
    reason_code: str
    execution: CypherExecution | None = None
    rows: list[dict] = Field(default_factory=list, max_length=20)
