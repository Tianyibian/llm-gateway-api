"""Compile an allowlisted connected graph pattern to parameterized read-only Cypher."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

from app.models.cypher import CypherPlan
from app.services.graphrag_guardrail import GraphRAGGuardrail

SCHEMA_VERSION = "aster-business-graph-v2"
EDGES = {
    ("Product", "SUPPLIED_BY", "Supplier"),
    ("Product", "BELONGS_TO", "Category"),
    ("Order", "CONTAINS", "OrderLine"),
    ("OrderLine", "OF_PRODUCT", "Product"),
}
SCHEMA = """Nodes: Product, Supplier, Category, Order, OrderLine.
All nodes have id and dataset. Product/Supplier/Category have name and source.
Order has order_date (date). OrderLine has units (integer) and
net_amount_micros (integer net revenue in millionths of currency, including discount).
Edges: Product-SUPPLIED_BY->Supplier; Product-BELONGS_TO->Category;
Order-CONTAINS->OrderLine; OrderLine-OF_PRODUCT->Product.
No customer, employee, contact, inventory, review, policy or free-text properties.
"""


@dataclass(frozen=True)
class CompiledCypher:
    cypher: str
    parameters: dict[str, str | int]
    result_limit: int
    target_label: str
    metric: str
    grouping: str


def compile_plan(plan: CypherPlan, *, question: str, dataset: str) -> CompiledCypher:
    # Revalidate copied/constructed models so callers cannot bypass the contract.
    plan = CypherPlan.model_validate(plan.model_dump())
    if plan.action != "query" or not plan.nodes:
        raise ValueError("Unsupported graph question")
    if plan.metric != "count" and plan.count_node is not None:
        raise ValueError("count_node is only valid for count metrics")
    nodes = {node.key: node.label for node in plan.nodes}
    if len(nodes) != len(plan.nodes) or plan.target not in nodes:
        raise ValueError("Duplicate nodes or missing target")
    if len(plan.edges) != len(nodes) - 1:
        raise ValueError("Only one connected tree pattern is allowed")
    adjacency = {key: set() for key in nodes}
    patterns = []
    for edge in plan.edges:
        if edge.source not in nodes or edge.target not in nodes:
            raise ValueError("Unknown node reference")
        if (nodes[edge.source], edge.relation, nodes[edge.target]) not in EDGES:
            raise ValueError("Unknown or reversed relationship")
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
        patterns.append(f"({edge.source}:{nodes[edge.source]})-[:{edge.relation}]->({edge.target}:{nodes[edge.target]})")
    seen, pending = set(), [next(iter(nodes))]
    while pending:
        key = pending.pop()
        if key not in seen:
            seen.add(key)
            pending.extend(adjacency[key] - seen)
    if seen != set(nodes):
        raise ValueError("Disconnected graph pattern")
    if not patterns:
        key = next(iter(nodes))
        patterns = [f"({key}:{nodes[key]})"]
    parameters: dict[str, str | int] = {"dataset": dataset, "limit": plan.limit + 1}
    conditions = [f"{key}.dataset = $dataset" for key in nodes]
    for index, item in enumerate(plan.filters):
        if nodes.get(item.node) not in {"Product", "Supplier", "Category"}:
            raise ValueError("Only public catalog names can be filtered")
        if not GraphRAGGuardrail._mentioned(question, item.value):
            raise ValueError("Filter entity was not mentioned in the question")
        parameter = f"name{index}"
        parameters[parameter] = item.value
        operator = "=" if item.operator == "equals" else "CONTAINS"
        conditions.append(f"toLower({item.node}.name) {operator} toLower(${parameter})")
    if plan.exclude_same_products:
        products = [key for key, label in nodes.items() if label == "Product"]
        if len(products) < 2:
            raise ValueError("Exclusion requires two Product nodes")
        conditions.extend(f"{a}.id <> {b}.id" for a, b in combinations(products, 2))
    orders = [key for key, label in nodes.items() if label == "Order"]
    lines = [key for key, label in nodes.items() if label == "OrderLine"]
    if len(orders) > 1 or len(lines) > 1:
        raise ValueError("At most one order and order-line binding is allowed")
    if plan.start_date and plan.end_date and plan.start_date > plan.end_date:
        raise ValueError("Reversed date range")
    for name, value, operator in [("start_date", plan.start_date, ">="), ("end_date", plan.end_date, "<=")]:
        if value:
            if not orders:
                raise ValueError("Date filter requires Order")
            parameters[name] = value.isoformat()
            conditions.append(f"{orders[0]}.order_date {operator} date(${name})")
    target = plan.target
    if plan.grouping == "month" and not orders:
        raise ValueError("Monthly grouping requires Order")
    if plan.grouping == "entity" and nodes[target] not in {"Product", "Supplier", "Category"}:
        raise ValueError("Entity grouping requires a catalog target")
    query = "MATCH " + ", ".join(patterns) + "\nWHERE " + " AND ".join(conditions)
    identity = f"{target}.id AS id, {target}.name AS name, {target}.source AS source"
    if plan.metric == "records":
        if plan.grouping != "none" or nodes[target] not in {"Product", "Supplier", "Category"}:
            raise ValueError("Only public catalog records can be returned")
        query += f"\nRETURN DISTINCT {identity}\nORDER BY id LIMIT $limit"
    else:
        if plan.metric in {"revenue", "units"}:
            if len(lines) != 1:
                raise ValueError("Sales metrics require OrderLine")
            measure_node = lines[0]
            metric = (f"sum({measure_node}.net_amount_micros) AS revenue_micros, "
                      f"sum({measure_node}.units) AS units_sold")
            ordering = "revenue_micros" if plan.metric == "revenue" else "units_sold"
        else:
            if plan.count_node not in nodes:
                raise ValueError("Count requires a known count_node")
            measure_node = plan.count_node
            metric, ordering = f"count(DISTINCT {measure_node}) AS count", "count"
        kept = list(dict.fromkeys([measure_node] + ([target] if plan.grouping == "entity" else []) + (orders if plan.grouping == "month" else [])))
        query += "\nWITH DISTINCT " + ", ".join(kept)
        if plan.grouping == "entity":
            query += f"\nRETURN {identity}, {metric}\nORDER BY {ordering} DESC, id LIMIT $limit"
        elif plan.grouping == "month":
            query += f"\nRETURN substring(toString({orders[0]}.order_date), 0, 7) AS month, {metric}\nORDER BY month LIMIT $limit"
        else:
            query += f"\nRETURN {metric}\nLIMIT $limit"
    return CompiledCypher(query, parameters, plan.limit, nodes[target], plan.metric, plan.grouping)
