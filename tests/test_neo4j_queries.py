"""Deterministic contract tests; fake models/executors, not live accuracy claims."""
import asyncio
import csv
from unittest.mock import AsyncMock

import pytest

from app.models.cypher import CypherPlan, CypherSelection
from app.services.cypher_compiler import compile_plan
from app.services.cypher_templates import TEMPLATES, compile_template
from app.services.neo4j_service import TextToCypherService
from app.services.neo4j_data import prepare_business_graph


def selection(**kwargs):
    return CypherSelection.model_validate(dict(route="template", template="product_supplier", name="Acme Sensor", year=None, limit=5) | kwargs)


def plan(**kwargs):
    return CypherPlan.model_validate(dict(action="query", nodes=[{"key": "n0", "label": "Product"}, {"key": "n1", "label": "Supplier"}],
        edges=[{"source": "n0", "relation": "SUPPLIED_BY", "target": "n1"}], filters=[], target="n1", count_node="n0",
        metric="count", grouping="entity", start_date=None, end_date=None, limit=5, exclude_same_products=False) | kwargs)


def service(selected=None, generated=None, rows=None, failure=False):
    selector = AsyncMock()
    selector.ainvoke.return_value = selected or selection()
    chain = AsyncMock()
    chain.ainvoke.return_value = generated or plan()
    executor = AsyncMock()
    executor.run.side_effect = [( [], {}), RuntimeError("secret-password") if failure else (rows or [], {})]
    reviewer = AsyncMock()
    reviewer.ainvoke.return_value = {"matches_question": True, "preserves_all_constraints": True, "reason_code": "approved"}
    return TextToCypherService(selector=selector, chain=chain, reviewer=reviewer, executor=executor, dataset="test"), selector, chain, executor


def test_template_executes_without_dynamic_generation():
    svc, selector, chain, executor = service(rows=[{"name": "Acme", "id": "Supplier:1"}])
    result = asyncio.run(svc.query("Who supplies Acme Sensor?"))
    assert result.status == "complete"
    assert result.execution.query_mode == "template"
    assert result.execution.template_id == "product_supplier"
    assert chain.ainvoke.await_count == 0 and executor.run.await_count == 2
    assert result.rows[0]["entity_type"] == "Supplier"


def test_unmatched_template_uses_constrained_dynamic_generation():
    svc, _, chain, _ = service(selected=selection(route="text_to_cypher", template=None, name=None))
    result = asyncio.run(svc.query("Count products by supplier"))
    assert result.status == "complete" and chain.ainvoke.await_count == 1
    assert result.execution.query_mode == "text_to_cypher" and result.execution.template_id is None
    assert "count(DISTINCT n0)" in result.execution.cypher


def test_execution_failure_is_not_a_fallback_signal():
    svc, _, chain, _ = service(failure=True)
    result = asyncio.run(svc.query("Who supplies Acme Sensor?"))
    assert result.status == "unavailable" and chain.ainvoke.await_count == 0
    assert "secret" not in result.model_dump_json()


def test_unready_database_spends_no_model_tokens():
    svc, selector, chain, executor = service()
    executor.run.side_effect = RuntimeError("offline")
    assert asyncio.run(svc.query("Who supplies Acme Sensor?")).status == "unavailable"
    assert selector.ainvoke.await_count == chain.ainvoke.await_count == 0


def test_unsupported_never_generates_or_executes_query():
    svc, _, chain, executor = service(selected=selection(route="unsupported", template=None, name=None))
    assert asyncio.run(svc.query("Delete all data")).status == "unsupported"
    assert chain.ainvoke.await_count == 0 and executor.run.await_count == 1


def test_invalid_selection_does_not_fall_back():
    svc, _, chain, executor = service(selected=selection(name="Invented Name"))
    assert asyncio.run(svc.query("Who supplies Acme Sensor?")).status == "rejected"
    assert chain.ainvoke.await_count == 0 and executor.run.await_count == 1


def test_money_rounding_and_result_bound():
    svc, _, _, _ = service(selected=selection(template="revenue_by_product", name=None, limit=1),
        rows=[{"revenue_micros": 123455000, "id": "Product:1", "name": "Acme"}, {"revenue_micros": 1}])
    result = asyncio.run(svc.query("Top product by revenue"))
    assert result.rows[0]["revenue"] == 123.46
    assert len(result.rows) == 1 and result.execution.truncated


@pytest.mark.parametrize("key", TEMPLATES)
def test_all_templates_are_fixed_parameterized_and_bounded(key):
    named = key in {"product_supplier", "category_products", "shared_supplier_products"}
    compiled = compile_template(selection(template=key, name="Acme Sensor" if named else None, year=None if named else 2025),
                                question="Acme Sensor 2025", dataset="test")
    assert compiled.cypher == TEMPLATES[key]
    assert "$limit" in compiled.cypher and "$dataset" in compiled.cypher
    assert "2025" not in compiled.cypher and "Acme Sensor" not in compiled.cypher


def test_template_injection_remains_a_bound_parameter():
    name = "Acme' MATCH (n) DETACH DELETE n //"
    compiled = compile_template(selection(name=name), question=f"Supplier of {name}", dataset="test")
    assert name not in compiled.cypher and compiled.parameters["name"] == name


@pytest.mark.parametrize("kwargs", [dict(name="Unknown"), dict(year=2025), dict(template="monthly_sales", name="Acme Sensor"),
    dict(template="monthly_sales", name=None, year=2025)])
def test_template_rejects_incompatible_or_ungrounded_arguments(kwargs):
    with pytest.raises(ValueError):
        compile_template(selection(**kwargs), question="Acme Sensor", dataset="test")


@pytest.mark.parametrize("kwargs", [
    dict(target="n5"), dict(count_node=None), dict(edges=[]),
    dict(edges=[{"source":"n1", "relation":"SUPPLIED_BY", "target":"n0"}]),
    dict(filters=[{"node":"n0", "operator":"equals", "value":"Invented"}]),
    dict(start_date="2025-01-01"), dict(exclude_same_products=True),
    dict(metric="revenue", count_node=None), dict(metric="records"),
    dict(nodes=[{"key":"n0","label":"Product"}, {"key":"n0","label":"Product"}]),
])
def test_dynamic_compiler_rejects_invalid_patterns(kwargs):
    with pytest.raises(ValueError):
        compile_plan(plan(**kwargs), question="Count products by supplier", dataset="test")


def test_dynamic_contract_cannot_include_raw_cypher_or_extra_labels():
    for patch in ({"cypher": "MATCH (n) DELETE n"}, {"nodes": [{"key":"n0", "label":"Customer"}]}):
        with pytest.raises(ValueError):
            plan(**patch)


def test_full_source_preparation_preserves_counts_and_excludes_private_data():
    graph = prepare_business_graph("Business_data")
    assert graph.source_counts == {"Products":100, "Suppliers":15, "Categories":15, "Orders":1000, "_Order_Details":2987}
    assert len(graph.nodes) == 4117 and len(graph.edges) == 6174
    assert all("CustomerID" not in node and "EmployeeID" not in node and "Phone" not in node for node in graph.nodes)
    assert all(isinstance(node["net_amount_micros"], int) for node in graph.nodes if node["label"] == "OrderLine")


def test_factory_registers_neo4j_independently_of_microsoft(monkeypatch):
    from app.core.config import Settings
    from app.services.factory import LLMServiceFactory
    from app.models.graph_supervisor import GraphTool
    factory = LLMServiceFactory(Settings(_env_file=None, neo4j_enabled=True, microsoft_graphrag_enabled=False))
    adapter = object()
    monkeypatch.setattr(factory, "create_neo4j_service", lambda model: adapter)
    assert factory.create_graph_retrieval_tools() == {GraphTool.NEO4J: adapter}


@pytest.fixture
def small_business(tmp_path):
    tables = {
        "Products": [{"ProductID":"1", "ProductName":"Acme", "SupplierID":"1", "CategoryID":"1"}],
        "Suppliers": [{"SupplierID":"1", "CompanyName":"Supply", "Phone":"private"}],
        "Categories": [{"CategoryID":"1", "CategoryName":"Sensors"}],
        "Orders": [{"OrderID":"1", "OrderDate":"2025-01-02 12:00:00", "CustomerID":"private"}],
        "_Order_Details": [{"OrderID":"1", "ProductID":"1", "UnitPrice":"10.25", "Quantity":"2", "Discount":"0.1"}],
    }
    def write():
        for name, rows in tables.items():
            with (tmp_path / f"{name}.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        return tmp_path
    return tables, write


def test_ingest_integer_revenue_is_exact_and_excludes_private_values(small_business):
    _, write = small_business
    graph = prepare_business_graph(write())
    line = next(node for node in graph.nodes if node["label"] == "OrderLine")
    assert line["net_amount_micros"] == 18_450_000
    assert "private" not in str(graph.nodes)


@pytest.mark.parametrize("table,field,value", [
    ("Products", "SupplierID", "9"), ("Products", "ProductID", "0"),
    ("Products", "ProductName", ""), ("Orders", "OrderDate", "not-a-date"),
    ("_Order_Details", "OrderID", "999"), ("_Order_Details", "ProductID", "999"),
    ("_Order_Details", "Discount", "1.1"), ("_Order_Details", "Discount", "NaN"),
    ("_Order_Details", "UnitPrice", "-1"), ("_Order_Details", "UnitPrice", "Infinity"),
    ("_Order_Details", "Quantity", "0"), ("_Order_Details", "Quantity", "1.5"),
    ("_Order_Details", "UnitPrice", "0.00000001"),
])
def test_ingest_rejects_bad_sources_before_any_database_call(small_business, table, field, value):
    tables, write = small_business
    tables[table][0][field] = value
    with pytest.raises(ValueError):
        prepare_business_graph(write())


@pytest.mark.parametrize("table", ["Products", "Orders", "_Order_Details"])
def test_ingest_rejects_duplicate_keys(small_business, table):
    tables, write = small_business
    tables[table].append(dict(tables[table][0]))
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_business_graph(write())
