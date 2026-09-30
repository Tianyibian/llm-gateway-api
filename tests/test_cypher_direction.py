"""Directions come from observed triples, not the bundled import conventions."""
import asyncio
from dataclasses import replace

import pytest

from app.services.cypher_compiler import compile_plan, EDGES
from app.services.cypher_templates import compile_template, TEMPLATES
from app.services.neo4j_schema import ObservedGraphSchema
from tests.test_neo4j_schema import node
from tests.test_neo4j_queries import plan, selection, service


FORWARD = ("Product", "SUPPLIED_BY", "Supplier")
REVERSE = ("Supplier", "SUPPLIED_BY", "Product")


def test_discovery_retains_reversed_actual_triple_but_not_unrelated_endpoints():
    schema = ObservedGraphSchema.from_rows("test", [node(), node("Supplier"), node("Order")], [
        {"sources": ["Supplier"], "relation": "SUPPLIED_BY", "targets": ["Product"]},
        {"sources": ["Order"], "relation": "SUPPLIED_BY", "targets": ["Product"]},
    ])
    assert schema.edges == frozenset({REVERSE})


@pytest.mark.parametrize("actual,arrow", [(FORWARD, "-[:SUPPLIED_BY]->"), (REVERSE, "<-[:SUPPLIED_BY]-")])
def test_dynamic_compiler_uses_database_direction(actual, arrow):
    compiled = compile_plan(plan(), question="Count products by supplier", dataset="test",
                            observed_edges=frozenset({actual}))
    assert f"(n0:Product){arrow}(n1:Supplier)" in compiled.cypher


def test_model_reversed_edge_is_aligned_to_actual_direction():
    generated = plan(edges=[{"source": "n1", "relation": "SUPPLIED_BY", "target": "n0"}])
    compiled = compile_plan(generated, question="Count products by supplier", dataset="test",
                            observed_edges=frozenset({FORWARD}))
    assert "(n1:Supplier)<-[:SUPPLIED_BY]-(n0:Product)" in compiled.cypher


@pytest.mark.parametrize("strategy", ["template", "text_to_cypher"])
@pytest.mark.parametrize("edges", [frozenset(), frozenset({FORWARD, REVERSE})])
def test_missing_or_ambiguous_direction_never_executes(strategy, edges):
    svc, _, _, executor = service()
    executor.schema.return_value = replace(executor.schema.return_value, edges=edges)
    result = asyncio.run(svc.query("Who supplies Acme Sensor?", strategy=strategy))
    assert result.status == "rejected"
    svc.reviewer.ainvoke.assert_not_awaited()
    assert executor.run.await_count == 1


@pytest.mark.parametrize("strategy", ["template", "text_to_cypher"])
def test_both_service_paths_adapt_to_reversed_database(strategy):
    svc, _, _, executor = service()
    executor.schema.return_value = replace(executor.schema.return_value, edges=frozenset({REVERSE}))
    result = asyncio.run(svc.query("Who supplies Acme Sensor?", strategy=strategy))
    assert result.status == "complete"
    assert "<-[:SUPPLIED_BY]-" in result.execution.cypher
    assert '["Supplier", "SUPPLIED_BY", "Product"]' in svc.reviewer.ainvoke.await_args.args[0]["schema"]


@pytest.mark.parametrize("template", list(TEMPLATES))
def test_templates_keep_filters_parameters_and_handle_overlapping_links(template):
    name = "Acme" if template in {"product_supplier", "shared_supplier_products", "category_products"} else None
    selected = selection(template=template, name=name)
    original = compile_template(selected, question="Acme", dataset="test")
    svc, _, _, executor = service()
    schema = replace(executor.schema.return_value, edges=frozenset((b, r, a) for a, r, b in EDGES))
    adapted = schema.orient_template(original)
    schema.validate_compiled(adapted)
    assert adapted.parameters == original.parameters
    assert adapted.result_limit == original.result_limit
    assert adapted.cypher.split("\nWHERE ")[1] == original.cypher.split("\nWHERE ")[1]
    with pytest.raises(ValueError, match="direction"):
        schema.validate_compiled(original)
    assert schema.orient_template(adapted) == adapted


def test_right_relation_name_wrong_endpoint_pair_is_rejected():
    _, _, _, executor = service()
    original = compile_template(selection(), question="Acme Sensor", dataset="test")
    wrong = replace(original, cypher=original.cypher.replace("s:Supplier", "s:Category"))
    with pytest.raises(ValueError, match="endpoints"):
        executor.schema.return_value.validate_compiled(wrong)


def test_second_link_in_shared_supplier_template_is_checked():
    _, _, _, executor = service()
    original = compile_template(selection(template="shared_supplier_products"), question="Acme Sensor", dataset="test")
    wrong = replace(original, cypher=original.cypher.replace("<-[:SUPPLIED_BY]-", "-[:SUPPLIED_BY]->"))
    with pytest.raises(ValueError, match="direction"):
        executor.schema.return_value.validate_compiled(wrong)


def test_dynamic_compilation_requires_observed_edges_no_static_fallback():
    with pytest.raises(TypeError):
        compile_plan(plan(), question="Count products by supplier", dataset="test")
