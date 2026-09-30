"""Deterministic resolution and safety contracts; fake catalog/model inputs."""
import asyncio
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from app.services.cypher_compiler import compile_plan
from app.services.entity_resolution import EntityBinding, candidates
from app.services.errors import EntityClarificationRequired
from tests.test_neo4j_queries import plan, selection, service
from tests.test_graph_supervisor import Plans, retrieve, task, service as supervisor
from tests.test_cypher_subgraph import executor_with_session


CATALOG = [{"id": "Category:7", "name": "Smart Speaker"}]


def category_plan(term="speakers"):
    return plan(nodes=[{"key": "n0", "label": "Product"}, {"key": "n1", "label": "Category"}],
                edges=[{"source": "n0", "relation": "BELONGS_TO", "target": "n1"}],
                filters=[{"node": "n1", "operator": "contains", "value": term}],
                target="n0", count_node=None, metric="records", grouping="none")


@pytest.mark.parametrize("surface,name", [
    ("speakers", "Smart Speaker"), ("speaker", "Smart Speaker"),
    ("CAMERAS", "Smart Camera"), ("ＳＰＥＡＫＥＲＳ", "Smart Speaker"),
    ("categories", "Category"), ("switches", "Smart Switch"),
])
def test_regular_variants_match_real_names(surface, name):
    rows = [{"id": "test", "name": name}]
    assert candidates(surface, rows) == rows


def test_no_substring_alias_or_fabricated_match():
    assert candidates("ring", [{"id": "1", "name": "String Light"}]) == []
    assert candidates("computers", CATALOG) == []
    assert candidates("audio devices", CATALOG) == []  # No unverified synonym mapping.


def test_exact_name_wins_and_partial_ambiguity_is_preserved():
    rows = CATALOG + [{"id": "Category:8", "name": "Outdoor Speaker"}]
    assert candidates("speakers", rows) == rows
    assert candidates("Smart Speaker", rows) == CATALOG


def test_plural_resolves_to_db_id_without_templates_or_question_rewrite():
    svc, selector, chain, executor = service(generated=category_plan(), rows=[{"id": "Product:1", "name": "Example"}])
    executor.catalog_entities.return_value = CATALOG
    query = "what type of speakers do you have"
    result = asyncio.run(svc.query(query, strategy="text_to_cypher"))
    assert result.status == "complete"
    selector.ainvoke.assert_not_awaited()
    assert chain.ainvoke.call_args.args[0]["question"] == query
    assert result.execution.template_id is None
    assert result.execution.query_mode == "text_to_cypher"
    assert result.execution.parameters["entity0"] == "Category:7"
    assert "n1.id = $entity0" in result.execution.cypher
    binding = result.execution.entity_resolutions[0]
    assert binding["surface"] == "speakers" and binding["name"] == "Smart Speaker"
    assert "Smart Speaker" in svc.reviewer.ainvoke.call_args.args[0]["entity_bindings"]
    executor.explain.assert_awaited_once()


def test_ambiguity_asks_user_and_never_runs_final_query():
    svc, _, _, executor = service(generated=category_plan())
    executor.catalog_entities.return_value = CATALOG + [{"id": "Category:8", "name": "Outdoor Speaker"}]
    result = asyncio.run(svc.query("Which speakers?", strategy="text_to_cypher"))
    assert result.status == "clarify"
    assert "Smart Speaker" in result.clarification and "Outdoor Speaker" in result.clarification
    assert executor.run.await_count == 1  # Readiness only; candidate enumeration is separately mocked.
    svc.reviewer.ainvoke.assert_not_awaited()
    executor.explain.assert_not_awaited()


def test_no_candidate_preserves_original_filter_and_zero_rows():
    svc, _, _, executor = service(generated=category_plan("computers"))
    executor.catalog_entities.return_value = CATALOG
    result = asyncio.run(svc.query("Which computers?", strategy="text_to_cypher"))
    assert result.status == "complete" and result.rows == []
    assert result.execution.parameters["name0"] == "computers"
    assert result.execution.entity_resolutions == []


def test_invented_canonical_name_cannot_bypass_original_word_provenance():
    svc, _, _, executor = service(generated=category_plan("Smart Speaker"))
    result = asyncio.run(svc.query("Which speakers?", strategy="text_to_cypher"))
    assert result.status == "rejected"
    executor.catalog_entities.assert_not_awaited()


@pytest.mark.parametrize("patch", [
    {"surface": "other"}, {"label": "Supplier"}, {"dataset": "another-tenant"},
    {"filter_index": 9}, {"entity_id": ""},
])
def test_binding_must_match_filter_type_surface_and_dataset(patch):
    binding = EntityBinding(0, "speakers", "Category", "Category:7", "Smart Speaker", "test")
    with pytest.raises(ValueError):
        compile_plan(category_plan(), question="Which speakers?", dataset="test",
                     observed_edges=frozenset({("Product", "BELONGS_TO", "Category")}),
                     bindings=(replace(binding, **patch),))


def test_resolution_does_not_skip_semantic_validation():
    svc, _, _, executor = service(generated=category_plan())
    executor.catalog_entities.return_value = CATALOG
    svc.reviewer.ainvoke.return_value = {"matches_question": False, "preserves_all_constraints": False,
                                       "reason_code": "missing_filter"}
    result = asyncio.run(svc.query("Which speakers?", strategy="text_to_cypher"))
    assert result.reason_code == "cypher_semantic_check_failed"
    executor.explain.assert_not_awaited()
    assert executor.run.await_count == 1


@pytest.mark.parametrize("strategy", ["auto", "template"])
def test_category_products_is_not_executed_even_if_selector_requests_it(strategy):
    svc, _, chain, executor = service(selected=selection(template="category_products", name="speakers"),
                                    generated=category_plan())
    executor.catalog_entities.return_value = CATALOG
    result = asyncio.run(svc.query("Which speakers?", strategy=strategy))
    if strategy == "auto":
        assert result.status == "complete" and result.execution.template_id is None
        assert chain.ainvoke.await_count == 1
    else:
        assert result.status == "unsupported" and executor.run.await_count == 1


def test_supervisor_delivers_clarification_without_second_scope_or_answer_generation():
    tool = AsyncMock()
    tool.retrieve.side_effect = EntityClarificationRequired("Which category: Smart Speaker or Outdoor Speaker?")
    engine, _ = supervisor(Plans(retrieve(task(question="Which speakers?"))), tool=tool)
    engine.answer_generator = AsyncMock()
    result = asyncio.run(engine.run("Which speakers?"))
    assert result.status == "clarify" and result.reason_code == "entity_ambiguous"
    assert "Outdoor Speaker" in result.answer
    assert engine.guardrail.calls == ["Which speakers?"]
    assert result.trace[0]["error"] == "entity_ambiguous"
    engine.answer_generator.generate.assert_not_awaited()


@pytest.mark.parametrize("rows", [
    [{"id": str(i), "name": "Item"} for i in range(2001)],
    [{"id": "same", "name": "A"}, {"id": "same", "name": "B"}],
    [{"id": "1", "name": None}],
])
def test_catalog_lookup_fails_closed_on_truncation_duplicate_ids_or_invalid_data(rows):
    executor, _ = executor_with_session("r")
    executor.run = AsyncMock(return_value=(rows, {}))
    with pytest.raises(ValueError):
        asyncio.run(executor.catalog_entities("Category"))
    compiled = executor.run.call_args.args[0]
    assert compiled.parameters == {"dataset": "test", "limit": 2001}
    assert "WHERE n.dataset = $dataset" in compiled.cypher


def test_entity_catalog_rejects_non_allowlisted_label_before_database_access():
    executor, _ = executor_with_session("r")
    executor.run = AsyncMock()
    with pytest.raises(ValueError):
        asyncio.run(executor.catalog_entities("Customer"))
    executor.run.assert_not_awaited()


def test_adapter_propagates_ambiguity_as_clarification_not_provider_failure():
    svc, _, _, executor = service(generated=category_plan())
    executor.catalog_entities.return_value = CATALOG + [{"id": "Category:8", "name": "Outdoor Speaker"}]
    with pytest.raises(EntityClarificationRequired, match="Outdoor Speaker"):
        asyncio.run(svc.retrieve("Which speakers?", limit=3, strategy="text_to_cypher"))
