"""Product business scope is distinct from records and adapter capabilities."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from app.models.graph_supervisor import GraphPlan, RetrievalEvidence
from app.services.clarification_service import ClarificationService
from app.services.query_classifier import QueryClassifier
from tests.test_graphrag_guardrail import ASSESSMENT, evaluate
from tests.test_graph_supervisor import Plans, finish, retrieve, service, task


@pytest.mark.parametrize("query", [
    "what type of speaker do you have", "Which cameras are in the catalog?",
    "What computers do you sell?", "List all product categories",
    "What is the price of Acme Sensor?", "Is Acme Sensor in stock right now?",
    "What are Acme Sensor's specifications?", "Is Acme Sensor compatible with my hub?",
])
def test_scope_contract_does_not_require_record_existence_or_missing_attribute_types(query):
    decision = evaluate({**ASSESSMENT, "intent": "exploratory", "entity_types": ["Product", "Category"],
                         "entity_mentions": [], "required_relations": []}, query=query, scope_only=True)
    assert decision.action == "allow"
    assert decision.entity_resolution == "not_performed"
    assert decision.retrieval_ready is False


@pytest.mark.parametrize("field", ["current_price", "live_stock", "specifications", "compatibility"])
def test_missing_product_data_gets_explicit_limitation_without_scope_rejection_or_tools(field):
    engine, tool = service(Plans({"action": "data_unavailable", "tasks": [], "evidence_ids": [],
                                 "missing_product_data": [field]}))
    engine.answer_generator = AsyncMock()
    result = asyncio.run(engine.run("Tell me about Acme Sensor"))
    assert result.reason_code == "product_data_unavailable"
    assert result.status == "unavailable"
    assert "Product questions are supported" in result.answer
    assert result.tool_calls == 0 and not tool.calls
    assert engine.guardrail.calls == ["Tell me about Acme Sensor"]
    engine.answer_generator.generate.assert_not_awaited()


@pytest.mark.parametrize("plan", [
    {"action": "data_unavailable", "tasks": [], "evidence_ids": [], "missing_product_data": []},
    {"action": "data_unavailable", "tasks": [task()], "evidence_ids": [], "missing_product_data": ["live_stock"]},
    {"action": "data_unavailable", "tasks": [], "evidence_ids": ["E1"], "missing_product_data": ["live_stock"]},
    {"action": "clarify", "tasks": [], "evidence_ids": [], "missing_product_data": ["live_stock"]},
    {"action": "data_unavailable", "tasks": [], "evidence_ids": [], "missing_product_data": ["all_products"]},
    {"action": "data_unavailable", "tasks": [], "evidence_ids": [], "missing_product_data": ["live_stock"], "price": 599},
])
def test_capability_limitations_cannot_smuggle_values_or_execution(plan):
    with pytest.raises(ValidationError):
        GraphPlan.model_validate(plan)


def test_product_routing_instructions_do_not_reintroduce_a_standalone_branch():
    assert "Product queries belong to graph_rag_search" in QueryClassifier.SYSTEM_PROMPT
    assert "missing price/stock/spec fields" in QueryClassifier.SYSTEM_PROMPT
    assert "Missing backend fields do not make a business" in ClarificationService.SYSTEM_PROMPT
    assert "reject those requests" not in ClarificationService.SYSTEM_PROMPT


def test_empty_snapshot_evidence_is_not_a_second_scope_rejection():
    tool = AsyncMock()
    tool.retrieve.return_value = [RetrievalEvidence(
        source_id="synthetic:empty-catalog", entities=[],
        text="No matching records in this snapshot. This is not live stock information.",
    )]
    query = "What computers do you sell?"
    engine, _ = service(Plans(retrieve(task(question=query)), finish("E1")), tool=tool)
    result = asyncio.run(engine.run(query))
    assert result.status == "complete"
    assert result.answer_evidence_ids == ["E1"]
    assert "No matching records in this snapshot" in result.answer
    assert result.tool_calls == 1
    assert engine.guardrail.calls == [query]
