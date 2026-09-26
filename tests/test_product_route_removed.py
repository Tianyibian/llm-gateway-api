"""Regression checks for removing the standalone CSV catalog branch."""
import asyncio
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.models.schemas import QueryClassification, QueryRoute
from app.services.assistant_service import AssistantGraphService
from app.services.clarification_service import ClarificationService
from app.services.query_classifier import QueryClassifier


class RemovedRouteChain:
    async def ainvoke(self, values):
        return {"route": "product_search", "reason": "Retired route", "confidence": 1.0}


def test_router_contract_has_only_current_text_routes():
    assert {route.value for route in QueryRoute} == {
        "general_search", "additional_search", "policy_search", "file_query",
        "analytics_search", "graph_rag_search",
    }


def test_classifier_rejects_retired_route_from_model():
    classifier = QueryClassifier(chain=RemovedRouteChain(), provider="fake", model="fake")
    with pytest.raises(ValidationError):
        asyncio.run(classifier.classify("Show product stock"))


def test_clarification_cannot_dispatch_to_retired_route():
    class Chain:
        async def ainvoke(self, values):
            return {"action": "ready", "missing": [], "resolved_query": "Show stock",
                    "next_route": "product_search"}
    result = asyncio.run(ClarificationService(chain=Chain()).assess("Show stock"))
    assert result["action"] == "unavailable"
    assert "next_route" not in result


def test_graph_builds_and_runs_without_catalog_dependency():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel

    class Classifier:
        async def classify(self, query, *, history=None):
            return QueryClassification(route=QueryRoute.GENERAL_SEARCH, reason="Greeting", confidence=1)

    service = AssistantGraphService.from_model(
        classifier=Classifier(), model_client=FakeListChatModel(responses=["Hello."]),
        provider="fake", model="fake",
    )
    nodes = service._graph.get_graph().nodes
    assert "search_products" not in nodes
    assert "product_answer" not in nodes
    assert {"graphrag", "clarify_request", "knowledge_answer", "vision_answer"} <= nodes.keys()

    async def run():
        return [event async for event in service.stream("Hello")]
    events = asyncio.run(run())
    assert events[0][1]["route"] == "general_search"
    assert not any(name == "sources" for name, _ in events)
    assert "".join(data["content"] for name, data in events if name == "delta") == "Hello."


def test_frontend_does_not_advertise_retired_catalog():
    static = Path(__file__).resolve().parents[1] / "app/static"
    for name in ("index.html", "app.js", "graph-inspector.js"):
        text = (static / name).read_text()
        assert "product_search" not in text
        assert "Product catalog" not in text
    assert "showProducts" not in (static / "app.js").read_text()


def test_graph_guardrail_prompt_distinguishes_current_and_historical_prices():
    from app.services.graphrag_guardrail import GraphRAGGuardrail
    prompt = GraphRAGGuardrail.SYSTEM_PROMPT
    assert "Historical order-line prices are not current catalog prices" in prompt
    assert "Mark these direct lookup requests out_of_scope" in prompt
