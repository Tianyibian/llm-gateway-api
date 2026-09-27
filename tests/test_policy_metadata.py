"""Metadata narrowing, shared hybrid corpus, API transport and scope tests."""
import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError
from sqlalchemy.dialects import postgresql
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.models.policy_filters import PolicyMetadataFilters
from app.models.schemas import AssistantRequest, QueryClassification, QueryRoute
from app.services.assistant_service import AssistantGraphService
from app.services.policy_guardrail import PolicyGuardrail
from tests.test_policy_ensemble import Chain, make_retriever, row
from tests.test_api import api_client
from app.main import app
from app.api.routes import get_assistant_service


@pytest.mark.parametrize("value", [
    {"visibility": "internal"}, {"where": "1=1"}, {"source_types": ["exe"]},
    {"categories": []}, {"categories": [" "]}, {"categories": [123]},
    {"source_paths": ["x"] * 21}, {"categories": "Policy"},
])
def test_invalid_filters_rejected(value):
    with pytest.raises(ValidationError):
        AssistantRequest(query="policy", user_id="test", policy_filters=value)


def test_all_filters_narrow_hybrid_candidates_without_interpolated_sql():
    matched = row("a", "refund policy", .1)
    matched[3].source_type = "pdf"
    matched[3].source_path = "x' OR 1=1 --"
    retriever, statements = make_retriever([matched, row("b", "refund policy", .2)])
    filters = PolicyMetadataFilters(categories=["Policy", "Returns"], source_types=["pdf"],
                                    source_paths=["x' OR 1=1 --"])
    result = asyncio.run(retriever.retrieve("refund", filters=filters))
    assert result[0].vector_rank == result[0].bm25_rank == 1
    assert len(result) == 1
    assert len(statements) == 1  # Both rankers consume the same public corpus.
    compiled = statements[0].compile(dialect=postgresql.dialect())
    sql = str(compiled)
    for name in ("visibility", "embedding_provider", "embedding_model", "category", "source_type", "source_path"):
        assert name in sql
    assert "x' OR 1=1 --" not in sql
    assert ["x' OR 1=1 --"] not in compiled.params.values()
    assert "source_type IN" not in sql and "category IN" not in sql
    assert "public" in compiled.params.values()


def test_empty_filtered_result_never_retries_without_filters():
    retriever, statements = make_retriever([])
    assert asyncio.run(retriever.retrieve("refund", filters=PolicyMetadataFilters(categories=["Missing"]))) == []
    assert len(statements) == 1


@pytest.mark.parametrize("via_clarification", [False, True])
@pytest.mark.parametrize("allowed", [False, True])
def test_filters_survive_clarification_but_cannot_bypass_policy_guard(via_clarification, allowed):
    class Classifier:
        async def classify(self, query, **kwargs):
            return QueryClassification(route=QueryRoute.ADDITIONAL_SEARCH if via_clarification else QueryRoute.POLICY_SEARCH,
                                       reason="test", confidence=1)
    class Clarifier:
        async def assess(self, *args, **kwargs):
            return {"action": "ready", "resolved_query": "refund policy", "next_route": "policy_search"}
    retriever = AsyncMock()
    retriever.retrieve.return_value = []
    filters = PolicyMetadataFilters(categories=["Policy"], source_types=["pdf"])
    service = AssistantGraphService.from_model(classifier=Classifier(), clarification_service=Clarifier(),
        knowledge_retriever=retriever, model_client=FakeListChatModel(responses=["must not generate"]),
        policy_guardrail=PolicyGuardrail(chain=Chain({"scope": "in_scope" if allowed else "out_of_scope",
                                                    "unsafe": False, "confidence": .99})), provider="fake", model="fake")
    async def run():
        return [event async for event in service.stream("policy?", policy_filters=filters)]
    events = asyncio.run(run())
    assert retriever.retrieve.await_count == int(allowed)
    if allowed:
        assert retriever.retrieve.call_args.kwargs["filters"] == filters
        sources = next(payload for name, payload in events if name == "sources")
        assert sources["metadata_filters"] == filters.model_dump(exclude_none=True)
        assert any("No policy candidates" in str(payload) for name, payload in events if name == "delta")
    assert not any("must not generate" in str(payload) for _, payload in events)


@pytest.mark.parametrize("multipart", [False, True])
def test_api_transports_filters(api_client, multipart):
    import json
    client, _ = api_client
    calls = []
    class Service:
        provider = model = "fake"
        async def stream(self, query, **kwargs):
            calls.append(kwargs["policy_filters"])
            yield "delta", {"content": "Filtered answer"}
    app.dependency_overrides[get_assistant_service] = Service
    payload = {"query": "return policy", "user_id": "metadata-test", "policy_filters": {"source_types": ["pdf"]}}
    if multipart:
        response = client.post("/api/assistant", files={key: (None, json.dumps(value) if key == "policy_filters" else value)
                                                       for key, value in payload.items()})
    else:
        response = client.post("/api/assistant", json=payload)
    assert response.status_code == 200
    assert calls[0].source_types == ["pdf"]
    assert "Filtered answer" in response.text


def test_api_forbids_permission_override(api_client):
    client, _ = api_client
    class Service:
        pass
    app.dependency_overrides[get_assistant_service] = Service
    response = client.post("/api/assistant", json={"user_id": "test", "query": "policy", "policy_filters": {"visibility": "internal"}})
    assert response.status_code == 422
