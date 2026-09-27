"""One scope-model assessment per business branch; execution checks remain separate."""
import asyncio
import ast
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.api.graphrag_routes import get_graph_supervisor
from app.main import app
from app.models.schemas import QueryRoute
from app.services.assistant_service import AssistantGraphService
from tests.test_assistant_service import FakeClassifier, FakeVisionService
from tests.test_graph_supervisor import FakeGuard, Plans, ROOT, finish, retrieve, service, task


@pytest.mark.parametrize("entry", ["service", "api", "assistant"])
def test_multiround_graph_uses_exactly_one_scope_call(entry):
    guard = FakeGuard()
    engine, tool = service(Plans(retrieve(), retrieve(task("Count Acme Supply products", parents=["E1"])), finish("E2")), guard=guard)
    if entry == "api":
        app.dependency_overrides[get_graph_supervisor] = lambda: engine
        try:
            with TestClient(app) as client:
                result = client.post("/api/graphrag/query", json={"query": ROOT}).json()
            assert result["status"] == "complete"
        finally:
            app.dependency_overrides.pop(get_graph_supervisor, None)
    elif entry == "service":
        assert asyncio.run(engine.run(ROOT)).status == "complete"
    else:
        assistant = AssistantGraphService.from_model(
            classifier=FakeClassifier(QueryRoute.GRAPH_RAG_SEARCH),
            model_client=FakeListChatModel(responses=["unused"]),
            graph_guardrail=guard, graph_supervisor=engine, provider="fake", model="fake",
        )
        async def run():
            return [event async for event in assistant.stream(ROOT)]
        events = asyncio.run(run())
        assert sum(name == "guardrail" for name, _ in events) == 1
        assert next(payload for name, payload in events if name == "supervisor")["status"] == "complete"
    assert guard.calls == [ROOT]
    assert len(tool.calls) == 2


@pytest.mark.parametrize("destination", ["policy_search", "graph_rag_search"])
def test_clarification_handoff_assesses_each_distinct_branch_once(destination):
    class Clarification:
        calls = 0
        async def assess(self, query, **kwargs):
            self.calls += 1
            return {"action": "ready", "resolved_query": ROOT, "next_route": destination}
    class Policy:
        calls = 0
        async def assess(self, query, **kwargs):
            self.calls += 1
            return {"allowed": True, "action": "allow"}
    class Retriever:
        calls = 0
        async def retrieve(self, query):
            self.calls += 1
            return []
    guard = FakeGuard()
    engine, _ = service(Plans(retrieve(), finish()), guard=guard)
    clarification, policy, retriever = Clarification(), Policy(), Retriever()
    assistant = AssistantGraphService.from_model(
        classifier=FakeClassifier(QueryRoute.ADDITIONAL_SEARCH),
        model_client=FakeListChatModel(responses=["unused"]),
        graph_guardrail=guard, graph_supervisor=engine, policy_guardrail=policy,
        clarification_service=clarification, knowledge_retriever=retriever,
        provider="fake", model="fake",
    )
    async def run():
        return [event async for event in assistant.stream("The product I mentioned")]
    asyncio.run(run())
    assert clarification.calls == 1
    assert policy.calls == (1 if destination == "policy_search" else 0)
    assert guard.calls == ([ROOT] if destination == "graph_rag_search" else [])


@pytest.mark.parametrize("route", [QueryRoute.GENERAL_SEARCH, QueryRoute.FILE_QUERY, QueryRoute.ANALYTICS_SEARCH])
def test_other_text_branches_do_not_invoke_scope_gates(route):
    class NoScope:
        async def assess(self, *args, **kwargs):
            pytest.fail("Unexpected scope call")
        async def evaluate(self, *args, **kwargs):
            pytest.fail("Unexpected scope call")
    assistant = AssistantGraphService.from_model(
        classifier=FakeClassifier(route), model_client=FakeListChatModel(responses=["General answer"]),
        graph_guardrail=NoScope(), policy_guardrail=NoScope(), clarification_service=NoScope(),
        provider="fake", model="fake",
    )
    async def run():
        return [event async for event in assistant.stream("Hello")]
    events = asyncio.run(run())
    assert not {"guardrail", "policy_guardrail", "clarification"}.intersection(name for name, _ in events)


def test_vision_branch_does_not_invoke_scope_gates():
    assistant = AssistantGraphService.from_model(
        classifier=FakeClassifier(QueryRoute.GENERAL_SEARCH), model_client=FakeListChatModel(responses=[]),
        vision_service=FakeVisionService(), provider="fake", model="fake",
    )
    async def run():
        return [event async for event in assistant.stream("What number is visible?", image_bytes=b"image-bytes", image_mime_type="image/png")]
    events = asyncio.run(run())
    assert not {"guardrail", "policy_guardrail", "clarification"}.intersection(name for name, _ in events)


def test_nested_planners_cannot_reintroduce_scope_calls():
    root = Path(__file__).resolve().parents[1]
    for file, expected in [("graph_supervisor.py", ["self.guardrail.evaluate"]), ("graph_agents.py", [])]:
        tree = ast.parse((root / "app/services" / file).read_text())
        calls = [ast.unparse(node.func) for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr in {"evaluate", "assess"}]
        assert calls == expected
