"""Request-scoped selection, never model-selected fallback or shared mutation."""
import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.main import app
from app.api.graphrag_routes import get_graph_supervisor
from app.models.schemas import AssistantRequest, QueryRoute
from app.models.graphrag import GraphQueryRequest
from app.models.graph_agents import AgentRole
from app.models.graph_supervisor import GraphTool, SupervisorResult, SupervisorLimits
from app.services.graph_supervisor import GraphRAGSupervisor
from app.services.graph_agents import HierarchicalGraphSupervisor, SpecialistAgent
from app.services.assistant_service import AssistantGraphService
from tests.test_graph_supervisor import FakeGuard, FakeTool, Plans, retrieve, task, ROOT
from tests.test_assistant_service import FakeClassifier
from tests.test_api import api_client, FakeAssistantService
from app.api.routes import get_assistant_service


def engine(tools):
    return GraphRAGSupervisor(chain=Plans(), guardrail=FakeGuard(), tools=tools,
                             limits=SupervisorLimits(max_tool_calls=2))


def test_defaults_are_local():
    assert AssistantRequest(query="Review themes", user_id="test").graphrag_search_mode == "local"
    assert GraphQueryRequest(query="Review themes").graphrag_search_mode == "local"


@pytest.mark.parametrize("mode", ["auto", "drift", "GLOBAL", "", None])
def test_invalid_request_mode_fails_closed(mode):
    with pytest.raises(ValidationError):
        AssistantRequest(query="Review themes", user_id="test", graphrag_search_mode=mode)
    with pytest.raises(ValidationError):
        GraphQueryRequest(query="Review themes", graphrag_search_mode=mode)


def test_selection_does_not_mutate_shared_tools_or_neo4j():
    original = engine({tool: FakeTool() for tool in GraphTool})
    local, global_ = original.with_search_mode(), original.with_search_mode("global")
    cypher = {GraphTool.NEO4J, GraphTool.PREDEFINED_CYPHER, GraphTool.TEXT_TO_CYPHER}
    assert set(local.tools) == cypher | {GraphTool.MS_LOCAL}
    assert set(global_.tools) == cypher | {GraphTool.MS_GLOBAL}
    assert set(original.tools) == set(GraphTool)
    assert local.tools[GraphTool.NEO4J] is original.tools[GraphTool.NEO4J]
    with pytest.raises(ValueError):
        original.with_search_mode("auto")


def test_missing_local_tool_never_falls_back_to_global():
    original = engine({GraphTool.MS_GLOBAL: FakeTool()})
    selected = original.with_search_mode()
    assert not selected.tools
    assert asyncio.run(selected.run_approved(ROOT, scope_approved=True)).status == "unavailable"


def test_model_cannot_execute_global_when_local_is_selected():
    tool = FakeTool()
    original = engine({GraphTool.MS_LOCAL: tool, GraphTool.MS_GLOBAL: tool})
    original._chain = Plans(retrieve(task(ROOT, "ms_global_search")))
    result = asyncio.run(original.with_search_mode().run_approved(ROOT, scope_approved=True))
    assert result.reason_code == "tool_not_connected"
    assert result.tool_calls == 0


def test_hierarchical_selection_only_changes_review_tools():
    agents = {role: SpecialistAgent(role=role, engine=engine(
        {GraphTool.MS_LOCAL: FakeTool(), GraphTool.MS_GLOBAL: FakeTool()} if role == AgentRole.REVIEWS
        else {GraphTool.NEO4J: FakeTool()})) for role in AgentRole}
    original = HierarchicalGraphSupervisor(chain=Plans(), guardrail=FakeGuard(), agents=agents)
    selected = original.with_search_mode("global")
    assert set(selected.agents[AgentRole.REVIEWS].engine.tools) == {GraphTool.MS_GLOBAL}
    assert set(original.agents[AgentRole.REVIEWS].engine.tools) == {GraphTool.MS_LOCAL, GraphTool.MS_GLOBAL}
    for role in (AgentRole.CATALOG, AgentRole.SALES):
        assert set(selected.agents[role].engine.tools) == {GraphTool.NEO4J}


class ModeSupervisor:
    def __init__(self, calls, mode=None):
        self.calls, self.mode = calls, mode

    def with_search_mode(self, mode="local"):
        return ModeSupervisor(self.calls, mode)

    async def run_approved(self, query, *, scope_approved):
        assert scope_approved
        return await self.run(query)

    async def run(self, query):
        await asyncio.sleep(0)
        self.calls.append(self.mode)
        return SupervisorResult(status="complete", reason_code="test", answer=self.mode)


def test_concurrent_assistant_requests_keep_their_own_mode():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    calls = []
    service = AssistantGraphService.from_model(
        classifier=FakeClassifier(QueryRoute.GRAPH_RAG_SEARCH),
        model_client=FakeListChatModel(responses=["unused"]), graph_guardrail=FakeGuard(),
        graph_supervisor=ModeSupervisor(calls), provider="test", model="test")
    async def run():
        async def collect(mode):
            return [e async for e in service.stream(ROOT, graphrag_search_mode=mode)]
        return await asyncio.gather(collect("local"), collect("global"))
    outputs = asyncio.run(run())
    assert sorted(calls) == ["global", "local"]
    for mode, events in zip(["local", "global"], outputs):
        assert "".join(p["content"] for n, p in events if n == "delta") == mode
        assert next(p for n, p in events if n == "route")["graphrag_search_mode"] == mode


@pytest.mark.parametrize("mode", [None, "global", "invalid"])
def test_graph_query_api_selection(mode):
    calls = []
    app.dependency_overrides[get_graph_supervisor] = lambda: ModeSupervisor(calls)
    try:
        with TestClient(app) as client:
            data = {"query": ROOT}
            if mode is not None:
                data["graphrag_search_mode"] = mode
            response = client.post("/api/graphrag/query", json=data)
        assert response.status_code == (422 if mode == "invalid" else 200)
        assert calls == ([] if mode == "invalid" else [mode or "local"])
    finally:
        app.dependency_overrides.pop(get_graph_supervisor, None)


@pytest.mark.parametrize("multipart", [False, True])
def test_assistant_endpoint_forwards_explicit_mode(api_client, multipart):
    calls = []
    class Service(FakeAssistantService):
        async def stream(self, *args, graphrag_search_mode="local", **kwargs):
            calls.append(graphrag_search_mode)
            async for item in super().stream(*args, **kwargs):
                yield item
    app.dependency_overrides[get_assistant_service] = Service
    client, _ = api_client
    data = {"query": "Review themes", "user_id": "mode-test", "graphrag_search_mode": "global"}
    options = {"files": {key: (None, value) for key, value in data.items()}} if multipart else {"json": data}
    response = client.post("/api/assistant", **options)
    assert response.status_code == 200
    assert calls == ["global"]
    assert '"graphrag_search_mode": "global"' in response.text


def test_frontend_defaults_to_local_and_sends_mode():
    root = Path(__file__).resolve().parents[1] / "app/static"
    assert '<option value="local" selected>' in (root / "index.html").read_text()
    script = (root / "app.js").read_text()
    assert 'formData.append("graphrag_search_mode", searchMode)' in script
    assert "graphrag_search_mode: searchMode" in script


@pytest.mark.parametrize("text,name,expected", [
    ("Poor app experiences", "APP EXPERIENCE", False),
    ("Poor app experience.", "APP EXPERIENCE", True),
    ("Shipping status", "Ring", False),
    ("The RING device", "Ring", True),
    ("ＡＰＰ  experience", "App experience", True),
    ("连接问题", "连接", True),
])
def test_worker_entity_matching_agrees_with_guardrail(text, name, expected):
    from graphrag_runtime.evidence import mentioned
    from app.services.graphrag_guardrail import GraphRAGGuardrail
    assert mentioned(text, name) is expected
    assert mentioned(text, name) == GraphRAGGuardrail._mentioned(text, name)


def test_global_report_evidence_keeps_text_but_excludes_substring_only_entities():
    from graphrag_runtime.evidence import candidates
    tables = {"entities": [{"title": "APP EXPERIENCE", "type": "SupportTopic"}],
              "text_units": [], "community_reports": [
                  {"community": 0, "level": 0, "id": "report-0", "full_content": "Some users described poor app experiences."}]}
    result = candidates({"reports": [{"id": 0}]}, tables, "[Data: Reports (0)]")
    assert len(result) == 1
    assert result[0]["entities"] == []
    assert "poor app experiences" in result[0]["text"]
