"""Deterministic hierarchical-agent tests with explicitly fake retrieval/models."""
import asyncio

import pytest
from pydantic import ValidationError

from app.models.graph_agents import AgentRole, DelegationPlan
from app.models.graph_supervisor import GraphTool, SupervisorLimits
from app.services.graph_agents import HierarchicalGraphSupervisor, SpecialistAgent
from app.services.graph_supervisor import GraphRAGSupervisor
from tests.test_graph_supervisor import FakeGuard, FakeTool, Plans, ROOT, task, retrieve, finish


CATALOG = "Who supplies Acme Sensor?"
REVIEWS = "Summarize support themes for Acme Sensor."


def assignment(role="catalog_agent", question=CATALOG, parents=None):
    return {"agent": role, "question": question, "parent_evidence_ids": parents or []}


def delegate(*tasks):
    return {"action": "delegate", "tasks": list(tasks) or [assignment()], "evidence_ids": []}


def specialist(role, plans, tool):
    key = GraphTool.MS_GLOBAL if role == AgentRole.REVIEWS else GraphTool.NEO4J
    engine = GraphRAGSupervisor(chain=plans, guardrail=FakeGuard(), tools={key: tool},
                                limits=SupervisorLimits(max_tool_calls=2, max_rounds=2))
    return SpecialistAgent(role=role, engine=engine)


def supervisor(plans, *, shared=None, review_error=False, limits=None):
    catalog_tool = shared or FakeTool()
    review_tool = shared or FakeTool(error=review_error)
    catalog = specialist(AgentRole.CATALOG, Plans(retrieve(task()), finish()), catalog_tool)
    reviews = specialist(AgentRole.REVIEWS, Plans(retrieve(task(REVIEWS, "ms_global_search")), finish()), review_tool)
    service = HierarchicalGraphSupervisor(chain=plans, guardrail=FakeGuard(),
        agents={AgentRole.CATALOG: catalog, AgentRole.REVIEWS: reviews}, limits=limits)
    return service, catalog, reviews


def test_supervisor_delegates_two_independent_goals_and_agents_choose_tools():
    shared = FakeTool(delay=0.01)
    plans = Plans(delegate(assignment(), assignment("reviews_agent", REVIEWS)), finish("E1", "E2"))
    service, catalog, reviews = supervisor(plans, shared=shared)
    result = asyncio.run(service.run(ROOT))
    assert result.status == "complete" and shared.peak == 2
    assert result.tool_calls == 2 and result.rounds == 1
    assert [run["agent"] for run in result.agent_runs] == ["catalog_agent", "reviews_agent"]
    assert [step["tool"] for step in result.trace] == ["neo4j_relationships", "ms_global_search"]
    assert [item.evidence_id for item in result.evidence] == ["E1", "E2"]
    assert [item.agent_run_id for item in result.evidence] == ["A1.1", "A1.2"]
    assert result.trace[0]["task_id"] == "A1.1/R1T1"
    assert catalog.engine._chain.inputs[0]["original_question"] == CATALOG
    assert reviews.engine._chain.inputs[0]["original_question"] == REVIEWS
    assert catalog.engine._chain.inputs[0]["evidence"] == reviews.engine._chain.inputs[0]["evidence"] == []
    assert len(plans.inputs[1]["agent_runs"]) == 2


def test_dependent_agent_runs_in_next_round_with_parent_lineage():
    followup = "Summarize support themes for Acme Supply products."
    service, _, reviews = supervisor(Plans(delegate(), delegate(assignment("reviews_agent", followup, ["E1"])), finish("E1", "E2")))
    reviews.engine._chain = Plans(retrieve(task(followup, "ms_global_search")), finish())
    result = asyncio.run(service.run(ROOT))
    assert result.status == "complete" and result.rounds == 2
    assert result.evidence[1].parent_evidence_ids == ["E1"]
    assert result.agent_runs[1]["parent_evidence_ids"] == ["E1"]


@pytest.mark.parametrize("bad", [
    assignment("reviews_agent", "Summarize themes for Invented Corp."),
    assignment("reviews_agent", REVIEWS, ["E999"]),
    assignment("sales_agent", "Revenue for 2025"),
])
def test_invalid_batch_launches_no_agents(bad):
    service, catalog, reviews = supervisor(Plans(delegate(assignment(), bad)))
    result = asyncio.run(service.run(ROOT))
    assert result.status in {"rejected", "unavailable"}
    assert not catalog.engine._chain.inputs and not reviews.engine._chain.inputs


def test_partial_agent_failure_preserves_success_but_not_full_success():
    service, _, _ = supervisor(Plans(delegate(assignment(), assignment("reviews_agent", REVIEWS))), review_error=True)
    result = asyncio.run(service.run(ROOT))
    assert result.status == "partial" and result.reason_code == "subagent_incomplete"
    assert len(result.evidence) == 1 and len(result.agent_runs) == 2
    assert result.agent_runs[1]["status"] == "unavailable"
    assert "private-provider-secret" not in result.model_dump_json()


def test_finish_cannot_silently_drop_one_agents_evidence():
    service, _, _ = supervisor(Plans(delegate(assignment(), assignment("reviews_agent", REVIEWS)), finish("E1")))
    result = asyncio.run(service.run(ROOT))
    assert result.status == "partial" and result.reason_code == "incomplete_agent_coverage"
    assert result.answer_evidence_ids == ["E1", "E2"]


def test_global_call_reservation_stops_oversized_batch():
    service, catalog, reviews = supervisor(Plans(delegate(assignment(), assignment("reviews_agent", REVIEWS))),
                                           limits=SupervisorLimits(max_tool_calls=2))
    result = asyncio.run(service.run(ROOT))
    assert result.reason_code == "delegation_budget_exhausted"
    assert not catalog.engine._chain.inputs and not reviews.engine._chain.inputs


def test_parent_timeout_cancels_running_specialists():
    tool = FakeTool(delay=10)
    service, _, _ = supervisor(Plans(delegate(assignment(), assignment("reviews_agent", REVIEWS))),
                              shared=tool, limits=SupervisorLimits(timeout_seconds=0.03))
    result = asyncio.run(service.run(ROOT))
    assert result.reason_code == "supervisor_timeout"
    assert tool.cancelled and tool.active == 0


def test_specialist_cannot_select_an_unregistered_tool():
    agent = specialist(AgentRole.CATALOG, Plans(retrieve(task(REVIEWS, "ms_global_search"))), FakeTool())
    result = asyncio.run(agent.run(CATALOG))
    assert result.reason_code == "tool_not_connected" and result.tool_calls == 0


def test_specialists_cannot_be_constructed_with_wider_permissions():
    engine = GraphRAGSupervisor(chain=Plans(), guardrail=FakeGuard(), tools={GraphTool.NEO4J: FakeTool()},
                                limits=SupervisorLimits(max_tool_calls=2))
    with pytest.raises(ValueError):
        SpecialistAgent(role=AgentRole.REVIEWS, engine=engine)


def test_supervisor_contract_has_agent_assignments_not_tool_calls():
    for invalid in [
        {"action": "delegate", "tasks": [task()], "evidence_ids": []},
        delegate({**assignment(), "tool": "neo4j_relationships"}),
        delegate({**assignment(), "agent": "shell_agent"}),
    ]:
        with pytest.raises(ValidationError):
            DelegationPlan.model_validate(invalid)


def test_root_denial_never_starts_specialists():
    service, catalog, _ = supervisor(Plans(delegate()))
    result = asyncio.run(service.run_approved(ROOT, scope_approved=False))
    assert result.status == "rejected" and not catalog.engine._chain.inputs


def test_repeated_assignment_does_not_loop():
    service, catalog, _ = supervisor(Plans(delegate(), delegate()))
    result = asyncio.run(service.run(ROOT))
    assert result.reason_code == "repeated_assignment"
    assert len(catalog.engine._chain.inputs) == 2  # One retrieve and one finish.


def test_factory_construction_provides_separate_specialist_graphs():
    from langchain_core.runnables import RunnableLambda
    class Model:
        def with_structured_output(self, schema):
            return RunnableLambda(lambda _: None)
    service = HierarchicalGraphSupervisor.from_model(Model(), guardrail=FakeGuard(), task_guardrail=FakeGuard(),
        tools={GraphTool.NEO4J: FakeTool(), GraphTool.MS_LOCAL: FakeTool(), GraphTool.MS_GLOBAL: FakeTool()})
    assert set(service.agents) == set(AgentRole)
    assert len({id(agent.engine._graph) for agent in service.agents.values()}) == 3
    assert set(service.agents[AgentRole.REVIEWS].engine.tools) == {GraphTool.MS_LOCAL, GraphTool.MS_GLOBAL}


def test_assistant_stream_exposes_subagents_then_maps_not_private_plans():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from app.services.assistant_service import AssistantGraphService
    from app.services.product_catalog import ProductCatalog
    from app.models.schemas import QueryRoute
    from tests.test_assistant_service import FakeClassifier, PROJECT_ROOT
    from tests.test_graph_answer import generator
    service, _, _ = supervisor(Plans(delegate(assignment(), assignment("reviews_agent", REVIEWS)), finish("E1", "E2")))
    service.answer_generator = generator()[0]
    assistant = AssistantGraphService.from_model(classifier=FakeClassifier(QueryRoute.GRAPH_RAG_SEARCH),
        model_client=FakeListChatModel(responses=["unused"]), product_catalog=ProductCatalog(PROJECT_ROOT / "Business_data"),
        graph_guardrail=service.guardrail, graph_supervisor=service, provider="test", model="test")
    async def run():
        return [item async for item in assistant.stream(ROOT)]
    events = asyncio.run(run())
    started = [p for n, p in events if n == "agent" and p["stage"] == "started"]
    assert {p["agent"] for p in started} == {"catalog_agent", "reviews_agent"}
    assert sum(n == "supervisor" for n, _ in events) == 1
    assert next(p for n, p in events if n == "answer_generation")["total"] == 2
    assert "[E1]" in "".join(p["content"] for n, p in events if n == "delta")
