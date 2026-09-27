"""Direct task orchestration contracts; these use explicit fake providers."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from app.core.config import Settings
from app.models.graph_supervisor import GraphTool, SupervisorLimits
from app.models.graph_answer import GeneratedGraphAnswer
from app.services.factory import LLMServiceFactory
from app.services.graph_supervisor import GraphRAGSupervisor
from app.services.neo4j_service import CypherRetrievalTool
from tests.test_graph_supervisor import FakeGuard, FakeTool, Plans, retrieve, task, finish, ROOT
from tests.test_neo4j_queries import service, selection


def supervisor(plans, tool=None, **kwargs):
    tool = tool or FakeTool()
    return GraphRAGSupervisor(chain=Plans(*plans), guardrail=FakeGuard(),
        tools={GraphTool.PREDEFINED_CYPHER: tool, GraphTool.TEXT_TO_CYPHER: tool,
               GraphTool.MS_GLOBAL: tool}, **kwargs)


def test_first_round_parallel_tasks_dispatch_directly_and_cover_all_results():
    tool = FakeTool(delay=0.01)
    svc = supervisor([retrieve(
        task("Who supplies Acme Sensor?", "predefined_cypher"),
        task("Count products", "text_to_cypher"),
        task("Summarize Acme Sensor themes", "ms_global_search")), finish("E1", "E2", "E3")], tool)
    result = asyncio.run(svc.run(ROOT))
    assert result.status == "complete" and result.tool_calls == 3 and result.rounds == 1
    assert tool.peak == 3
    assert result.agent_runs == []
    assert [row["tool"] for row in result.trace] == ["predefined_cypher", "text_to_cypher", "ms_global_search"]


def test_followup_uses_real_prior_evidence_not_a_guessed_entity():
    svc = supervisor([retrieve(task("Who supplies Acme Sensor?", "predefined_cypher")),
        retrieve(task("Count Acme Supply products", "text_to_cypher", ["E1"])), finish("E1", "E2")])
    result = asyncio.run(svc.run(ROOT))
    assert result.status == "complete" and result.rounds == 2
    assert svc._chain.inputs[1]["evidence"][0]["entities"][0]["text"] == "Acme Supply"
    assert result.trace[1]["parent_evidence_ids"] == ["E1"]


def test_first_batch_cannot_guess_a_dependency_and_executes_nothing():
    tool = FakeTool()
    svc = supervisor([retrieve(task("Who supplies Acme Sensor?", "predefined_cypher"),
                               task("Count Acme Supply products", "text_to_cypher"))], tool)
    assert asyncio.run(svc.run(ROOT)).reason_code == "entity_without_lineage"
    assert tool.calls == []


def test_finish_cannot_silently_drop_a_completed_subtask():
    svc = supervisor([retrieve(task("Count products", "text_to_cypher"),
                               task("Who supplies Acme Sensor?", "predefined_cypher")), finish("E1")])
    result = asyncio.run(svc.run(ROOT))
    assert result.status == "partial" and result.reason_code == "incomplete_task_coverage"
    assert result.answer_evidence_ids == ["E1", "E2"]


def test_finish_includes_declared_prerequisites_of_selected_followup():
    svc = supervisor([retrieve(task("Who supplies Acme Sensor?", "predefined_cypher")),
        retrieve(task("Count Acme Supply products", "text_to_cypher", ["E1"])), finish("E2")])
    result = asyncio.run(svc.run(ROOT))
    assert result.status == "complete"
    assert result.answer_evidence_ids == ["E1", "E2"]


def test_only_root_scope_guard_is_preserved_by_mode_selection():
    tool = FakeTool()
    guard = FakeGuard(blocked="Count")
    svc = supervisor([retrieve(task("Count products", "text_to_cypher")), finish()], tool)
    svc.guardrail = guard
    selected = svc.with_search_mode()
    assert selected.guardrail is guard
    assert not hasattr(selected, "task_guardrail")
    assert asyncio.run(selected.run(ROOT)).status == "complete"
    assert guard.calls == [ROOT]
    assert tool.calls == ["Count products"]


def test_root_rejection_never_calls_planner_or_tools():
    svc = supervisor([])
    svc.guardrail = FakeGuard(blocked="Acme")
    assert asyncio.run(svc.run(ROOT)).status == "rejected"
    assert svc._chain.inputs == []


def test_failure_preserves_successful_evidence_for_partial_reduce():
    generator = AsyncMock()
    generator.generate.return_value = GeneratedGraphAnswer(answer="Partial findings", status="complete",
        reason_code="map_reduce_validated", cited_evidence_ids=["E1"], limited=True)
    svc = supervisor([retrieve(task("Who supplies Acme Sensor?", "predefined_cypher"),
                               task("Count products", "text_to_cypher"))], answer_generator=generator)
    svc.tools[GraphTool.TEXT_TO_CYPHER] = FakeTool(error=True)
    result = asyncio.run(svc.run(ROOT))
    assert result.status == "partial" and result.answer == "Partial findings"
    assert generator.generate.await_args.args[1].status == "partial"
    assert len(result.evidence) == 1 and result.trace[1]["error"] == "retrieval_failed"


def test_factory_builds_direct_planner_not_specialists(monkeypatch):
    factory = LLMServiceFactory(Settings(_env_file=None, provider="ollama"))
    class Model:
        def with_structured_output(self, schema):
            from langchain_core.runnables import RunnableLambda
            return RunnableLambda(lambda _: {})
    monkeypatch.setattr(factory, "_build_langchain_model", lambda **_: Model())
    monkeypatch.setattr(factory, "_guardrail_from_model", lambda *args, **kwargs: FakeGuard())
    engine = factory.create_graph_supervisor(tools={GraphTool.PREDEFINED_CYPHER: FakeTool()})
    assert type(engine) is GraphRAGSupervisor
    assert not hasattr(engine, "agents")
    assert engine.answer_generator is not None


def test_template_tool_never_falls_back_to_dynamic_generation():
    svc, _, chain, executor = service(selected=selection(route="text_to_cypher", template=None, name=None))
    result = asyncio.run(svc.query("Count products by supplier", strategy="template"))
    assert result.status == "unsupported"
    chain.ainvoke.assert_not_awaited()
    assert executor.run.await_count == 1


def test_dynamic_tool_bypasses_template_selector_and_keeps_all_checks():
    svc, selector, chain, executor = service()
    result = asyncio.run(svc.query("Count products by supplier", strategy="text_to_cypher"))
    assert result.status == "complete" and result.execution.query_mode == "text_to_cypher"
    selector.ainvoke.assert_not_awaited()
    chain.ainvoke.assert_awaited_once()
    svc.reviewer.ainvoke.assert_awaited_once()
    executor.explain.assert_awaited_once()


@pytest.mark.parametrize("strategy", ["template", "text_to_cypher"])
def test_both_explicit_tools_stop_before_execution_when_review_rejects(strategy):
    svc, _, _, executor = service()
    svc.reviewer.ainvoke.return_value = {"matches_question": False, "preserves_all_constraints": False,
                                       "reason_code": "wrong_metric"}
    query = "Who supplies Acme Sensor?" if strategy == "template" else "Count products by supplier"
    result = asyncio.run(svc.query(query, strategy=strategy))
    assert result.reason_code == "cypher_semantic_check_failed"
    executor.explain.assert_not_awaited()
    assert executor.run.await_count == 1


@pytest.mark.parametrize("strategy", ["template", "text_to_cypher"])
def test_tool_adapter_pins_strategy(strategy):
    service = AsyncMock()
    service.retrieve.return_value = []
    tool = CypherRetrievalTool(service, strategy=strategy)
    assert asyncio.run(tool.retrieve("Question", limit=3)) == []
    service.retrieve.assert_awaited_once_with("Question", limit=3, strategy=strategy)


def test_validated_task_progress_contains_real_tool_and_task_ids():
    svc = supervisor([retrieve(task("Count products", "text_to_cypher")), finish()])
    async def collect():
        return [event async for event in svc._graph.astream(
            {"query": ROOT, "evidence": [], "trace": [], "rounds": 0, "tool_calls": 0, "seen_tasks": []},
            stream_mode="custom")]
    events = asyncio.run(collect())
    assert [e["payload"]["stage"] for e in events] == ["started", "completed"]
    assert all(e["event"] == "graph_task" and e["payload"]["tool"] == "text_to_cypher" for e in events)
    assert events[0]["payload"]["task_id"] == "R1T1"


def test_empty_subtask_is_not_hidden_by_another_success():
    svc = supervisor([retrieve(task("Count products", "text_to_cypher"),
                               task("Who supplies Acme Sensor?", "predefined_cypher")), finish()])
    empty = AsyncMock()
    empty.retrieve.return_value = []
    svc.tools[GraphTool.TEXT_TO_CYPHER] = empty
    result = asyncio.run(svc.run(ROOT))
    assert result.status == "partial" and result.reason_code == "task_without_evidence"


def test_assistant_stream_reports_guardrail_before_task_start_once():
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from app.models.schemas import QueryRoute
    from app.services.assistant_service import AssistantGraphService
    from tests.test_assistant_service import FakeClassifier
    engine = supervisor([retrieve(task("Count products", "text_to_cypher")), finish()])
    svc = AssistantGraphService.from_model(classifier=FakeClassifier(QueryRoute.GRAPH_RAG_SEARCH),
        model_client=FakeListChatModel(responses=["unused"]), graph_guardrail=FakeGuard(),
        graph_supervisor=engine, provider="test", model="test")
    async def collect():
        return [name async for name, _ in svc.stream(ROOT)]
    names = asyncio.run(collect())
    assert names.count("guardrail") == 1
    assert names.index("guardrail") < names.index("graph_task") < names.index("supervisor") < names.index("delta")
