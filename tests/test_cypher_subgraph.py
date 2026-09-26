"""Subgraph order and fail-closed execution tests; no real writes are attempted."""
import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from app.services.cypher_checks import validate_explain
from app.services.cypher_compiler import CompiledCypher, SCHEMA_VERSION
from app.services.cypher_templates import compile_template
from app.services.neo4j_service import Neo4jExecutor
from tests.test_neo4j_queries import service, selection
from tests.test_cypher_checks import summary


@pytest.mark.parametrize("strategy,expected", [
    ("text_to_cypher", ["prepare_query", "generate_cypher", "validate_cypher", "execute_cypher"]),
    ("template", ["prepare_query", "compile_template", "validate_cypher", "execute_cypher"]),
])
def test_actual_langgraph_node_order(strategy, expected):
    svc, _, _, _ = service()
    async def collect():
        return [next(iter(update)) async for update in svc._graph.astream(
            {"question": "Who supplies Acme Sensor?", "strategy": strategy}, stream_mode="updates")]
    assert asyncio.run(collect()) == expected


@pytest.mark.parametrize("strategy", ["template", "text_to_cypher"])
def test_failed_validation_has_no_edge_to_execution(strategy):
    svc, _, _, executor = service()
    svc.reviewer.ainvoke.return_value = {"matches_question": False, "preserves_all_constraints": False,
                                       "reason_code": "wrong_metric"}
    async def collect():
        return [next(iter(update)) async for update in svc._graph.astream(
            {"question": "Who supplies Acme Sensor?", "strategy": strategy}, stream_mode="updates")]
    nodes = asyncio.run(collect())
    assert nodes[-1] == "validate_cypher" and "execute_cypher" not in nodes
    assert executor.run.await_count == 1  # Read-only snapshot check, not the data query.


@pytest.mark.parametrize("strategy", ["template", "text_to_cypher"])
@pytest.mark.parametrize("kind", ["w", "rw", "s", None])
def test_even_model_approved_queries_stop_on_non_read_explain(strategy, kind):
    svc, _, _, executor = service()
    async def explain(_):
        validate_explain(summary(query_type=kind))
    executor.explain.side_effect = explain
    result = asyncio.run(svc.query("Who supplies Acme Sensor?", strategy=strategy))
    assert result.status == "rejected"
    assert executor.run.await_count == 1


def template_state():
    selected = selection()
    compiled = compile_template(selected, question="Who supplies Acme Sensor?", dataset="test")
    return {"selection": selected, "question": "Who supplies Acme Sensor?", "compiled": compiled}


@pytest.mark.parametrize("change", ["write", "parameters"])
def test_validation_rejects_candidate_not_produced_by_compiler(change):
    svc, _, _, executor = service()
    state = template_state()
    state["compiled"] = (replace(state["compiled"], cypher="CREATE (n:Product) RETURN n") if change == "write"
                         else replace(state["compiled"], parameters={"name": "Someone else"}))
    with pytest.raises(ValueError):
        asyncio.run(svc._validate_cypher(state))
    svc.reviewer.ainvoke.assert_not_awaited()
    executor.run.assert_not_awaited()


@pytest.mark.parametrize("change", ["missing_approval", "query", "parameters"])
def test_execution_requires_unchanged_validated_candidate(change):
    svc, _, _, executor = service()
    state = template_state()
    if change != "missing_approval":
        state["approval"] = svc._fingerprint(state["compiled"])
        if change == "query":
            state["compiled"] = replace(state["compiled"], cypher="MATCH (n) DELETE n")
        else:
            state["compiled"].parameters["name"] = "Changed name"
    with pytest.raises(ValueError):
        asyncio.run(svc._execute_cypher(state))
    executor.run.assert_not_awaited()


@pytest.mark.parametrize("raw", [
    {"action": "CREATE", "cypher": "CREATE (n)"},
    {"route": "template", "template": "delete_all", "name": None, "year": None, "limit": 20},
])
def test_write_actions_or_unknown_template_cannot_enter_execution(raw):
    svc, selector, chain, executor = service()
    dynamic = "action" in raw
    (chain if dynamic else selector).ainvoke.return_value = raw
    result = asyncio.run(svc.query("Delete products", strategy="text_to_cypher" if dynamic else "template"))
    assert result.status == "rejected"
    assert executor.run.await_count == 1
    svc.reviewer.ainvoke.assert_not_awaited()


class Result:
    def __init__(self, rows=(), plan_summary=None):
        self.rows, self.plan_summary = rows, plan_summary
    def __aiter__(self):
        async def records():
            for row in self.rows:
                yield SimpleNamespace(data=lambda row=row: row)
        return records()
    async def consume(self):
        return self.plan_summary


def executor_with_session(kind):
    settings = SimpleNamespace(neo4j_enabled=True, neo4j_uri="bolt://unused", neo4j_user="test",
        neo4j_password=SecretStr("not-a-credential"), neo4j_database="neo4j", neo4j_dataset="test",
        neo4j_query_timeout_seconds=5, neo4j_max_estimated_rows=100_000)
    executor = Neo4jExecutor(settings)
    session = AsyncMock()
    session.run.side_effect = [Result([{"version": SCHEMA_VERSION, "nodes": 1, "edges": 0}]),
                               Result(plan_summary=summary(query_type=kind)), Result([{"count": 1}])]
    driver = MagicMock()
    driver.session.return_value.__aenter__.return_value = session
    factory = MagicMock()
    factory.return_value.__aenter__.return_value = driver
    executor._driver_factory = factory
    return executor, session


@pytest.mark.parametrize("kind", ["w", "rw", "s", None])
def test_direct_executor_cannot_bypass_explain_gate(kind):
    executor, session = executor_with_session(kind)
    compiled = CompiledCypher("CREATE (n:Product) RETURN n", {}, 1, "Product", "records", "none")
    with pytest.raises(ValueError):
        asyncio.run(executor.run(compiled))
    assert session.run.await_count == 2
    assert session.run.await_args_list[1].args[0].text.startswith("EXPLAIN ")
    assert all(call.args[0].text != compiled.cypher for call in session.run.await_args_list)


def test_direct_executor_checks_read_plan_before_running_data_query():
    executor, session = executor_with_session("r")
    compiled = template_state()["compiled"]
    rows, _ = asyncio.run(executor.run(compiled))
    assert rows == [{"count": 1}]
    assert [call.args[0].text for call in session.run.await_args_list][1:] == ["EXPLAIN " + compiled.cypher, compiled.cypher]


def test_explain_only_never_runs_candidate():
    executor, session = executor_with_session("r")
    asyncio.run(executor.explain(template_state()["compiled"]))
    assert session.run.await_count == 2


def test_concurrent_queries_do_not_share_candidate_or_approval_state():
    svc, selector, _, executor = service()
    async def choose(inputs):
        return selection(name=inputs["question"].removeprefix("Who supplies ").removesuffix("?"))
    selector.ainvoke.side_effect = choose
    executor.run.side_effect = None
    executor.run.return_value = ([], {})
    async def run():
        return await asyncio.gather(svc.query("Who supplies Acme Sensor?", strategy="template"),
                                    svc.query("Who supplies Beta Sensor?", strategy="template"))
    results = asyncio.run(run())
    assert all(r.status == "complete" for r in results)
    assert [r.execution.parameters["name"] for r in results] == ["Acme Sensor", "Beta Sensor"]
