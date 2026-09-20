"""Fail-closed query checks using fake model reviews and Neo4j summaries."""
import asyncio
from types import SimpleNamespace

import pytest

from app.services.cypher_checks import validate_explain
from tests.test_neo4j_queries import service


def summary(**updates):
    return SimpleNamespace(**(dict(query_type="r", plan={"args": {"EstimatedRows": 10}, "children": []}, notifications=[]) | updates))


def test_read_only_plan_passes():
    validate_explain(summary())


@pytest.mark.parametrize("kind", ["w", "rw", "s", None])
def test_non_read_plans_fail(kind):
    with pytest.raises(ValueError):
        validate_explain(summary(query_type=kind))


@pytest.mark.parametrize("count", [100001, float("inf"), float("nan"), -1])
def test_nested_plan_budget_failures(count):
    with pytest.raises(ValueError):
        validate_explain(summary(plan={"children": [{"args": {"EstimatedRows": count}}]}))


@pytest.mark.parametrize("code", ["UnknownLabelWarning", "UnknownRelationshipTypeWarning", "UnknownPropertyKeyWarning"])
def test_unknown_schema_notifications_fail(code):
    with pytest.raises(ValueError):
        validate_explain(summary(notifications=[{"code": f"Neo.ClientNotification.Statement.{code}"}]))


def test_bad_semantic_review_never_executes_data_query():
    svc, _, _, executor = service()
    svc.reviewer.ainvoke.return_value = {"matches_question": True, "preserves_all_constraints": False, "reason_code": "missing_filter"}
    answer = asyncio.run(svc.query("Who supplies Acme Sensor?"))
    assert answer.status == "rejected" and answer.reason_code == "cypher_semantic_check_failed"
    executor.explain.assert_not_awaited()
    assert executor.run.await_count == 1  # Snapshot readiness only.


def test_explain_failure_prevents_data_query():
    svc, _, _, executor = service()
    executor.explain.side_effect = ValueError("unsafe plan")
    answer = asyncio.run(svc.query("Who supplies Acme Sensor?"))
    assert answer.status == "rejected" and executor.run.await_count == 1


def test_passed_checks_are_attached_to_execution():
    svc, _, _, executor = service()
    answer = asyncio.run(svc.query("Who supplies Acme Sensor?"))
    assert answer.execution.checks == ["schema_and_parameters", "question_alignment", "neo4j_explain_read_only", "plan_budget"]
    executor.explain.assert_awaited_once()


@pytest.mark.parametrize("review", [None, {"matches_question": True}, {"matches_question": True, "preserves_all_constraints": True, "reason_code": "wrong_metric"}])
def test_invalid_reviews_fail_closed(review):
    svc, _, _, executor = service()
    svc.reviewer.ainvoke.return_value = review
    assert asyncio.run(svc.query("Who supplies Acme Sensor?")).status == "rejected"
    assert executor.run.await_count == 1
