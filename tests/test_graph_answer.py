"""Deterministic answer-contract tests; model calls are explicitly mocked."""
import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from app.models.graph_supervisor import GraphEvidence, SupervisorResult
from app.services.graph_answer import GraphAnswerGenerator


def evidence(index=1, text="Acme supplies Sensor. Revenue: 123.45."):
    return GraphEvidence(evidence_id=f"E{index}", source_id=f"test:{index}", text=text,
                         entities=[], tool="neo4j_relationships", task_id=f"T{index}", parent_evidence_ids=[])


def result(*items, status="complete"):
    items = list(items) or [evidence()]
    return SupervisorResult(status=status, reason_code="test", answer="RAW DEBUG EVIDENCE",
                            evidence=items, answer_evidence_ids=[item.evidence_id for item in items])


def generator(**kwargs):
    mapper, reducer = AsyncMock(), AsyncMock()
    async def extract(values):
        return {"relevant": True, "excerpt_ids": [part["excerpt_id"] for part in json.loads(values["context"])["excerpts"]]}
    mapper.ainvoke.side_effect = extract
    reducer.ainvoke.return_value = {"status": "answered", "paragraphs": [
        {"text": "Acme supplies Sensor, with revenue of 123.45.", "evidence_ids": ["E1"]}]}
    return GraphAnswerGenerator(mapper=mapper, reducer=reducer, **kwargs), mapper, reducer


def test_map_reduce_produces_answer_not_evidence_dump():
    service, mapper, reducer = generator()
    answer = asyncio.run(service.generate("Who supplies Sensor?", result()))
    assert answer.status == "complete" and answer.cited_evidence_ids == ["E1"]
    assert answer.answer.endswith("[E1]") and "RAW DEBUG" not in answer.answer
    assert json.loads(mapper.ainvoke.call_args.args[0]["context"])["question"] == "Who supplies Sensor?"
    assert json.loads(reducer.ainvoke.call_args.args[0]["context"])["evidence"][0]["quotes"] == ["Acme supplies Sensor.", "Revenue: 123.45."]


def test_maps_only_selected_evidence():
    service, mapper, _ = generator()
    retrieved = result(evidence(), evidence(2)).model_copy(update={"answer_evidence_ids": ["E1"]})
    answer = asyncio.run(service.generate("query", retrieved))
    assert answer.mapped_evidence == mapper.ainvoke.await_count == 1


@pytest.mark.parametrize("invalid", ["invented quote", "Acme supplies Sensor. Revenue: 999.99."])
def test_fabricated_map_excerpt_stops_before_reduce(invalid):
    service, mapper, reducer = generator()
    mapper.ainvoke.side_effect = None
    mapper.ainvoke.return_value = {"relevant": True, "quotes": [invalid]}
    answer = asyncio.run(service.generate("query", result()))
    assert answer.status == "unavailable" and answer.reason_code == "answer_validation_failed"
    reducer.ainvoke.assert_not_awaited()


@pytest.mark.parametrize("ids", [[999], [0], [True]])
def test_unknown_or_invalid_excerpt_ids_fail_before_reduce(ids):
    service, mapper, reducer = generator()
    mapper.ainvoke.side_effect = None
    mapper.ainvoke.return_value = {"relevant": True, "excerpt_ids": ids}
    answer = asyncio.run(service.generate("query", result()))
    assert answer.status == "unavailable"
    reducer.ainvoke.assert_not_awaited()


@pytest.mark.parametrize("paragraph", [
    {"text": "Acme", "evidence_ids": ["E99"]},
    {"text": "Acme [E1]", "evidence_ids": ["E1"]},
    {"text": "Acme", "evidence_ids": []},
    {"text": "   ", "evidence_ids": ["E1"]},
])
def test_rejects_invalid_citations(paragraph):
    service, _, reducer = generator()
    reducer.ainvoke.return_value = {"status": "answered", "paragraphs": [paragraph]}
    answer = asyncio.run(service.generate("query", result()))
    assert answer.status == "unavailable" and "[E99]" not in answer.answer


def test_no_relevant_sources_does_not_invoke_reduce():
    service, mapper, reducer = generator()
    mapper.ainvoke.side_effect = None
    mapper.ainvoke.return_value = {"relevant": False, "excerpt_ids": []}
    answer = asyncio.run(service.generate("query", result()))
    assert answer.status == "insufficient" and not answer.cited_evidence_ids
    reducer.ainvoke.assert_not_awaited()


@pytest.mark.parametrize("status,text", [("partial", "Acme"), ("complete", "bounded/truncated: True. Acme")])
def test_incomplete_retrieval_has_server_added_warning(status, text):
    service, _, _ = generator()
    answer = asyncio.run(service.generate("query", result(evidence(text=text), status=status)))
    assert answer.limited and "partial or truncated" in answer.answer


def test_bad_source_lineage_stops_before_models():
    service, mapper, _ = generator()
    for retrieved in [result(evidence(), evidence()), result().model_copy(update={"answer_evidence_ids": ["E99"]})]:
        assert asyncio.run(service.generate("query", retrieved)).status == "unavailable"
    mapper.ainvoke.assert_not_awaited()


def test_parallel_maps_are_bounded_and_ordered():
    service, mapper, reducer = generator(max_parallel=2)
    active = peak = 0
    async def extract(values):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {"relevant": True, "excerpt_ids": [part["excerpt_id"] for part in json.loads(values["context"])["excerpts"]]}
    mapper.ainvoke.side_effect = extract
    answer = asyncio.run(service.generate("query", result(*(evidence(i) for i in range(1, 6)))))
    assert peak == 2 and active == 0 and answer.mapped_evidence == 5
    assert [e["evidence_id"] for e in json.loads(reducer.ainvoke.call_args.args[0]["context"])["evidence"]] == [f"E{i}" for i in range(1, 6)]


def test_timeout_cancels_maps_and_returns_safe_error():
    service, mapper, reducer = generator(timeout=0.01)
    cancelled = []
    async def slow(values):
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
    mapper.ainvoke.side_effect = slow
    answer = asyncio.run(service.generate("query", result()))
    assert answer.reason_code == "answer_generation_timeout" and cancelled
    reducer.ainvoke.assert_not_awaited()


def test_provider_error_is_sanitized():
    service, mapper, _ = generator()
    mapper.ainvoke.side_effect = RuntimeError("private-api-key")
    answer = asyncio.run(service.generate("query", result()))
    assert answer.reason_code == "answer_generation_failed"
    assert "private-api-key" not in answer.model_dump_json()


def test_reducer_can_decline_insufficient_evidence():
    service, _, reducer = generator()
    reducer.ainvoke.return_value = {"status": "insufficient", "paragraphs": []}
    answer = asyncio.run(service.generate("query", result()))
    assert answer.status == "insufficient" and not answer.cited_evidence_ids


def test_supervisor_hands_original_query_to_generator():
    from tests.test_graph_supervisor import service, Plans, retrieve, finish, ROOT
    supervisor, _ = service(Plans(retrieve(), finish()))
    answer_service, _, _ = generator()
    spy = AsyncMock(wraps=answer_service)
    supervisor.answer_generator = spy
    answer = asyncio.run(supervisor.run(ROOT))
    assert answer.answer_generation["status"] == "complete"
    assert spy.generate.call_args.args[0] == ROOT
    assert answer.answer_evidence_ids == ["E1"]


def test_supervisor_preserves_trace_on_answer_failure():
    from tests.test_graph_supervisor import service, Plans, retrieve, finish, ROOT
    supervisor, _ = service(Plans(retrieve(), finish()))
    answer_service, mapper, _ = generator()
    mapper.ainvoke.side_effect = RuntimeError("secret")
    supervisor.answer_generator = answer_service
    answer = asyncio.run(supervisor.run(ROOT))
    assert answer.status == "unavailable" and answer.evidence and answer.trace
    assert answer.answer_evidence_ids == [] and "RAW DEBUG" not in answer.answer
