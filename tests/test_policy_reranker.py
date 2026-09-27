"""Pipeline order and context contracts; fake scores are explicitly injected."""
import asyncio
from unittest.mock import AsyncMock

import pytest

from app.models.policy_filters import PolicyMetadataFilters
from app.services.errors import KnowledgeBaseNotReadyError
from app.services.policy_reranker import PolicyCrossEncoder
from tests.test_policy_ensemble import make_retriever, row


def test_hybrid_then_metadata_then_rerank_then_top_k():
    rows = [row("a", "refund overview", .01), row("b", "refund exact conditions", .2), row("c", "refund excluded", .3)]
    rows[2][3].category = "Other"
    ranker = AsyncMock(model="fake-cross-encoder")
    ranker.score.return_value = [-1.0, 9.0]
    retriever, _ = make_retriever(rows, default_k=1, candidate_k=3, reranker=ranker)
    results = asyncio.run(retriever.retrieve("refund", filters=PolicyMetadataFilters(categories=["Policy"])))
    assert len(results) == 1 and results[0].source_path == "b.txt"
    assert ranker.score.call_args.args == ("refund", ["Policy\nrefund overview", "Policy\nrefund exact conditions"])
    assert results[0].score_type == "cross_encoder" and results[0].score == 9
    assert results[0].reranker_rank == 1 and results[0].hybrid_rank == 2
    assert results[0].rrf_score is not None


def test_metadata_cannot_rescue_documents_outside_hybrid_pool_or_relax_filters():
    first, outside = row("a", "refund", .01), row("z", "other content", .99)
    first[3].category = "Other"
    ranker = AsyncMock()
    retriever, _ = make_retriever([first, outside], candidate_k=1, default_k=1, reranker=ranker)
    assert asyncio.run(retriever.retrieve("refund", filters=PolicyMetadataFilters(categories=["Policy"]))) == []
    ranker.score.assert_not_awaited()


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], [], [1, 2]])
def test_invalid_reranker_output_fails_closed(scores):
    ranker = AsyncMock()
    ranker.score.return_value = scores
    retriever, _ = make_retriever([row("a", "refund", .1)], reranker=ranker)
    with pytest.raises(KnowledgeBaseNotReadyError):
        asyncio.run(retriever.retrieve("refund"))


def test_equal_scores_preserve_hybrid_order_and_context_budget_keeps_whole_chunks():
    ranker = AsyncMock(model="fake")
    ranker.score.return_value = [1, 1]
    retriever, _ = make_retriever([row("a", "refund " * 40, .1), row("b", "refund " * 40, .2)],
                                 reranker=ranker, context_max_characters=500)
    results = asyncio.run(retriever.retrieve("refund"))
    assert len(results) == 1 and results[0].source_path == "a.txt"
    assert results[0].content == "refund " * 40


def test_local_reranker_failure_is_safe(monkeypatch):
    ranker = PolicyCrossEncoder(model_path="/nonexistent", timeout=.01)
    def failure(*args):
        raise RuntimeError("secret provider details")
    monkeypatch.setattr(ranker, "_score", failure)
    with pytest.raises(KnowledgeBaseNotReadyError, match="unavailable") as exc:
        asyncio.run(ranker.score("query", ["doc"]))
    assert "secret" not in str(exc.value)


def test_local_reranker_timeout_stops_the_request(monkeypatch):
    import time
    ranker = PolicyCrossEncoder(model_path="/nonexistent", timeout=.001)
    def slow(*args):
        time.sleep(.03)
        return [1.0]
    monkeypatch.setattr(ranker, "_score", slow)
    with pytest.raises(KnowledgeBaseNotReadyError):
        asyncio.run(ranker.score("query", ["doc"]))


def test_busy_worker_does_not_start_another_inference():
    from app.services.policy_reranker import _inference_lock
    ranker = PolicyCrossEncoder(model_path="/nonexistent")
    with _inference_lock:
        with pytest.raises(KnowledgeBaseNotReadyError):
            asyncio.run(ranker.score("query", ["doc"]))


def test_context_budget_never_returns_an_oversized_excerpt():
    retriever, _ = make_retriever([row("a", "long " * 200, .1)], context_max_characters=500)
    with pytest.raises(KnowledgeBaseNotReadyError, match="context budget"):
        asyncio.run(retriever.retrieve("policy"))


def test_empty_candidates_skip_local_inference(monkeypatch):
    ranker = PolicyCrossEncoder(model_path="/nonexistent")
    assert asyncio.run(ranker.score("query", [])) == []


def test_only_reranked_top_chunks_reach_policy_answer_context():
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from app.services.assistant_service import AssistantGraphService
    from app.models.schemas import QueryRoute
    from tests.test_assistant_service import FakeClassifier, AllowPolicyGuardrail

    prompts = []
    class Capture(BaseCallbackHandler):
        def on_chat_model_start(self, serialized, messages, **kwargs):
            prompts.extend(message.content for batch in messages for message in batch)
    ranker = AsyncMock(model="fake")
    ranker.score.return_value = [-1, 7]
    retriever, _ = make_retriever([row("a", "LOSE_CONTEXT", .1), row("b", "WIN_CONTEXT", .2)],
                                 default_k=1, reranker=ranker)
    service = AssistantGraphService.from_model(classifier=FakeClassifier(QueryRoute.POLICY_SEARCH),
        policy_guardrail=AllowPolicyGuardrail(), knowledge_retriever=retriever,
        model_client=FakeListChatModel(responses=["Answer [1]"], callbacks=[Capture()]), provider="fake", model="fake")
    async def run():
        return [event async for event in service.stream("return policy")]
    events = asyncio.run(run())
    prompt = "\n".join(prompts)
    assert "WIN_CONTEXT" in prompt and "LOSE_CONTEXT" not in prompt
    sources = next(payload for name, payload in events if name == "sources")
    assert sources["documents"][0]["source_path"] == "b.txt"
    assert sources["retrieval_method"].endswith("cross_encoder")
