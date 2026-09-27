"""Deterministic guardrail contracts, graph short-circuiting and hybrid ranking."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from sqlalchemy.dialects import postgresql

from app.models.schemas import QueryClassification, QueryRoute
from app.services.assistant_service import AssistantGraphService
from app.services.ensemble_retriever import bm25_rank, reciprocal_rank_fusion, tokenize
from app.services.errors import KnowledgeBaseNotReadyError
from app.services.knowledge_service import KnowledgeRetriever
from app.services.policy_guardrail import PolicyGuardrail


class Chain:
    def __init__(self, result):
        self.result, self.inputs = result, []

    async def ainvoke(self, value):
        self.inputs.append(value)
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


@pytest.mark.parametrize("assessment,action", [
    ({"scope": "in_scope", "unsafe": False, "confidence": .99}, "allow"),
    ({"scope": "out_of_scope", "unsafe": False, "confidence": .99}, "reject"),
    ({"scope": "mixed", "unsafe": False, "confidence": .99}, "reject"),
    ({"scope": "in_scope", "unsafe": True, "confidence": .99}, "reject"),
    ({"scope": "unclear", "unsafe": False, "confidence": .99}, "clarify"),
    ({"scope": "in_scope", "unsafe": False, "confidence": .5}, "clarify"),
    ({"scope": "in_scope", "unsafe": False, "confidence": 2}, "unavailable"),
    ({"scope": "in_scope", "unsafe": False, "confidence": float("nan")}, "unavailable"),
    ({"allowed": True}, "unavailable"),
    ({"scope": "in_scope", "unsafe": False, "confidence": .99, "allowed": True}, "unavailable"),
    ({"scope": "in_scope", "unsafe": "false", "confidence": .99}, "unavailable"),
    (RuntimeError("Secret provider diagnostics"), "unavailable"),
])
def test_guard_contract(assessment, action):
    result = asyncio.run(PolicyGuardrail(chain=Chain(assessment)).assess("What is the return policy?"))
    assert result["action"] == action
    assert result["allowed"] is (action == "allow")
    assert "Secret provider" not in str(result)


def test_guard_timeout_is_closed():
    class Slow:
        async def ainvoke(self, value):
            await asyncio.sleep(1)
    result = asyncio.run(PolicyGuardrail(chain=Slow(), timeout=.001).assess("policy"))
    assert result["action"] == "unavailable"


@pytest.mark.parametrize("via_clarification", [False, True])
@pytest.mark.parametrize("guard_state", ["allow", "reject", "unavailable", "missing"])
def test_graph_enforces_policy_gate(via_clarification, guard_state):
    class Classifier:
        async def classify(self, query, **kwargs):
            return QueryClassification(
                route=QueryRoute.ADDITIONAL_SEARCH if via_clarification else QueryRoute.POLICY_SEARCH,
                resolved_query="Resolved policy question", reason="Test routing", confidence=1,
            )
    class Clarifier:
        async def assess(self, query, **kwargs):
            return {"action": "ready", "resolved_query": "Resolved policy question", "next_route": "policy_search"}
    class Retriever:
        calls = 0
        async def retrieve(self, query):
            self.calls += 1
            return []

    async def run():
        chain = Chain({"scope": "in_scope" if guard_state == "allow" else "out_of_scope", "unsafe": False, "confidence": .99}
                      if guard_state != "unavailable" else ValueError("invalid"))
        retriever = Retriever()
        service = AssistantGraphService.from_model(
            classifier=Classifier(), clarification_service=Clarifier(),
            policy_guardrail=None if guard_state == "missing" else PolicyGuardrail(chain=chain),
            knowledge_retriever=retriever, model_client=FakeListChatModel(responses=["must not run"]),
            provider="fake", model="fake",
        )
        events = [item async for item in service.stream("Original untrusted request", history=[("human", "context")])]
        names = [item[0] for item in events]
        assert retriever.calls == (1 if guard_state == "allow" else 0)
        assert "policy_guardrail" in names
        if guard_state == "allow":
            assert names.index("policy_guardrail") < names.index("sources")
        else:
            assert "sources" not in names
        if chain.inputs:
            context = json.loads(chain.inputs[0]["context"])
            assert context["original_query"] == "Original untrusted request"
            assert context["resolved_query"] == "Resolved policy question"
            assert context["history"] == [["human", "context"]]
        assert not any("must not run" in str(payload) for _, payload in events)
    asyncio.run(run())


def test_bm25_keywords_zero_matches_and_normalization():
    corpus = {"a": "ZX-100 warranty warranty", "b": "Shipping returns", "c": ""}
    ranking = bm25_rank("ＺＸ-100 WARRANTY", corpus, limit=10)
    assert [key for key, _ in ranking] == ["a"]
    assert bm25_rank("missing", corpus, limit=10) == []
    assert bm25_rank("", corpus, limit=10) == []
    assert bm25_rank("warranty", {}, limit=10) == []
    assert tokenize("退款") == ["退", "款"]


def test_bm25_rare_term_and_length_normalization():
    assert bm25_rank("refund", {"a": "refund", "b": "refund " + "other " * 100}, limit=2)[0][0] == "a"


def test_rrf_deduplicates_and_combines_ranks_not_scores():
    result = reciprocal_rank_fusion(["a", "a", "b"], ["b", "c"], limit=3)
    assert result[0][0] == "b"
    assert result[0][1] == pytest.approx(1 / 62 + 1 / 61)
    assert len(result) == 3
    assert reciprocal_rank_fusion([], [], limit=5) == []


class Embeddings:
    provider = "fake"
    model = "fake-embedding"
    async def embed_query(self, query):
        return [1.0] * 768


def make_retriever(rows, **kwargs):
    statements = []
    class Session:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def execute(self, statement):
            statements.append(statement)
            return SimpleNamespace(all=lambda: rows)
    return KnowledgeRetriever(session_factory=Session, embedding_service=Embeddings(), **kwargs), statements


def row(key, content, distance):
    document = SimpleNamespace(title="Policy", source_path=f"{key}.txt", document_metadata={"source_file": f"{key}.txt"}, source_type="txt", category="Policy")
    return key, content, 0, document, distance


def test_ensemble_can_recover_lexical_hit_outside_vector_candidates_and_filters():
    retriever, statements = make_retriever([
        row("a", "general policy", .01), row("b", "ZX100 exact warranty", .8),
    ], default_k=1, candidate_k=1)
    # k=2 widens both candidate sets: the dual-ranked lexical hit wins fusion.
    results = asyncio.run(retriever.retrieve("ZX100", k=2, category="Policy"))
    assert results[0].source_path == "b.txt"
    assert results[0].vector_rank == 2 and results[0].bm25_rank == 1
    citation = results[0].citation()
    assert citation["score_type"] == "hybrid_rrf"
    assert citation["bm25_score"] > 0
    # BM25 searches beyond the vector candidate window. Equal RRF scores use ID ties.
    narrow, _ = make_retriever([row("z", "general policy", .01), row("a", "ZX100 warranty", .8)],
                              default_k=1, candidate_k=1)
    recovered = asyncio.run(narrow.retrieve("ZX100"))[0]
    assert recovered.source_path == "a.txt"
    assert recovered.vector_rank is None and recovered.bm25_rank == 1
    compiled = statements[0].compile(dialect=postgresql.dialect())
    sql = str(compiled)
    assert "visibility =" in sql and "embedding_provider =" in sql and "embedding_model =" in sql
    assert "category =" not in sql and "<=>" in sql  # Metadata filtering follows hybrid ranking.
    assert "public" in compiled.params.values()
    assert "fake-embedding" in compiled.params.values()


@pytest.mark.parametrize("rows,limit", [([], 10), ([row("a", "a", .1), row("b", "b", .2)], 1)])
def test_empty_or_oversized_corpus_is_not_silently_searched(rows, limit):
    retriever, _ = make_retriever(rows, max_corpus_chunks=limit)
    with pytest.raises(KnowledgeBaseNotReadyError):
        asyncio.run(retriever.retrieve("policy"))


@pytest.mark.parametrize("query,k", [("", 5), ("policy", 0), ("policy", 101)])
def test_bad_retrieval_input(query, k):
    retriever, statements = make_retriever([])
    with pytest.raises(ValueError):
        asyncio.run(retriever.retrieve(query, k=k))
    assert not statements
