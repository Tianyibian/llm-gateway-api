# Guarded policy retrieval

Optional [metadata filters](policy-metadata-filters.md) narrow the fused hybrid
candidates before local Cross-Encoder reranking. Public visibility is enforced
before any retrieval; user filters are not access permissions.

The `policy_search` branch follows this workflow:

`router (or clarification) -> assess_policy_scope -> retrieve_knowledge -> knowledge_answer`

Rejection, uncertainty and validation failure end the branch before embeddings,
database retrieval or answer generation. The scope gate uses structured model
output, validated by Pydantic, and a server-owned decision rule. It sees the original
request, the resolved question and conversational context. The model cannot rewrite
the question or grant access. Policies, published support procedures, manuals and
troubleshooting are supported; mixed/out-of-scope requests, private records,
credentials, writes and rule-bypass instructions are rejected. Ambiguous requests
receive a fixed clarification question. A missing gate or invalid/failed model
response fails closed. This is not a global guardrail and not a guarantee that an
LLM can identify every adversarial prompt.

## Ensemble retriever

`KnowledgeRetriever` implements a project-owned ensemble, not LangChain's
`EnsembleRetriever` class. Its provider-independent ranking helpers are in
`app/services/ensemble_retriever.py`. Reranking adds local inference dependencies
but requires no database migration. See [reranker setup](policy-reranker.md).

1. Embed the resolved question with the same provider/model used for ingestion.
2. PostgreSQL computes exact pgvector cosine distances for eligible public chunks.
3. Independently score all eligible chunks with Okapi BM25 (`k1=1.5`, `b=0.75`),
   using title + content. BM25 does **not** only rerank the vector candidates.
4. Take the top candidate window from each ranking, then fuse with equal-weight
   reciprocal-rank fusion: `score(chunk) = sum(1 / (60 + rank))`.
5. Deduplicate by stable chunk ID and retain the fused candidate union (up to 40
   by default), then apply explicit metadata filters. Do not refill excluded hits.
6. Score the remaining question/passage pairs with a local Cross-Encoder; select
   up to five whole chunks within the context-character budget.
7. Generate a cited answer from these excerpts. Retrieved text is untrusted data,
   not system instructions; insufficient evidence should produce an explicit limitation.

Both rankings share public visibility and embedding-provider/model constraints
from one database snapshot. Category/source filters follow fusion. No private chunks are used for BM25
statistics. Citations retain source file, page and chunk index, and include separate
vector/BM25 ranks and raw scores. An RRF score is **not** a confidence percentage.
Zero-keyword matches are excluded from the BM25 ranking; vector-only results can
still be returned. Failure of a backend is not silently represented as hybrid success.

This prototype scans a bounded corpus, calculating distances in PostgreSQL and
BM25 in a worker thread. It does not use an ANN index or a persistent BM25 index.
It rejects oversized corpora rather than calculating BM25 on a truncated subset.
For large deployments, replace this scan with a maintained lexical search index
and indexed vector candidates while retaining identical authorization filters.
Tokenization is Unicode-aware but has no stemming, language-specific segmentation
or translation. English policy queries best match the current English corpus.
Hybrid retrieval improves complementary matching; it does not prove relevance or
the factual correctness of every generated sentence. There is no independent
answer-entailment verifier in this branch.

## Configuration and observable behavior

- `RAG_RETRIEVAL_K=5`: final excerpts.
- `RAG_ENSEMBLE_CANDIDATE_K=20`: candidate window per method (at least final k).
- `RAG_BM25_MAX_CORPUS_CHUNKS=10000`: bounded scan limit.
- `POLICY_GUARDRAIL_TIMEOUT_SECONDS=45`.
- `POLICY_GUARDRAIL_MIN_CONFIDENCE=0.75`: model scope-assessment threshold, not answer confidence.

The SSE stream emits `policy_guardrail` before retrieval, then `sources` with
`backend=policy_ensemble` and `retrieval_method=pgvector_bm25_rrf_metadata_cross_encoder`
when reranked sources are present. The prototype UI
shows these events in Execution details and per-source ranking provenance.

Try: `What is the return policy?`, `How do I reset my account password?`, or
`What is the warranty policy for this product?` (provide a product for specific terms).
A router may send unrelated questions to other branches; testing a rejection at
this gate directly is different from testing the top-level router.

## Verification

Deterministic tests (fake models, no API charges):

```bash
.venv-langchain/bin/python -m pytest tests/test_policy_ensemble.py tests/test_assistant_service.py -q
node --test tests/frontend/graph-inspector.test.cjs
```

Opt-in real-provider and read-only database evaluation:

```bash
.venv-langchain/bin/python -m app.cli.evaluate_policy_ensemble
```

The live evaluator uses the configured model and embeddings and can incur API
charges. It does not ingest data or persist conversations. Small live cases are
smoke tests, not a comprehensive security or retrieval-quality benchmark.
