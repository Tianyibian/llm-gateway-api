# Policy retrieve, filter, rerank and answer

The active policy pipeline is:

1. One policy scope guardrail.
2. Public, embedding-compatible SQL corpus; vector and BM25 ranking independently.
3. RRF over the union of each ranker's top candidates (20 each by default).
4. Explicit metadata filters on that fused pool, not on the SQL corpus.
5. Local Cross-Encoder scores question/title/chunk pairs.
6. Select up to `RAG_RETRIEVAL_K=5` whole chunks under a character budget.
7. Feed those chunks, source headers and citation numbers to the answer LLM.

## Setup

```bash
.venv-langchain/bin/python -m pip install -r requirements/reranker.txt
.venv-langchain/bin/python -m app.cli.prepare_policy_reranker
```

The preparation command downloads `cross-encoder/ms-marco-MiniLM-L6-v2`, pinned to
revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`, under ignored
`.local/policy-reranker/`. It downloads safetensors weights/config/tokenizer files,
not remote Python code. Runtime loading is local-only with `trust_remote_code=False`.
Policies are not uploaded to the model host. Existing embedding and final-answer
providers still receive their usual inputs.

This is an English passage-ranking model, not a validated multilingual reranker.
Its raw relevance logits may be negative or exceed one; never display them as
probabilities or answer confidence. Model details: [official model card](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2).

## Bounds and failure behavior

- `RAG_RERANKER_ENABLED=true` enables the application factory's reranker. Set false
  explicitly for an ablation run; disabled outputs remain labeled `hybrid_rrf`.
- `RAG_RERANKER_MODEL_PATH=.local/policy-reranker` selects the prepared artifact.
- `RAG_RERANKER_TIMEOUT_SECONDS=60` bounds each async scoring request.
- `RAG_CONTEXT_MAX_CHARACTERS=16000` bounds whole chunks plus conservative source
  header allowance. This is a **character** budget, not an exact model-token count;
  conversation history and system/user prompts are separate.
- Cross-Encoder input is limited to 512 tokens per pair; long text can be truncated
  for scoring. Existing ingestion uses 1,200-character chunks by default.
- Inference uses CPU, batches of eight, a cached model and a process-wide busy
  lock. Model work runs off the async event loop. A timed-out thread cannot be
  force-killed; it retains the lock until completion, and new work fails busy.
- Missing weights, invalid scores and inference failure stop answer generation;
  there is no silent fallback pretending that reranking succeeded.
- Post-retrieval filters may yield fewer than k hits or no hits. Never broaden the
  filters or pad with excluded documents. A matching document outside the hybrid
  candidate pool cannot be recovered by the reranker.
- No `policy_type` tagging or inferred metadata filters are added in this step.

## Implementation and verification

- `app/services/policy_reranker.py`: local scorer and loading/concurrency controls.
- `app/services/knowledge_service.py`: hybrid → filter → rerank → bounded context.
- `app/services/assistant_service.py`: source context and cited answer generation.
- Frontend source cards show hybrid rank, reranker rank and score. Execution
  details distinguish the actual reranking path from plain RRF retrieval.

```bash
.venv-langchain/bin/python -m pytest tests/test_policy_reranker.py tests/test_policy_metadata.py -q
.venv-langchain/bin/python -m app.cli.evaluate_policy_reranker
```

Unit tests use explicitly fake scores and inspect the final model context. The
live evaluator uses real PostgreSQL, embeddings, local Cross-Encoder and answer
LLM, without saving conversations. It incurs existing model-provider costs.

Verified on 2026-09-26: 593 backend tests and 19 frontend inspector tests passed.
Live unfiltered/PDF-filtered/no-match retrieval checks passed. The full live
assistant test passed one policy guardrail, retained PDF-only reranked context,
returned five descending-score excerpts, and generated a cited answer. These
are smoke tests rather than a multilingual or comprehensive relevance benchmark.
