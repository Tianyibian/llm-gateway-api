# Microsoft GraphRAG runtime

Production orchestration uses the [direct task planner](graph-task-planner.md):
the planner assigns document-analysis subtasks directly to
the request-selected Local (default) or Global retrieval tool. DRIFT remains a
low-level experimental adapter. See [Local vs Global and API selection](local-global-search.md).
Our outer MapReduce remains a separate final-answer stage.

## Implemented architecture

This integration calls the official `graphrag==3.1.2` Python APIs. It builds
an extracted entity/relationship graph, detects communities, generates
community reports, and embeds indexed content. It is not a renamed vector
search or a synthetic retrieval fixture.

```text
Reviewed catalog + review text
  -> Microsoft standard indexing
     -> Parquet: documents, text units, entities, relationships, communities, reports
     -> LanceDB: entity descriptions, text units, community reports

FastAPI /api/graphrag/query (or the assistant's graph branch)
  -> scope guardrail
  -> task planner: decompose -> dispatch tools -> inspect/replan
     -> request-pinned Local / Global adapter
        -> isolated Microsoft GraphRAG process -> OpenAI + local index
     -> source-backed excerpts -> extractive Map -> grounded Reduce -> cited answer
```

PostgreSQL still stores conversations; pgvector still serves the separate
Knowledge Base RAG path. Microsoft GraphRAG uses its own Parquet/LanceDB
workspace. Neo4j is an independent optional backend; see
[its two query strategies and setup](neo4j-query-strategies.md).
An unavailable Neo4j task is not silently answered by another tool.

| Mode | Retrieval role |
|---|---|
| Local | Entity-focused context from graph relationships and source text |
| Global | Corpus-level themes using community reports |
| DRIFT | Community-informed exploratory search with follow-up local retrieval |

The supervisor chooses among registered tools using the question and prior
evidence. Its tasks are validated Pydantic plans, not arbitrary executable code.
The outer supervisor and DRIFT both have bounded iteration limits.

## Isolated installation

Run from the repository root. The application stays in `.venv-langchain`;
GraphRAG's dependencies are installed separately:

```bash
python3.13 -m venv .venv-graphrag
.venv-graphrag/bin/python -m pip install -r requirements/graphrag.txt
.venv-graphrag/bin/python -m pip check
```

The tested interpreter is Python 3.13.9. The direct GraphRAG dependency is
pinned; transitive dependencies are not fully locked, so a fresh installation
still requires verification. No Azure or Neo4j account is required by this
OpenAI-backed configuration.

### Source installation

The local runtime can use the official Git checkout instead of the published
GraphRAG packages. Keep the same `v3.1.2` release to avoid changing the index
format or dependency contract. From the project root, after provisioning the
isolated environment above:

```bash
# Initialize the commit pinned by this project's Git submodule.
git submodule update --init --recursive third_party/graphrag
git -C third_party/graphrag rev-parse HEAD
# Expected commit: 243637c4eb94e34c3a5e5c7d871a725e8d6b77fc
.venv-graphrag/bin/python -m pip install --no-deps -r requirements/graphrag-source.txt
.venv-graphrag/bin/python -m pip check
.venv-graphrag/bin/python -c 'import graphrag; print(graphrag.__file__)'
```

The import path should point into `third_party/graphrag/packages/graphrag/`.
All eight upstream GraphRAG workspace packages are editable installs; unrelated
dependencies remain installed distributions. `--no-deps` preserves the existing
dependency set; it is not a substitute for provisioning dependencies first.
Build dependencies may still be downloaded during installation.

This source checkout is a Git submodule: the parent repository records the
official upstream URL and exact commit, not a copy of the upstream files.
It remains excluded from the application Docker context.
See [source navigation and clone instructions](../third_party/README.md).
Keep it in place while editable installs are active. Restart running processes
after switching installations or editing source. Switching to the identical
release does not itself require reindexing, but later source or model changes
may invalidate index compatibility. Our runtime still enforces version `3.1.2`.

To switch back to published packages without deleting the source checkout:

```bash
.venv-graphrag/bin/python -m pip install --force-reinstall --no-deps \
  graphrag==3.1.2 graphrag-cache==3.1.2 graphrag-chunking==3.1.2 \
  graphrag-common==3.1.2 graphrag-input==3.1.2 graphrag-llm==3.1.2 \
  graphrag-storage==3.1.2 graphrag-vectors==3.1.2
.venv-graphrag/bin/python -m pip check
```

## Prepare, approve, and index

For `merged_reviews.csv`, use the [merged review importer](merged-review-indexing.md).
It reconstructs individual reviews, validates original IDs, removes customer
fields, and creates token-bounded per-product documents in a new workspace.

First prepare a **new** small corpus using the
[business data exporter](business-graphrag.md). Review every exported text file
for privacy and quality before granting permission to send it to OpenAI.
Column allowlists and PII heuristics do not replace this review.

The example below assumes a reviewed `.local/graphrag-review-seed-v3` exists.
Use a new workspace name; preparation refuses to overwrite an existing one.

```bash
.venv-graphrag/bin/python graphrag_runtime/worker.py prepare \
  --source .local/graphrag-review-seed-v3 \
  --root .local/graphrag-ms-business-v2 --approve-reviewed-inputs

# Paid OpenAI calls: run only after reviewing and approving the corpus.
.venv-graphrag/bin/python graphrag_runtime/worker.py index \
  --root .local/graphrag-ms-business-v2
.venv-graphrag/bin/python graphrag_runtime/worker.py verify \
  --root .local/graphrag-ms-business-v2
```

Preparation copies the tracked settings and extraction prompt, not credentials.
The worker reads `GRAPHRAG_API_KEY` / `OPENAI_API_KEY` from the process environment,
or `OPENAI_API_KEY` from the repository's ignored `.env`. It never copies the
key into the index. Inputs are limited to 64 documents / 200 KB per workspace.

The template uses `gpt-5.6-luna` for extraction/query generation and
`text-embedding-3-small` with 1,536-dimensional vectors. Index concurrency is
two; DRIFT uses one follow-up at one depth. Review settings before a new build.
Changing the embedding model or dimensions requires a compatible new index.

Verification checks six Parquet artifacts and three nonempty vector tables,
then records configuration, corpus-manifest, and artifact hashes. The adapter
checks the verification record before registration and querying. This is a
local consistency check, not a signed attestation or a security boundary against
someone who can modify the workspace. Vector contents are checked at verify
time, not cryptographically hashed on every API request.

## Enable the API

The active local index is `.local/graphrag-ms-merged-full-20260926`, containing
all 5,000 prepared reviews. See [full build verification and query limitations](merged-review-indexing.md#full-index-activation).
The earlier `.local/graphrag-ms-business-v2` is retained for rollback. The original
`.local/graphrag-ms-business-v1` is also retained but has a damaged report artifact.
Put these
non-secret settings in `.env`, retaining the existing server-side OpenAI key:

```dotenv
MICROSOFT_GRAPHRAG_ENABLED=true
MICROSOFT_GRAPHRAG_ROOT=.local/graphrag-ms-merged-full-20260926
MICROSOFT_GRAPHRAG_PYTHON=.venv-graphrag/bin/python
MICROSOFT_GRAPHRAG_TIMEOUT_SECONDS=90
```

Restart Uvicorn in the application environment:

```bash
source .venv-langchain/bin/activate
uvicorn app.main:app --reload
curl http://127.0.0.1:8000/api/graphrag/status
curl -X POST http://127.0.0.1:8000/api/graphrag/query \
  -H 'Content-Type: application/json' \
  -d '{"query":"Summarize the recurring support themes described in the sampled Eufy Smart Speaker Essential reviews."}'
```

`/query` returns JSON with status, evidence, citations, and execution trace.
The assistant endpoint retains its SSE interface. After retrieval, MapReduce
builds a natural-language answer with validated source IDs. Progress events are
live; final text is buffered until validation, then emitted in segmented deltas.
This does not stream the internal GraphRAG model's generation tokens.
The scope-only `/guardrail` endpoint deliberately
does not inspect the index: use `/status`, not its `retrieval_ready: false`
field, to inspect adapter readiness.

Import `postman/Microsoft_GraphRAG_Live.postman_collection.json`, set `base_url`
to your running server, and run its three requests in order. No key belongs in
Postman. Set its request timeout to at least 360 seconds for the supervisor.
Postman does not start the server, and these requests do not use mock tools.

## Verification

```bash
# Deterministic tests: injected model/process outputs, no API cost.
.venv-langchain/bin/python -m pytest -q

# Real FastAPI routing + guardrail + supervisor + real Microsoft index/LLM.
# In-process ASGI transport, no dependency overrides or mock retrieval.
.venv-langchain/bin/python -m app.cli.evaluate_live_graphrag \
  --output .local/graphrag-real-api-supervisor.json

# Direct official search API smoke test; replace local with global or drift.
.venv-graphrag/bin/python graphrag_runtime/worker.py query \
  --root .local/graphrag-ms-business-v2 --method local \
  --query 'What setup concerns appear in the sampled Eufy Smart Speaker Essential reviews?' \
  --output .local/graphrag-local-query.json
```

The initial real build used 20 catalog documents plus 21 grouped review
documents containing 100 selected reviews. It produced 41 text units,
99 entities, 110 relationships, 9 communities, and 9 community reports.
Vector tables contain 99 entity, 41 text-unit, and 9 report vectors. Indexing
took approximately 183 seconds on this local run; latency is not a benchmark.

The replacement `v2` uses exactly the same 41 approved input documents and
extraction configuration. A fresh standard build took about 147 seconds and
produced 97 entities, 108 relationships, and 9 community reports. Model-derived
extractions can differ between runs. Both real Local and Global smoke queries
returned evidence and passed reference-ID validation; this does not establish
claim-level accuracy. The original damaged workspace was not overwritten.

All three official query modes have been exercised against this index. The
live FastAPI/supervisor smoke test passed seven connectivity/provenance checks.
The deterministic regression suite passed 203 tests. These results do not
establish semantic accuracy, complete coverage, or production security.

## Read Parquet without editing the index

Parquet is a binary table format, not editable plain text. Use the isolated
runtime's read-only preview command (no model calls):

```bash
.venv-graphrag/bin/python -m app.cli.inspect_graphrag \
  --root .local/graphrag-ms-business-v2 --table entities --limit 5
.venv-graphrag/bin/python -m app.cli.inspect_graphrag \
  --root .local/graphrag-ms-business-v2 --table community_reports --limit 3
.venv-graphrag/bin/python -m app.cli.inspect_graphrag \
  --root .local/graphrag-ms-business-v2 --table relationships --schema
```

Use the workspace selected by `MICROSOFT_GRAPHRAG_ROOT`. The viewer reports the
total row count and columns, then reads a bounded batch. Long cells are shortened
for display only. `--columns title description` selects specific fields.
For notebook exploration, `pandas.read_parquet(path)` loads an in-memory
DataFrame; inspecting or changing that DataFrame does not change the file unless
you explicitly write it back. Never save changes into a verified `output/`.

If an index artifact is damaged, preserve the old workspace and build a new
version from the approved inputs. Verify its tables/vector store and perform a
real query before updating the local `MICROSOFT_GRAPHRAG_ROOT`. Rebuilding calls
the configured model API and can incur charges. Do not merely regenerate the
verification hashes to approve a modified output file.

## Evidence quality and deployment limits

- Source IDs resolve to actual retrieved text-unit/report records, not model
  invented IDs. The worker checks generated citation IDs against context tables.
  Invalid references trigger an extractive fallback. ID validity does **not**
  prove that a cited passage supports every generated claim.
- The supervisor consumes indexed excerpts, not the worker's generated answer.
  Selected excerpts feed parallel extractive maps and a grounded answer reducer.
  Quote and citation checks do not prove claim-level factual correctness.
  Community reports are themselves model-generated summaries.
- Sample reviews contain contradictions and some implausible product features.
  They are unverified opinions, not manufacturer specifications or return policy.
  The deterministic sample is concentrated on two products and is not representative
  of all 5,000 source reviews. Missing retrieved evidence is not proof of absence.
- Extracted SupportTopic nodes and relationship descriptions are model outputs.
  The domain prompt does not enforce Neo4j constraints or guarantee exact joins.
- This is one explicitly approved local corpus. Authentication, tenant-specific
  corpus ACLs, rate limiting, deployment-wide concurrency limits, spend controls,
  and held-out relevance/security evaluations are required before public exposure.
  The per-client worker semaphore is not a global cross-request concurrency cap.
- Indexes, raw query reports, and `.env` stay Git-ignored. Query reports may contain
  business text; inspect them before sharing. The diagnostic endpoints do not
  write conversations or replace ownership checks in conversation routes.

Official references: [Microsoft GraphRAG setup](https://microsoft.github.io/graphrag/get_started/),
[query modes](https://microsoft.github.io/graphrag/query/overview/), and
[OpenAI embedding model](https://developers.openai.com/api/docs/models/text-embedding-3-small).
