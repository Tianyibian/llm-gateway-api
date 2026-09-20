# Chat interface and GraphRAG testing

The existing Aster frontend at `http://127.0.0.1:8000/` now displays graph index
readiness and per-message execution details. It uses the real `/api/assistant`
SSE endpoint, not a mock response or a forced GraphRAG route.

## Start locally

```bash
docker desktop start
docker compose -f compose.yaml up -d postgres
source .venv-langchain/bin/activate
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

The explicit Compose filename selects the existing local database configuration
and avoids ambiguity with the separate `docker-compose.yml` file. On a fresh
database, apply the existing Alembic migrations before sending messages.
Configure the [Microsoft index](microsoft-graphrag.md) separately; its runtime
and `.env` credentials remain server-side. Opening the page checks index
readiness without making an LLM call. Sending a question can incur API charges.

## Send and inspect

1. Open the page and confirm the index badge says `Graph index ready`.
   This checks local artifacts, not OpenAI connectivity.
2. Click **GraphRAG: review themes** (or **Explore GraphRAG reviews** on first load).
3. Expand **Execution details** beneath the answer to see:
   - Router: actual selected branch and classification explanation.
   - Guardrail: scope decision and diagnostic reason code.
   - Supervisor: final status, retrieval rounds and call count.
   - Tasks: subquestions, selected tools, parent evidence IDs and errors.
   - Evidence: expandable indexed excerpts and source IDs; selected citations
     are marked `used in answer`.
   A compact panel above the answer also names the selected branch and the
   retrieval workers actually reported in the supervisor trace. These workers
   are Local/Global/DRIFT tool adapters, not separate autonomous agents. Failed
   attempts remain visible; a route selection alone never claims a tool ran.
4. Click **New conversation** to isolate a different test. The send button and
   conversation switching are blocked while a response is in progress.

The app does not force graph routing when a suggestion is clicked. A model
routing mistake stays visible as the actual selected route. Only the GraphRAG
branch uses its scope guardrail. A general greeting or coding question can
legitimately go to the general branch; that is not a graph-guardrail bypass.
Use the direct `/api/graphrag/query` diagnostic endpoint when specifically
testing how that branch rejects a coding question.

## Suggested questions

- `Summarize the recurring support themes described in the sampled Eufy Smart Speaker Essential reviews.`
- `Compare the setup and connectivity concerns in the sampled Eufy Smart Speaker Essential and Belkin Wemo Security System Pro reviews. Describe where retrieved evidence is insufficient.`
- `What can you help me with?` — checks that non-graph routing still works.

A complex-looking question does not guarantee several tool calls. The planner
may decide that one retrieval provides enough evidence; inspect the actual trace.
Use explicit product names: conversation turns are persisted, but the graph
branch currently receives the current question, not a resolved historical
referent. Follow-ups such as `What about its supplier?` can fail scope checks.

## Display and persistence limits

- Guardrail/supervisor details appear when the graph branch returns; they are
  not live intermediate planner events or hidden chain-of-thought.
- The assistant uses SSE, but GraphRAG currently returns cited excerpts after
  retrieval rather than streaming a synthesized answer token by token.
- Only user/assistant text is stored in PostgreSQL. Execution panels are kept
  in the current page session and are not reconstructed from saved messages.
- Reviews are sampled and unverified. Source IDs establish provenance, not
  claim accuracy. Neo4j evidence shows the actual template/dynamic query strategy,
  parameterized Cypher and result truncation after successful execution.
- If conversation history is unavailable, start Docker/PostgreSQL. If graph
  readiness is unavailable, inspect index settings and verification. Model
  failures can still occur even when the local index is ready.

## Automated checks

```bash
.venv-langchain/bin/python -m pytest -q
node --test tests/frontend/graph-inspector.test.cjs
node --check app/static/app.js
```

The Node tests use a small simulated DOM boundary to verify rendering, event
isolation and literal-text evidence handling. They are not browser layout or
real-model tests. The pytest suite checks API contracts and static asset serving.
Live browser verification must be performed separately against real services.

During live integration, OpenAI rejected a classifier enum `$ref` with a sibling
description. The field now keeps a bare reference; the enum/prompt retain route
documentation and validation. A regression test covers that schema shape.
See the [official structured-output reference examples](https://developers.openai.com/api/docs/guides/structured-outputs#definitions-are-supported).
