# GraphRAG scope guardrail and backend selection

Production now uses a [two-level business supervisor and specialist agents](hierarchical-agents.md).
The root scope-only gate permits supported cross-backend composite questions but
does not grant tool capabilities. The backend-selection checks below remain
mandatory inside each specialist before individual tool calls.

## Current implementation

The assistant's `graph_rag_search` branch now contains a LangGraph subgraph
with a semantic scope gate. Other assistant branches do not use this gate.
An LLM selects either `neo4j` or `microsoft_graphrag` using three inputs:

1. A server-owned, read-only business scope definition.
2. An approved domain schema and backend capability descriptions.
3. The user's question, passed as an untrusted human message.

The policy lives in `app/graphrag_policy.json`. It is a **prospective domain
contract**, not a schema discovered from a connected Neo4j instance. The real
Microsoft index uses a domain extraction prompt, but its model-extracted
relationships are not hard constraints. Entity names in tests are not proof
that records exist.

The LangChain pipeline is `ChatPromptTemplate | model.with_structured_output`.
Its output is validated again with Pydantic and deterministic Python checks:
allowed entity types, directed relationships, grounded name mentions, detected
risks, confidence threshold, and backend/intent compatibility. Requests cannot
override the server policy or supply execution permissions.

| Decision | Meaning |
|---|---|
| `allow` | Within the approved scope; includes a recommended backend and eligible tools. |
| `clarify` | Mixed tasks, missing referents, low confidence, or an incompatible backend choice. |
| `reject` | Outside the scope, unsupported schema, ungrounded extraction, or a detected risk. |
| `unavailable` | Model timeout, provider failure, or malformed model output; never bypass the gate. |

`allow` is **not execution authorization**. Every scope response currently includes
`retrieval_ready: false` and `entity_resolution: "not_performed"`.
The adaptive supervisor is implemented and connected after the allow edge.
The optional Microsoft GraphRAG 3.1.2 adapter now connects a real index; inspect
`GET /api/graphrag/status` for execution readiness. The scope-only response does
not inspect the index, which is why its `retrieval_ready` stays false. The optional
[Neo4j worker](neo4j-query-strategies.md) independently verifies its imported
business snapshot before selecting a template or constrained dynamic query.
With no enabled adapters, the supervisor returns
`tools_not_connected`. Model confidence is not a calibrated safety score,
and a prompt-based gate cannot replace authentication or tool permissions.

## Intended backend capabilities

| Question type | Backend recommendation | Eligible retrieval |
|---|---|---|
| Exact product/supplier connections and bounded catalog traversal | Neo4j | Read-only relationship queries |
| Document-grounded explanation around an entity | Microsoft GraphRAG | Local search |
| Themes across the knowledge corpus | Microsoft GraphRAG | Global search over community reports |
| Exploratory cross-document synthesis | Microsoft GraphRAG | Local or DRIFT search |

This is a capability policy for this application, not a claim that Neo4j can
never perform semantic retrieval. Microsoft GraphRAG is a retrieval/indexing
framework; Neo4j is a graph database. They are not interchangeable products.

## API and Postman

Start the server in the existing environment:

```bash
source .venv-langchain/bin/activate
uvicorn app.main:app --reload
```

The diagnostic endpoint invokes the configured real model but performs no
retrieval or conversation writes:

```bash
curl -X POST http://127.0.0.1:8000/api/graphrag/guardrail \
  -H 'Content-Type: application/json' \
  -d '{"query":"Who supplies Philips Hue Smart Lock Max?"}'
```

Import `postman/GraphRAG_Guardrail.postman_collection.json`, set `base_url`,
and run the collection. Postman does not start Uvicorn or mock the model.
Each request asserts a specific semantic outcome; a provider error does not
count as a passing rejection. Responses use HTTP 200 for completed decisions,
503 for unavailable assessments, and 422 for invalid request bodies.

The guardrail follows `LLM_PROVIDER` and its chat model. Restart Uvicorn after
configuration changes. Ollama runs never silently switch to a paid provider.
When OpenAI is selected, the model receives the question plus the policy's
scope/schema/capabilities. No business files or database records are loaded by
this service. Keep API credentials in the ignored local `.env`.

This diagnostic endpoint is intended for local development. Authentication,
rate limiting, and per-tenant data permissions are prerequisites for public
deployment; the semantic gate is not an authorization boundary.

## Testing

Deterministic contract and integration tests require no model or database:

```bash
python -m pytest -q
```

They inject model outputs to verify validation, failures, FastAPI responses,
and LangGraph branch isolation. They do **not** establish model classification
accuracy. Evaluate that separately with the explicit, opt-in live runner:

```bash
python -m app.cli.evaluate_graphrag_guardrail --provider ollama \
  --include-assessments --output .local/graphrag-ollama.json

# Paid API calls; uses the local key and OPENAI_CHAT_MODEL.
python -m app.cli.evaluate_graphrag_guardrail --provider openai \
  --include-assessments --output .local/graphrag-openai.json
```

Use `--case neo4j_supplier` for a one-question smoke test and `--model` for an
explicit model override. The runner evaluates English and Chinese questions,
mixed requests, prompt-injection attempts, unsupported relationships, missing
referents, and benign quoted attacks. Failed expectations exit with code 1.
Reports are ignored by Git and contain structured assessments, not hidden
chain-of-thought or credentials. No retrieval is executed by this runner.

These fixtures are a development evaluation set, not an independent safety
benchmark. Keep new paraphrases and adversarial cases as a held-out evaluation
before enabling real tools. A passing small set does not prove a secure gate.

## Preparing Microsoft GraphRAG

The [implemented Microsoft GraphRAG runtime](microsoft-graphrag.md) follows the
preparation principles below and has been tested on an approved small catalog
and review corpus with real OpenAI calls.

1. Use a separate virtual environment and pin a tested GraphRAG version.
   Initialize a dedicated indexing workspace, **not the application root**:
   GraphRAG initialization generates its own configuration and environment file.
2. Select a small, public knowledge corpus first. Normalize content and assign
   stable source IDs, source versions, and visibility metadata. Exclude private
   customer records and internal documents. Metadata alone does not enforce
   access control: use isolated authorized corpora/indices and enforce access
   again at retrieval time.
3. Prepare product/supplier/category facts from approved CSVs using stable IDs.
   Keep exact financial aggregations in the existing analytics branch. Review
   any text representations of structured facts before indexing them.
4. Build text units, extracted entities/relationships, communities/reports,
   and embeddings. Start small and inspect extraction quality and source
   provenance before committing to a larger indexing bill.
5. Audit how extracted relationships map to the approved domain contract.
   Microsoft extraction labels are not automatically constrained by Neo4j's
   labels or the application's policy. Record index version and capabilities.
6. Connect read-only local/global/DRIFT adapters with normalized evidence
   results and source citations. Validate entity existence, permissions,
   index readiness, and evidence lineage before each execution.

Microsoft GraphRAG does not require a Neo4j database. Prepare Neo4j separately
with stable business IDs, constraints, approved relationship ingestion,
read-only application credentials, and bounded parameterized query templates.
Do not execute arbitrary LLM-generated Cypher.

Official references:
[getting started](https://microsoft.github.io/graphrag/get_started/),
[indexing dataflow](https://microsoft.github.io/graphrag/index/default_dataflow/),
[query modes](https://microsoft.github.io/graphrag/query/overview/).

## Implemented: GraphRAG-only adaptive supervisor

The implemented flow in `app/services/graph_supervisor.py` is:

The root handoff supplies only the original request question (with surrounding
whitespace trimmed) and a server-computed `scope_approved` boolean. The guardrail
does not generate or rewrite a question or supply a decomposition. Its detailed
diagnostics remain available for observability but are not passed to the planner.
Only the supervisor creates subtask questions and chooses their tools. Each
proposed subtask still undergoes scope, capability, and evidence-lineage checks
before execution. `scope_approved` is an internal argument, never a client-settable
request field. Rejection, clarification, and assessment failure all stop entry.

Root decisions of `reject` or `clarify` immediately return the fixed English
message `Sorry, this type of query is not supported.` without planning or
retrieval. Reason codes remain available for diagnostics. Assessment failures
instead return a retry-later message (HTTP 503 on the direct query endpoint),
so a provider outage is not misrepresented as an unsupported topic. These
messages are deterministic, not generated with another LLM call.

```text
Existing classifier
  -> GraphRAG branch
     -> scope guardrail
        -> decompose business goals
           -> delegate independent specialist agents
              -> each agent plans, validates and calls its own tools
                 -> merge evidence / plan dependent assignments
                    -> parallel evidence maps -> grounded reduce -> cited answer
```

The business supervisor uses `DelegationPlan` to choose specialist agents.
Each specialist independently uses `GraphPlan` to choose its allowed retrieval
tools based on its assigned question and returned evidence. Both levels use
LangGraph conditional edges and bounded `asyncio.gather` execution, not arbitrary
model-generated code or an agent for every sentence.
Dependent tasks run in later rounds. There are no runtime-generated arbitrary
tools, SQL, Cypher, or executable code.

All proposed tasks must be revalidated. New entities introduced by retrieval
need traceable evidence, not fabricated model names. The entire batch is checked
before any retrieval begins: registered tool, scope decision, backend/tool match,
known parent evidence IDs, and original-question/cited-evidence entity lineage.
The model still makes semantic judgments, including when evidence is sufficient;
these checks cannot mathematically prove relevance or answer completeness.

`SupervisorLimits` defaults to 3 retrieval rounds, 6 retrieval calls, 3 parallel
workers, 3 evidence items per call, 2,000 characters per excerpt, a 30-second
planner/tool timeout, and a 120-second supervisor timeout. There can be at most
4 planner invocations and 6 per-task scope checks, in addition to the initial
scope check. The initial guardrail has its own configured timeout. Provider
retries and internal DRIFT operations are not individual retrieval calls in this
counter; real adapters must impose their own internal request/token budgets.
The application factory uses 90-second call and 300-second total supervisor
limits for the optional real adapter. Its worker separately limits DRIFT to one
follow-up at one depth and applies a 90-second query deadline. These time and
context limits are not an exact dollar-spend cap.

Repeated query/tool pairs stop the loop. Unknown evidence IDs, invalid outputs,
and failures do not count as success. Cancellation propagates to active work.
Every returned evidence item preserves its source ID, assigned evidence ID,
task ID, tool, and parent evidence IDs. `answer_evidence_ids` records which
excerpts support the final answer; the full evidence ledger is retained for lineage.
Production factories attach the MapReduce answer generator. Isolated orchestration
tests may omit it to inspect the retrieval-only evidence contract.
The final output is extractive, not an implemented free-form answer synthesizer.

Use `POST /api/graphrag/query` with the same `{"query":"..."}` body to test the
supervisor directly. If all adapters are disabled or unready, an in-scope query
returns HTTP 503 with `reason_code: "tools_not_connected"` and zero retrieval
calls. A ready Microsoft adapter performs real retrieval. The assistant SSE
path emits a `supervisor` event after
this branch completes. No global supervisor was added to other branches.

An opt-in evaluation uses real model planning/scope checks and synthetic
retrieval fixtures, clearly labeled in its report:

```bash
python -m app.cli.evaluate_graph_supervisor --provider openai \
  --output .local/graphrag-supervisor-openai.json
```

This sends only synthetic questions/evidence plus the approved domain policy,
not `Business_data`. It is **not a live Neo4j/Microsoft retrieval test**. The
fixtures are never registered in application requests. Deterministic pytest
tests additionally cover parallel execution and all control limits.

See [building from Business_data](business-graphrag.md) for the local data
preparation command and the remaining Neo4j work. The separate
`python -m app.cli.evaluate_live_graphrag` runner exercises FastAPI, the real
model, and the real Microsoft index without dependency overrides or mocks;
see [live verification](microsoft-graphrag.md#verification).

DRIFT already performs adaptive follow-up retrieval, so
avoid accidentally multiplying its internal rounds with an unbounded outer
supervisor. See the official
[DRIFT workflow](https://microsoft.github.io/graphrag/query/drift_search/).
