# Neo4j query strategies

GraphRAG supports **template-first Cypher** and **constrained Text-to-Cypher**.
Microsoft GraphRAG remains a separate document backend, not a source of exact
transaction totals.

## Execution flow

Router → graph scope guardrail → business supervisor → catalog/sales specialist
→ Neo4j tool → strategy selector
→ template OR structured query compiler → semantic review → EXPLAIN → Neo4j
→ evidence → parallel Map → Reduce → citation validation → final answer.

## Query checks and final answer

Both query strategies share fail-closed checks:

1. The deterministic compiler validates the schema, grounded parameters, bounded
   graph shape and result limit. Models never submit arbitrary executable Cypher.
2. A separate structured model call reviews the compiled query against the original
   worker question, rejecting omitted filters, wrong metrics or missing constraints.
3. Neo4j `EXPLAIN` checks the actual syntax/schema without executing the data query.
   Only read-only plans are accepted. Unknown schema notifications and estimated
   intermediate row counts above `NEO4J_MAX_ESTIMATED_ROWS` stop execution.
4. The data query retains its database timeout, result cap and read-only database
   configuration. Estimated cost and model review are safeguards, not proofs of
   correctness or substitutes for database access control.

`GraphAnswerGenerator` maps up to 12 selected evidence excerpts, with at most three
concurrent model calls. The mapper selects numbered source passages; the server
validates those IDs and extracts exact original text. The model cannot rewrite
the mapped quotes. Unknown IDs or model-supplied replacement quotes are rejected.
Reduce receives only those validated excerpts plus the original user question.
Every answer paragraph must cite known mapped evidence IDs; the server renders
the citation markers. Partial/truncated evidence adds an explicit limitation note.
Unknown citations, fabricated quotes, provider failures and timeouts return a safe
failure response rather than an unvalidated answer. Raw evidence and query checks
remain visible under **Execution details**, not as the main chat response.

The frontend receives `answer_generation` progress events for mapping, reduction
and citation validation. Final text is buffered until validation and then delivered
as segmented SSE deltas; this is **not token-by-token model streaming**. The final
answer, not intermediate map outputs, is saved in the conversation. Reopening a
conversation restores its messages; execution diagnostics are live-turn metadata.

`GRAPH_ANSWER_TIMEOUT_SECONDS` defaults to 120 seconds after the retrieval budget;
individual map/reduce calls are capped at 45 seconds. Source-ID and exact-quote
checks do not prove every generated statement is entailed by a source. Numerical
or high-stakes answers still need independent comparisons and human review.

Tests: `python -m pytest tests/test_cypher_checks.py tests/test_graph_answer.py`
uses explicit model/executor mocks for deterministic failure coverage. Real backend
validation is separate: `python -m app.cli.evaluate_neo4j --live-model`.
For live chat/SSE/persistence and final monthly-revenue comparisons, start the app
and run `python -m app.cli.evaluate_graph_answers --include-microsoft`. This uses
real model calls, requires both indexes, and saves isolated test conversations.

A template is selected only when it covers the whole question. A template miss
selects the dynamic path; invalid arguments, timeouts and database failures do
not trigger another strategy or backend. Selection uses a model call; templates
avoid the additional dynamic-plan generation call, not all model calls.

| Template | Semantics |
| --- | --- |
| `product_supplier` | Supplier of one exactly named product |
| `category_products` | Products in one exactly named category |
| `shared_supplier_products` | Other products sharing a product's supplier |
| `revenue_by_product` | Net-revenue product ranking, all data or one year |
| `revenue_by_supplier` | Net-revenue supplier ranking, all data or one year |
| `monthly_sales` | Chronological monthly revenue and units, all data or one year |

These Cypher strings are fixed. Names, years, dataset and limits are bound
parameters. Names and years must be grounded in the original question.

Dynamic queries are **not unrestricted model-authored Cypher**. The model returns
a typed `CypherPlan`; a deterministic compiler validates and produces Cypher.
It supports connected traversals of up to six nodes, public name filters, absolute
date ranges, counts, revenue and units. Unknown labels/properties, raw commands,
disconnected patterns and writes are rejected. Output is limited to 20 rows with
explicit truncation metadata. Specify a calendar year for a complete twelve-month
series. Semantic selection is probabilistic: validators constrain capabilities
but do not prove that every detail of a question was understood.

## Choosing a backend

- Standard rankings, monthly totals and relationship analytics default to Neo4j.
- Snowflake is optional: standard reports use it only when both
  `SNOWFLAKE_ENABLED=true` and `ANALYTICS_BACKEND=snowflake`. Otherwise the
  router sends analytics to the guarded graph branch, even if the model suggests
  the disabled Snowflake route. Database errors never trigger a backend switch.
- Relationship-defined cohorts, such as sales of other products sharing a named
  product's supplier, use Neo4j.
- Explicit supported Neo4j comparisons can use the graph branch for reporting.
  This does not imply that a graph is inherently faster or more accurate.
- Review themes and cross-document explanations use Microsoft GraphRAG.

## Data and numerical consistency

The loader imports allowlisted fields from Products, Suppliers, Categories,
Orders and `_Order_Details`; customer/employee/contact fields are excluded.
The current source prepares 4,117 nodes and 6,174 edges. Revenue uses
`UnitPrice * Quantity * (1 - Discount)`, validated with Decimal and stored as
integer millionths before final cent rounding. The atomic import checks keys,
relationships and counts before publishing a versioned snapshot marker. Existing
datasets are never overwritten; use a new dataset name for another snapshot.
Uniqueness constraints protect concurrent imports. This is not a live ERP feed.

## Local setup

```bash
.venv-langchain/bin/python -m pip install -r requirements-neo4j.txt
.venv-langchain/bin/python -m app.cli.prepare_neo4j
NEO4J_READ_ONLY=false docker compose --env-file .env.neo4j -f compose.neo4j.yaml up -d
# Wait for Neo4j to accept connections; validate without writes first:
.venv-langchain/bin/python -m app.cli.load_neo4j
.venv-langchain/bin/python -m app.cli.load_neo4j --execute
# Switch to database-enforced read-only mode after import:
docker compose --env-file .env.neo4j -f compose.neo4j.yaml up -d
```

The generator creates `.env.neo4j` with a random password and owner-only
permissions; Git ignores it. It never replaces an existing file. Settings load
`.env`, then `.env.neo4j`; process environment variables override both. Do not
include credentials in screenshots or share either environment file.

The Community development recipe uses one local administrator identity and a
database-wide read-only setting after import. It is **not production RBAC**.
Production requires a database-enforced read-only application principal, separate
loader identity, TLS, secret management, authorization and audit logs. The driver
`READ_ACCESS` flag is routing guidance, not a security boundary. See the official
[driver guidance](https://neo4j.com/docs/python-manual/current/query-simple/) and
[read-only configuration](https://neo4j.com/docs/operations-manual/current/configuration/configuration-settings/).

Restart the application after configuration changes. Both the existing
`POST /api/graphrag/query` endpoint and assistant frontend use this worker.
`GET /api/graphrag/neo4j/status` checks snapshot readiness without model tokens.
`/api/graphrag/status` continues to describe Microsoft GraphRAG only.

Execution details show the actual `query_mode`, `template_id`, Cypher, bound
parameters, row count and truncation. These are diagnostics, not hidden reasoning.

## Verification

```bash
.venv-langchain/bin/python -m pytest tests -q
node --test tests/frontend/*.test.cjs
.venv-langchain/bin/python -m app.cli.evaluate_neo4j
# Real configured model, routing, guardrails and supervisor (uses model tokens):
.venv-langchain/bin/python -m app.cli.evaluate_neo4j --live-model
```

Use `/health` to inspect the effective `analytics_backend`. Backend selection is
server configuration, not an unrestricted per-message Snowflake override.

Unit tests use fake models/executors to test deterministic boundaries. The
evaluation command uses real Neo4j and an independent CSV baseline. CSV comparison
does not prove Snowflake ingestion or Snowflake query success.

Example requests, with a JSON body such as `{"query": "..."}`:

- `Using Neo4j, show monthly sales for 2025` → `monthly_sales` template.
- `Using Neo4j, count products by supplier` → dynamic structured plan.
- `Delete all products` → guardrail rejection, no retrieval.

Use an exact catalog name for product-specific requests. Unknown names produce
no matches, not invented entities. Do not expose this prototype publicly without
authentication and data authorization.
