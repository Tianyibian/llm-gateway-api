# Validated Cypher retrieval subgraph

`TextToCypherService` owns a compiled LangGraph (`validated-cypher-retrieval`).
The outer task planner still sees two retrieval tools, not individual internal
steps that it can skip or rearrange.

## Paths

- Dynamic: `prepare_query -> generate_cypher -> validate_cypher -> execute_cypher`.
- Predefined: `prepare_query -> compile_template -> validate_cypher -> execute_cypher`.
- Unsupported selection or generation stops before validation/execution.
- Rejected validation stops before execution. No automatic repair/retry is added.

`prepare_query` checks the versioned Neo4j snapshot. Dynamic mode bypasses the
template selector. Template mode uses a structured `CypherSelection` to select an
exact supported template; it does not fall back to dynamic generation.

`generate_cypher` asks for a Pydantic `CypherPlan`, then uses the deterministic
server compiler. It does not accept executable query text from the model.

`validate_cypher` re-derives the candidate from the allowlisted plan or template,
checks exact agreement, asks the semantic reviewer to compare it with the question,
and checks Neo4j's EXPLAIN result. Only query type `r` is accepted; `w`, `rw`, `s`
and unknown types are rejected. Estimated-row budgets and unknown-schema warnings
are also checked. Approval is tied to a fingerprint of the candidate and parameters.
That fingerprint is an internal integrity check, not a public authorization token.

`execute_cypher` requires an unchanged validated candidate and retrieves bounded
rows. The executor independently repeats EXPLAIN immediately before a data query,
so direct calls to `Neo4jExecutor.run(compiled)` cannot bypass the read-only plan
gate. It snapshots parameters before awaiting the driver. Results still use the
existing `RetrievalEvidence` format for the outer planner and MapReduce.

Every public `query()` invocation creates fresh state, accepts only a question and
strategy, and has the existing overall timeout. It accepts no caller approval,
compiled query or checkpoint. Concurrent calls do not share candidate state.

## How predefined Cypher works

1. `CypherSelection` supplies a restricted schema: template enum, name, year, limit.
   Extra fields are forbidden; a model cannot submit its own `cypher` field.
2. The model chooses a template and grounded parameter values, not query syntax.
3. `TEMPLATES` in `cypher_templates.py` maps each allowed name to reviewed Cypher.
4. `compile_template()` validates combinations, checks names/years against the
   question, and binds values as driver parameters. Values are not interpolated.
5. The same validation and execution nodes used by dynamic queries are mandatory.

JSON Schema is a format/capability contract, not sufficient write protection by
itself. Semantic review can be wrong; database privileges/configuration remain
important defense in depth.

## Database enforcement

On 2026-09-26, read-only metadata checks against the local development database
reported Neo4j Community 5.26.30, `server.databases.default_to_read_only=true`, and
`neo4j` database access `read-only`. Current-user roles were null. This is a
database-wide read-only configuration, not a dedicated per-user reader role.
No writes or permission changes were used to check this. Future configuration
changes can invalidate these observations.

The driver `READ_ACCESS` setting is routing guidance, not a write-prevention ACL.
Use database-enforced read-only application credentials on a deployment that
supports them, separate from ingestion/admin credentials. The local Community
recipe instead runs the business database read-only after ingestion. The public
query API exposes no database-administration or procedure-execution capability.

## Tests

```bash
.venv-langchain/bin/python -m pytest tests/test_cypher_subgraph.py tests/test_cypher_checks.py tests/test_neo4j_queries.py -q
```

Tests use explicit fake models/driver summaries and cover actual node order,
semantic rejection, non-read plans, candidate tampering, missing approval,
invalid write actions, unknown templates, direct-executor bypass attempts and
concurrent request isolation. They do not attempt real CREATE/DELETE operations.

See [Neo4j query strategies](neo4j-query-strategies.md) and
[direct task planning](graph-task-planner.md) for the outer workflow.
