# Hierarchical GraphRAG agents

The production branch separates business delegation from tool selection:

1. The router selects GraphRAG. A scope-only root guard checks the entire question
   against the business domain and schema. Supported composite questions may need
   both Neo4j and Microsoft GraphRAG. Root approval grants no tool capability.
2. `HierarchicalGraphSupervisor` produces a `DelegationPlan`: agent role,
   self-contained subquestion and parent evidence IDs. Its schema has no tool
   field. Independent goals run in parallel; dependent goals wait for actual
   prerequisite evidence and a later planning round.
3. Each `SpecialistAgent` has its own prompt, compiled LangGraph planner loop,
   invocation-local state and tool allowlist. It inspects results and can plan
   another bounded retrieval. Agents do not share mutable evidence state.
4. The supervisor remaps local evidence IDs into a request-wide ledger, checks
   lineage and preserves per-agent outcomes. A failed specialist cannot silently
   become a fully completed request.
5. The shared MapReduce generator selects original evidence passages, synthesizes
   the final answer and validates source IDs before SSE delivery. Map calls process
   evidence; they are not additional business agents.

| Specialist | Responsibility | Allowed tools |
| --- | --- | --- |
| `catalog_agent` | Product/supplier/category relationships and catalog counts | `neo4j_relationships` |
| `sales_agent` | Revenue, units, transaction aggregates and trends | `neo4j_relationships` |
| `reviews_agent` | Reviews, support themes and document synthesis | Microsoft Local, Global and DRIFT |

Neo4j still selects template-first or constrained Text-to-Cypher internally.
Schema/parameter checks, semantic review, EXPLAIN, timeouts and read-only database
controls remain. Each tool task undergoes strict backend-aware guard validation.
Scope, decomposition and semantic completeness remain model judgments, not proofs.

Production limits: three supervisor rounds, three concurrent specialists and six
total retrieval calls. Each assignment reserves two calls. Each specialist has
two rounds, two calls and one concurrent retrieval. Parent retrieval has a
480-second deadline; specialists have 240 seconds. Answer generation retains its
separate configured budget (120 seconds by default). Cancellation propagates to
children. Independent Microsoft requests may serialize inside the shared client.

The API retains `trace` for tool calls and adds `agent_runs` for business
assignments, outcomes and evidence IDs. Evidence includes `agent_role` and
`agent_run_id`. The frontend shows live `agent` started/completed events, then
MapReduce progress. These are application-level agent graphs, not necessarily
different foundation models or separate servers.

## Verification

Deterministic tests explicitly use fake providers/tools for parallelism, state
isolation, dependencies, permissions, lineage, partial failure and cancellation:

```bash
python -m pytest tests/test_graph_agents.py tests/test_graphrag_guardrail.py
node --test tests/frontend/*.test.cjs
```

With the app, Neo4j and Microsoft index running, the opt-in live test spends model
tokens and persists a dedicated test conversation:

```bash
python -m app.cli.evaluate_graph_agents
```

It asks for Eufy Smart Speaker Essential's supplier, 2025 discounted sales revenue
and sampled review themes. Assertions check three specialists, their actual
tools, budgets, final synthesis, SSE and persistence. This is an integration
check, not an accuracy benchmark or a guarantee of identical plans for all wording.

`GraphRAGSupervisor` in `graph_supervisor.py` is retained as the bounded tool-level
planner inside each specialist and in focused tests. Production factories build
`HierarchicalGraphSupervisor` from `graph_agents.py` above it. Snowflake remains a
separate optional analytics branch, not an implicit fallback tool for specialists.
