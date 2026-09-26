# Direct GraphRAG task planner

The active application factory builds `GraphRAGSupervisor`, not the legacy
`HierarchicalGraphSupervisor`. Catalog, sales and review requests share one
guarded GraphRAG branch. Fixed business-role agents are no longer required.
The older classes remain for compatibility and regression tests only.

## Execution

1. The router selects `graph_rag_search`. With the default Neo4j analytics
   backend, ordinary revenue rankings and monthly totals also enter this branch.
   Snowflake remains a separately configured optional analytics route; policy,
   uploads, images, clarification and general conversation retain their routes.
2. The root guard evaluates the entire question against scope and approved
   schema. Rejection stops before planning. Approval passes the original question,
   not rewritten guardrail diagnostics, to the planner.
3. The planner returns a validated `GraphPlan`: focused task questions, explicit
   tools and parent evidence IDs. Up to three independent tasks may run in the
   first round. Dependent tasks wait for actual evidence in a later round.
4. The executor validates the whole batch before starting any tool: registered
   tool, strict per-task backend guard, parent IDs, grounded entity names,
   duplicates and budgets. It then dispatches directly to tool adapters.
5. Retrieved evidence and execution traces return to the planner. It may retrieve
   another bounded step, finish with supporting evidence, or request clarification.
   Selected follow-up evidence brings its declared prerequisite sources into
   synthesis as well. Finishing cannot silently omit an independent evidenced task.
6. `GraphAnswerGenerator` maps sources to exact original excerpts, then reduces
   those excerpts into a cited answer. Partial retrieval remains partial, and
   missing evidence is not filled with model guesses.

## Tools

| Registered name | Execution |
| --- | --- |
| `predefined_cypher` | Fixed reviewed template plus grounded parameters. An unmatched question fails closed; no silent dynamic fallback. |
| `text_to_cypher` | Model-generated constrained `CypherPlan`, compiled by server code. It bypasses template selection but retains semantic review and read-only EXPLAIN checks. |
| `ms_local_search` | Official Microsoft entity-grounded search; default request mode. |
| `ms_global_search` | Official community-report MapReduce; explicit `graphrag_search_mode: "global"`. |

Only one Microsoft mode is registered for each public request. Both Cypher tools
remain available. `neo4j_relationships` is a legacy enum/automatic strategy, not
registered by the active factory. Unknown tools cannot execute. To add a tool,
define its contract, scope/capability mapping, adapter, factory registration and
tests; the planner cannot invent nodes.

The LangGraph has planning and validated-dispatch nodes; the tools are Python
adapters invoked by dispatch, not a newly compiled LangGraph node per subtask.
Task IDs such as `R1T1` are assigned by the server. Dependencies reference prior
evidence IDs (for example `E1`), never guessed results from the same batch.

## Limits and visibility

Three rounds, six tool calls and three concurrent retrievals; 90-second per-call
and 480-second orchestration limits in the factory. Answer generation has its own
configured budget. Microsoft calls may serialize within its shared client.
The frontend displays `graph_task` started/completed events, tool names and
dependencies, followed by MapReduce progress and the final `trace`.
`agent_runs` is empty in this direct workflow and retained for response compatibility.
The scope decision is streamed before task execution begins, rather than waiting
for the whole nested graph to complete. Final answer text is still buffered until
citation validation, then delivered in SSE segments; it is not live model-token streaming.

Scope, decomposition and semantic coverage still involve model judgments.
Citation validation verifies source IDs, not universal factual entailment.
The Microsoft index is sampled; neither mode supports exact population statistics.

## Verify

Deterministic tests deliberately use fake providers to exercise failure paths:

```bash
.venv-langchain/bin/python -m pytest tests/test_graph_task_planner.py tests/test_graph_supervisor.py tests/test_graph_search_modes.py tests/test_cypher_checks.py -q
node --test tests/frontend/*.test.cjs
```

Live integration (real providers, API cost, no conversation persistence):

```bash
.venv-langchain/bin/python -m app.cli.evaluate_graph_tasks
```

The evaluation checks real traces rather than assuming a question ran every tool.
Different model plans can be valid; a failed check is reported, not relabeled a pass.

Example composite query:

> For Eufy Smart Speaker Essential, identify its supplier, report its total discounted revenue and units sold in 2025, and summarize its sampled review support themes.

Expected capabilities: supplier template, filtered-sales Text-to-Cypher and
Microsoft Local Search. The answer must distinguish transaction facts from sampled
review opinions, and must not infer causation between them.
