# Local and Global Microsoft GraphRAG search

Both implementations use the official pinned Microsoft GraphRAG source checkout.
No new index or duplicate retrieval implementation is needed.

| | Local | Global |
| --- | --- | --- |
| Starting point | Semantically related entities | Community reports at the selected hierarchy level |
| Context | Entities, relationships, associated text units and community reports | Batches of community reports |
| Answer generation | Fit relevant evidence into a context window, then generate | Map batches into scored points, filter/rank, then Reduce |
| Good question | What connectivity concerns are described for a named product? | What recurring support themes appear across the indexed reviews? |
| Main limitation | Retrieved neighborhood is not exhaustive corpus coverage | Reports compress details and may propagate extraction errors |

Local does not mean offline, and Global does not mean Internet search. Both use
the same approved local Parquet/LanceDB index and the configured model provider.
Our index is a sample, not all source reviews; neither mode proves exact counts,
percentages, representative prevalence, or real-time inventory. Use Neo4j or the
optional relational analytics backend for supported transaction aggregates.

## Selection contract

`graphrag_search_mode` accepts `local` (default) or `global` on:

- `POST /api/assistant` in JSON or multipart form data.
- `POST /api/graphrag/query` in JSON.

The UI selector defaults to Local on page load. It controls review retrieval only:
the business router still chooses whether the request needs GraphRAG at all.
Picking Global is not a request to bypass routing or the scope guardrail.
Natural-language instructions do not override the selected mode. Select Global
in the UI or request field to compare it with Local.

```json
{
  "user_id": "search-demo",
  "query": "Summarize recurring support themes in the sampled smart-home reviews.",
  "graphrag_search_mode": "global"
}
```

Omit `user_id` for `/api/graphrag/query`, which does not persist conversation turns.
Start separate conversations for controlled comparisons so previous answers do
not influence the question. Actual executed tools appear in the trace as
`ms_local_search` or `ms_global_search`; a requested mode is not evidence that
the tool actually ran. A failed/empty mode never silently falls back to another.

For a no-history comparison, import
`postman/GraphRAG_Search_Modes.postman_collection.json` and run both requests.
They ask the same question through real providers and assert actual tool traces.
Allow a request timeout of at least 600 seconds for the bounded supervisor workflow.

## Enforcement and implementation

The direct planner decomposes catalog, sales, and review goals into tool tasks.
Before running it, `with_search_mode()` creates a request-local engine with
only the selected Microsoft tool registered. Both explicit Cypher tools remain available.
The original engines are never mutated, so concurrent requests can select
different modes. Tool allowlisting prevents model plans from changing modes.

Both Local and Global are eligible for approved document questions. Search
suitability is separate from authorization: the domain, schema, backend, secret,
write and transaction checks remain enforced. DRIFT remains in the low-level
adapter/CLI for experiments, but cannot be selected through this two-mode API.

Read the code in this order:

1. `app/models/schemas.py`: assistant request and default.
2. `app/services/graphrag_service.py`: scope gate and request-mode handoff.
3. `app/services/graph_supervisor.py`: direct task planning and per-request tool restriction.
4. `app/services/microsoft_graphrag.py`: isolated-process adapter (default Local).
5. `graphrag_runtime/worker.py`: calls `api.local_search()` or `api.global_search()`.
6. `third_party/graphrag/packages/graphrag/graphrag/query/structured_search/`:
   `local_search/` builds entity-grounded context; `global_search/` performs
   community-report MapReduce.

Microsoft Global Search's internal MapReduce is distinct from our outer
MapReduce answer generator, which combines evidence from business subagents.

Official references: [Local](https://microsoft.github.io/graphrag/query/local_search/)
and [Global](https://microsoft.github.io/graphrag/query/global_search/).
