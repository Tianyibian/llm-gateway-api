# Product questions inside GraphRAG

There is no standalone product route. The router sends product discovery, category
browsing, product facts, price, stock, specifications and compatibility questions
to `graph_rag_search`. The branch performs one scope assessment, then plans against
registered tools and available data. Clarification can resolve missing user input
and hand off to this same branch. Policies still use the policy branch.

Business scope, record existence and data freshness are distinct:

- A catalog inquiry does not claim current stock or authorize a purchase.
- Unknown categories pass scope; an executed query can return no matching records
  in the imported snapshot. This is not proof of universal nonexistence.
- Product/category browsing uses Text-to-Cypher with database-grounded entity
  resolution, not the `category_products` template. See [entity resolution](entity-resolution.md).
  Regular English plurals and case variants can map to unique database names/IDs;
  ambiguous candidates request clarification. No product-specific alias was added.
  This is not a complete multilingual, irregular-plural or synonym resolver.
- Cypher record queries return one target entity's identity per task; the planner
  must not expand a product-list request into unsupported multi-entity columns.
  Query-alignment validation remains enabled and rejects incomplete queries.
  The planner receives the same schema used by the Cypher compiler. Product-kind
  discovery filters category membership rather than substituting product-name
  matching. Map/Reduce also receives the retrieval question so empty rows retain
  the context and snapshot limitations of the lookup that produced them.
- Current adapters expose identities, categories, suppliers and historical sales,
  not authoritative current prices, live stock, specifications or compatibility.
  For those missing capabilities, the planner returns structured
  `data_unavailable` with an allowlisted `missing_product_data` list. The server
  returns `product_data_unavailable` and a fixed explanation, without claiming a
  product is unavailable/out of stock. A composite request needing missing data
  is reported as not fully answerable rather than silently dropping requirements.
  This outcome currently uses status `unavailable` (HTTP 503 on the standalone
  graph endpoint); the assistant SSE still displays the explanation.
- Current-price requests must not use historical revenue or transaction prices.
  Review opinions must not be substituted for authoritative specifications.

No database migration, ingestion, inventory integration or write permission is
part of this change. Future data adapters must establish source, field coverage
and freshness before these capability declarations are updated.

## Checks

```bash
.venv-langchain/bin/python -m pytest tests/test_graph_product_capabilities.py tests/test_product_route_removed.py tests/test_branch_scope_counts.py -q
.venv-langchain/bin/python -m app.cli.evaluate_graph_products
```

The first command uses deterministic fake models. The second is opt-in and uses
real configured models and public Neo4j snapshot reads, without saving conversations.
It tests multiple categories, an empty-match case, missing fields and a prohibited
write. Provider calls incur charges; these smoke tests do not prove universal
classification correctness or alias coverage.

## Earlier verification snapshot, before entity resolution (2026-09-26)

- Deterministic backend suite: 537 passed (one third-party deprecation warning).
- Frontend graph-inspector suite: 17 passed.
- Latest live eight-case run: 7 passed, 1 failed. Product speakers, cameras,
  categories, explicit price/stock/specification limitations, and prohibited-write
  blocking passed. Every case made exactly one branch scope assessment.
- Remaining failure: the computer lookup executed successfully with no matches,
  but the planner continued investigating and proposed an entity without valid
  evidence lineage. The provenance check rejected that follow-up. The assistant
  did not deliver the intended final no-matches explanation.
- Earlier live iterations also exposed query-generation/alignment failures;
  one passing run is not evidence of universal prompt reliability. No validation
  checks were disabled to obtain these results. Empty-result planning remains an
  open regression, so the live suite must not be reported as fully passing.

The live evaluator calls the real classifier and GraphRAG supervisor directly;
it is not an HTTP/SSE end-to-end test and does not save conversation messages.
