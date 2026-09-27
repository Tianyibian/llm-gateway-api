# Documentation

## Start here

- [Project map and reading order](project-map.md): code responsibilities and entry points.
- [Main README](../README.md): features, configuration, API usage and tests.

## Architecture and retrieval

- [Product queries in GraphRAG](graph-product-queries.md): catalog retrieval, absent data and snapshot limitations without a separate product route.
- [Entity resolution](entity-resolution.md): database-grounded name matching and ambiguity clarification inside Text-to-Cypher.
- [Branch scope policy](branch-guardrails.md): at most one scope guardrail per business branch, separate from execution checks.
- [Policy ensemble](policy-ensemble.md): policy scope guardrail, vector/BM25 retrieval and RRF provenance.
- [Policy metadata filters](policy-metadata-filters.md): request-level narrowing of fused hybrid candidates.
- [Policy reranker](policy-reranker.md): hybrid candidates, metadata filtering, local Cross-Encoder and top context selection.
- [Cypher subgraph](cypher-subgraph.md): generation, shared validation, execution and read-only enforcement.

- [Direct task planner](graph-task-planner.md): active GraphRAG decomposition, tools, dependencies and evidence synthesis.
- [Hierarchical agents](hierarchical-agents.md): retained legacy architecture, not the active API path.
- [GraphRAG guardrail and workflow](graphrag.md): scope checks and bounded planning.
- [Business data preparation](business-graphrag.md): source CSVs and reviewed text preparation.
- [Microsoft GraphRAG](microsoft-graphrag.md): isolated runtime, indexing, queries and Parquet viewing.
- [Merged review indexing](merged-review-indexing.md): source reconciliation, token-bounded chunks and versioned index builds.
- [Neo4j query strategies](neo4j-query-strategies.md): predefined Cypher and constrained Text2Cypher.
- [Snowflake authentication](snowflake-authentication.md): optional warehouse access.

## Validation

- [Chat testing](chat-testing.md): browser-based execution inspection.
- [Postman guide](../postman/README.md): real-provider API tests, distinct from fake-model unit tests.
- [Exploration notebooks](../notebooks/README.md): environment selection and reference material.

Examples that call external models or build indexes can incur charges. Local
`.env` files, keys, generated indexes and diagnostic outputs are not documentation
assets and should not be committed.
