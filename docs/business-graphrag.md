# Building graph retrieval from Business_data

## What the current dataset supports

The current catalog contains 100 products, 15 suppliers, and 15 categories.
These are source-file counts, not records in a connected Neo4j instance.

| Source | Graph representation | Relationship source |
|---|---|---|
| `Products.csv` | Product nodes keyed by `ProductID` | Explicit foreign keys below |
| `Suppliers.csv` | Supplier nodes keyed by `SupplierID` | `Product -SUPPLIED_BY-> Supplier` from `Products.SupplierID` |
| `Categories.csv` | Category nodes keyed by `CategoryID` | `Product -BELONGS_TO-> Category` from `Products.CategoryID` |

Do not ask an LLM to guess these joins. Preserve stable keys separately from
display names: different products can share a name, and names can change.
`Policy` relationships require reviewed authoritative documents. `SupportTopic`
entities are now extracted from the reviewed sample, not inferred solely from
catalog names. Review-derived connections remain unverified customer opinions.

The initial export excludes supplier contacts/addresses, customer and employee
records, orders, reviews linked to customers, prices, stock, and unreviewed
descriptions. Some category descriptions contain apparent random text, so
indexing them unfiltered would degrade retrieval quality. Free-text reviews
also need quality and privacy review before later use.

## Step 1: prepare and inspect a small local seed

```bash
source .venv-langchain/bin/activate
python -m app.cli.prepare_graph_data --data-dir Business_data --limit 20 \
  --output .local/graphrag-catalog-seed-v2
```

Use a **new output directory** each time. Existing directories are never
overwritten. The command makes no network or LLM calls. It validates required
columns, positive IDs, duplicates, and foreign keys across the full dataset
before sampling. Only allowlisted columns reach the output.

```text
.local/graphrag-catalog-seed-v2/
  manifest.json       counts, source hashes, exclusions, review requirement
  nodes.json          catalog nodes with source provenance
  relationships.json explicit catalog edges with source-column provenance
  input/
    product-1.txt     joined product/supplier/category facts
    ...
```

The first 20 products produced 45 distinct nodes, 40 relationships, and 20
documents in the local v1 export. `index_built` remains false. These files are
preparation artifacts, not a completed Microsoft GraphRAG index. Git ignores
the `.local/graphrag-*` outputs. Allowlisting reduces exposure but does not
itself classify a business record as approved for external publication.

### Optional review corpus, adapted from preprocess_data.py

The original script joins reviews with product/category/supplier/customer data
from MySQL and groups the resulting text. The application adaptation is
`app/services/graph_review_data.py`; the original script is unchanged and is
not executed or imported. No MySQL dependency is added.

```bash
python -m app.cli.prepare_graph_data --include-reviews --limit 20 \
  --review-limit 100 --review-group-size 5 --review-max-chars 6000 \
  --output .local/graphrag-review-seed-v3
```

This joins reviews to the selected catalog and groups by **CategoryID plus
ProductID**, preserving the exact product context. Documents retain every
ReviewID in `review_sources.json`, rather than just the first row's metadata.
Content-derived filenames are stable across identical exports, unlike random
UUIDs. Both group size and length limits are enforced. Length is measured in
characters, explicitly **not model tokens**; tokenization is configured later
in the selected GraphRAG runtime. Oversized reviews are skipped for review,
not silently truncated. Fractional ratings are preserved.

Customer identifiers/company/location columns are excluded. Obvious email,
phone, or URL-like review content is quarantined, but this is only a heuristic:
names, addresses, or other sensitive material may remain in free text. Human
privacy and quality review is mandatory before cloud indexing. Deterministic
sampling is not statistically representative of the entire review population.
Reviews are labeled unverified customer opinions, never authoritative policy.
Supplier is not assumed to mean manufacturer.

The source file has 5,000 reviews. Comments on reliability, setup, battery life,
or compatibility can supply documentary evidence for Product/SupportTopic
relationships during Microsoft GraphRAG extraction. Do not create Customer
nodes or derive return eligibility from review opinions. Exact counts and
rating aggregates should remain structured analytics queries.

## Step 2: build the Neo4j path from explicit relationships

The independent Neo4j loader is now implemented: see
[Neo4j query strategies](neo4j-query-strategies.md). It imports catalog and
transaction data atomically, checks counts and refuses to overwrite an existing
dataset. The older catalog export JSON remains a separate preparation artifact.

Ingestion and runtime readers should use different permissions. Runtime tools
should expose only bounded, read-only templates, for example:

1. Resolve a product name to a unique Product key, or ask for clarification.
2. Get its supplier via `SUPPLIED_BY`.
3. Get other products for that supplier, optionally filtered by `BELONGS_TO`.

Keep template selection and parameters separate from LLM text. Add limits,
timeouts, source IDs, index versions, and tenant/corpus permission checks in
the adapter. Do not expose arbitrary Cypher execution to the planner.

## Step 3: build the Microsoft GraphRAG path

Microsoft GraphRAG is a separate indexing/retrieval framework, not a Neo4j
connection mode. Use a separate virtual environment, select and lock a tested
package version, and initialize a dedicated workspace. Do not initialize it
in the application root or replace the application's `.env`.

After installing and configuring the selected version in that environment,
the official CLI workflow is `graphrag init --root <workspace>` followed by
`graphrag index --root <workspace>`. Review the generated settings before
running indexing: model/embedding provider, paths, entity types, prompts,
chunking, concurrency, and costs all matter. This project now uses a pinned
3.1.2 Python API worker instead of requiring the operator to run the vendor CLI
directly. It has built a real index from 41 approved documents, with 99 entities,
110 relationships, and 9 community reports. See the
[runtime setup and results](microsoft-graphrag.md) for reproducible commands.

Start from the prepared `input/*.txt` facts. The index builds text units,
extracted entities/relationships, communities/reports, and embeddings. Review
whether extraction preserves original IDs and links back to source text.
The approved domain schema does not automatically constrain extraction or
guarantee that the index contains the expected relationships.

Twenty catalog documents are useful for a small pipeline exercise but are not
a rich support corpus. Add reviewed public Knowledge Base documents to study
cross-document support themes and policy connections. Preserve document IDs,
versions, and access boundaries. Never fabricate policy links to make a demo
appear complete.

Official references: [setup](https://microsoft.github.io/graphrag/get_started/)
and [indexing dataflow](https://microsoft.github.io/graphrag/index/default_dataflow/).

## Step 4: connect adapters to the implemented supervisor

Implement the `GraphRetrievalTool.retrieve(question, limit=...)` protocol in
`app/services/graph_supervisor.py`. Register only real, ready, server-owned
adapters for `neo4j_relationships`, `ms_local_search`, `ms_global_search`, and
`ms_drift_search`. The factory now registers all three Microsoft adapters when
enabled and verified. The independent Neo4j adapter checks snapshot readiness
before model planning; it is not a placeholder and does not require the Microsoft
index to be enabled.

Return bounded `RetrievalEvidence` items with source IDs, text, and
source-verified entity names/types. Enforce source visibility and resolve
entity ambiguity inside the adapter. DRIFT needs an internal budget in
addition to the outer supervisor budget. A registry entry must not be added
merely because a provider name is configured.

The implemented supervisor will:

1. Require the original question to pass the branch's scope guardrail.
2. Plan one initial retrieval using registered capabilities.
3. Inspect returned evidence before planning further tasks.
4. Revalidate each task's scope, tool suitability, and entity lineage.
5. Execute independent tasks concurrently within fixed limits.
6. Stop on completion, clarification, invalid output, repeats, failure, or budget.
7. Map selected evidence excerpts, reduce them into a cited natural-language answer,
   and retain the full evidence ledger for inspection.

## Step 5: test each layer honestly

- Unit/integration tests use injected outputs to verify mechanics and safety
  controls. They do not measure model quality.
- The real-model guardrail runner tests semantic classification with no retrieval.
- The real-model supervisor runner uses clearly synthetic retrieval fixtures.
  It tests adaptive planning, not graph database or index connectivity.
- `app.cli.evaluate_live_graphrag` exercises actual Microsoft retrieval through
  FastAPI and the real supervisor. Direct worker tests have exercised Local,
  Global, and DRIFT. These are connectivity/provenance smoke tests, not a
  complete answer-quality or security evaluation. Expand held-out cases and
  add tenant permission tests before public deployment.

Do not count an unavailable tool, an empty graph, or a mocked retrieval result
as successful end-to-end GraphRAG. Independent held-out queries are still needed
before treating the semantic gate as production-ready.
