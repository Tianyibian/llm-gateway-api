# Indexing merged reviews

`Business_data/merged_reviews.csv` is a legacy export with `id`, `text`,
`token_count`, and `CategoryName` columns. One CSV row contains several reviews
separated by `<ROW_SEP>`. Its token counts are heuristic, not model token counts.
The reference script also omits its group-size check in the category branch.
Do not use those values as hard indexing limits.

## Preparation

Implementation: `app/services/merged_review_data.py`, invoked by
`app/cli/prepare_merged_reviews.py` in the isolated GraphRAG environment.

1. Split each merged row at review boundaries. Reject malformed records.
2. Match each record against `Reviews.csv` using its customer key, rating, date,
   and complete review text. This is a local join, not an LLM inference.
   Require exactly one match to recover the original ReviewID and ProductID.
3. Validate the product name, supplier and category against the catalog tables.
   Whitespace/case normalization is allowed, fuzzy matching is not.
4. Exclude customer IDs, company names, locations and prices from exported text.
   Quarantine obvious contact-like free text and oversized individual reviews.
   This heuristic does not guarantee complete anonymization.
5. Group by category **and product ID**, with at most five complete reviews and
   1,000 tokens per document. Count the final text, metadata and separators using
   GraphRAG's `o200k_base` tokenizer. Never silently truncate review content.
6. Record original review IDs, merged row IDs, block positions and SHA-256 hashes
   in `review_sources.json` and `manifest.json`. Never invent a missing ReviewID.

Two catalog records share the name `Ring Smart Speaker Plus` (Product:77 and
Product:87). Original review foreign keys disambiguate them. The extraction
prompt requests Product IDs in entity titles so the model does not collapse them
by display name. This is still model-guided extraction, not a database constraint.

All 5,000 source reviews in this snapshot match uniquely. Full preparation
produces 1,042 documents covering 100 product IDs and 15 categories, with no
skipped reviews. Their text contains 617,028 tokens (largest document: 728).
This is **input text volume, not total billable API usage or a price estimate**.
Extraction prompts, description summaries, reports and retries add usage.

The default pilot selects 100 reviews using deterministic category/product
round-robin sampling. It covers every product in this snapshot, but is not
statistically representative. Full mode uses every eligible review. Exact
duplicate source blocks are dropped; identical opinions from different original
records are retained. Preparation outputs are local and are not indexes.

```bash
# Local preprocessing only. The first tokenizer use may download vocabulary.
.venv-graphrag/bin/python -m app.cli.prepare_merged_reviews \
  --limit 100 --output .local/graphrag-merged-seed-pilot-NEW

.venv-graphrag/bin/python -m app.cli.prepare_merged_reviews \
  --all --output .local/graphrag-merged-seed-full-NEW
```

## Build and validate a new workspace

Review the exported texts and approve model API costs before indexing. Never
overwrite the active workspace. Preparation retains conservative default limits
of 64 documents / 200 KB. Raising them requires explicit command-line values;
the hard ceiling remains 2,000 documents / 10 MB.

```bash
# Pilot: 100 documents, approximately 48 KB.
.venv-graphrag/bin/python graphrag_runtime/worker.py prepare \
  --source .local/graphrag-merged-seed-pilot-NEW \
  --root .local/graphrag-ms-merged-pilot-NEW \
  --max-documents 100 --approve-reviewed-inputs

# Paid model calls start here.
.venv-graphrag/bin/python graphrag_runtime/worker.py index \
  --root .local/graphrag-ms-merged-pilot-NEW
.venv-graphrag/bin/python graphrag_runtime/worker.py verify \
  --root .local/graphrag-ms-merged-pilot-NEW
.venv-graphrag/bin/python graphrag_runtime/worker.py query \
  --root .local/graphrag-ms-merged-pilot-NEW --method local \
  --query 'What do the sampled Eufy Smart Speaker Essential reviews report?' \
  --output .local/graphrag-merged-local-check.json
```

For an explicitly approved full build, use the full seed, a different root,
`--max-documents 1100 --max-input-bytes 3000000` on `prepare`, and
`--index-timeout-seconds 7200` on `index`. These are size/time bounds, not hard
spend caps. Failed builds retain their local cache. Completed builds cannot be
overwritten by the worker.

The official GraphRAG chunker still runs. Because every prepared document fits
within its configured 1,000-token budget, it should yield one text unit per
document. Verify this against the actual Parquet output rather than assuming it.
The worker copies and fingerprints the preparation manifest and review-source
mapping so text-unit document titles can be traced back to CSV records.

Before activation, verify artifact/vector counts, text-unit coverage, the two
same-name product entities, and real Local and Global query results. Only then
update the local `MICROSOFT_GRAPHRAG_ROOT` and restart the application. A pilot
should not silently replace a broader full index. Keep the previous root for
rollback. Indexes, caches and query reports stay under ignored `.local/graphrag-*`.

Reviews remain unverified opinions, including contradictory or implausible
features in the synthetic data. They are not authoritative policy, specifications,
current inventory, or exact revenue data. Neo4j and policy retrieval are unchanged.

## Local validation snapshot

The 2026-09-26 pilot build at `.local/graphrag-ms-merged-pilot-20260926`
completed real Microsoft GraphRAG standard indexing in approximately 383 seconds:
100 documents, 100 text units, 224 entities, 374 relationships and 55 community
reports. The three vector tables contain 224 entity, 100 text-unit and 55 report
vectors. All 100 product IDs appear in entity titles, including separate records
for Product:77 and Product:87. These are observations, not future-run guarantees.

The full preparation at `.local/graphrag-merged-seed-full-20260926` passed checks
for all 5,000 unique original review IDs, complete source mappings, group-size
and actual-token bounds, and exclusion of customer/price headers. The backend
regression suite passed 613 tests. Most regression tests are deterministic and
are separate from the real index/query checks.

Real Local and Global queries both returned five evidence records with valid
reference IDs resolving to the new Parquet artifacts. Local Search retrieved
Review:4581 for Eufy Smart Speaker Essential and reported the connection concern
with a single-review caveat. Global Search summarized support, setup, connectivity
and related concerns across the sample. Citation-ID validity is not a full
claim-entailment or retrieval-quality evaluation. Raw results remain in ignored
`.local/graphrag-merged-local-check-20260926.json` and
`.local/graphrag-merged-global-check-20260926.json`.

## Full-index activation

After explicit approval, all 5,000 reviews were indexed at
`.local/graphrag-ms-merged-full-20260926`. This is now the active local application
root. The original `.local/graphrag-ms-business-v2` and the pilot are retained for
rollback. Only the ignored `.env` root setting was changed, followed by a local
server restart. Neo4j, policy retrieval and conversation storage were not rebuilt.

Verified full-index counts:

| Artifact | Rows |
| --- | ---: |
| Documents / text units | 1,042 each |
| Entities | 409 |
| Relationships | 3,884 |
| Communities / reports | 103 each |
| Entity / text-unit / report vectors | 409 / 1,042 / 103 |

All 5,000 unique review IDs remain traceable. Every input document maps to one
unchanged text unit. All relationship endpoints resolve to existing entities,
entity/relationship IDs are unique, and product IDs cover exactly 1–100.
The two same-name Ring products remain separate.

This workspace uses eight concurrent requests. The initial two-request worker
was interrupted and resumed with its completed cache retained. The resumed build
took approximately 1,265 seconds; this is not the total wall-clock or billed
usage. The tracked template still defaults to two concurrent requests. Completed
extraction, description-summary and report cache entries all have `stop` finish
reasons, with no recorded retries. This does not establish semantic completeness.

Real query checks:

- Global Search returned five report evidence records with valid citation IDs.
- Local Search retrieved four source records containing the requested Eufy
  reviews, but its native generated answer confused embedded Review IDs with
  Sources-table IDs. The existing citation validator rejected that answer and
  returned verified excerpts. **Native Local answer citation validation did not
  pass this test.** No validation rule was relaxed.
- The application does not forward that native answer. Its planner consumes
  source excerpts and its outer MapReduce generates a separately validated
  answer. The real FastAPI/supervisor test passed 7/7 checks against the full
  index, with `map_reduce_validated` and citations `[E1] [E2] [E3]`.
- After activation, the running server's readiness endpoint reports the full
  counts above, and the frontend and health endpoints return HTTP 200. A real
  HTTP query to that restarted server also passed all five checks: HTTP 200,
  complete status, actual Local Search execution, validated MapReduce and known
  answer citations. These diagnostic queries did not persist conversations.

The full-index query and end-to-end reports are in ignored
`.local/graphrag-merged-full-local-check-20260926.json`,
`.local/graphrag-merged-full-global-check-20260926.json`, and
`.local/graphrag-merged-full-api-check-20260926.json`. Reference-ID checks do not
prove claim-level entailment. Keep the native Local citation limitation visible
when evaluating future query or prompting changes.
