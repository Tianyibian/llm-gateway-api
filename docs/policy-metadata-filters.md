# Policy metadata filters

`POST /api/assistant` accepts optional explicit `policy_filters`. These narrow the
public Knowledge Base used by the policy branch; they neither force that route
nor bypass its scope guardrail. Other branches do not use these filters.

```json
{
  "user_id": "policy-test",
  "query": "What is the return policy?",
  "policy_filters": {
    "categories": ["Policy"],
    "source_types": ["pdf"]
  }
}
```

Allowed fields:

- `categories`: 1–20 exact category names, trimmed, case-sensitive.
- `source_types`: 1–4 values from `faq`, `html`, `pdf`, `docx`.
- `source_paths`: 1–20 exact indexed source paths. These are metadata comparisons,
  not filesystem reads; use a `source_path` returned in an earlier citation.

Values within one field use OR; different fields use AND. Omitted/null fields
impose no additional restriction. Empty lists, blank names, unknown filter fields
and permission overrides such as `visibility` are rejected. Source visibility
(`public`) and embedding compatibility remain fixed server-side SQL predicates.
Metadata filters use exact Python comparisons on the fused candidates, not SQL
or filesystem operations. Server-owned SQL constraints remain parameterized.

The user-selected metadata filters now apply **after** vector/BM25 ranking and RRF
fusion, **before** Cross-Encoder reranking and final top-k. This requested order can
miss matching documents outside the bounded hybrid candidate pool. A no-match
filter returns no candidates and an explicit
assistant explanation, never an unfiltered retry or a fabricated policy answer.

JSON and multipart requests are supported; in multipart, send `policy_filters`
as a JSON-encoded text field. Filters apply to the current request and must be
resent for later turns. Clarification handoffs within the same request preserve
them. The frontend execution inspector displays applied filters in the `sources`
event; a filter-selection form is not implemented, so set filters through the API.
No automatic LLM extraction of filters, ingestion change or database migration
is included.

## Earlier metadata-only verification

- Automated backend suite: 580 passed; frontend inspector suite: 18 passed.
- Real read-only retrieval checks passed for combined category/PDF constraints,
  exact source-path selection and a nonexistent category. Both rankers contributed.

The current [reranker pipeline](policy-reranker.md) supersedes the original
prefilter-only implementation. The API field names and permission constraints
are unchanged; filter placement is now post-hybrid as requested.
