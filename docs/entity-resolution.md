# Database-grounded entity resolution

Product browsing stays inside `graph_rag_search`, using **Text-to-Cypher**, not
`category_products`. Existing other predefined retrieval tools remain available.
There is still one branch scope guardrail; resolving a name is not another scope
assessment.

## Execution

1. `generate_cypher` produces a structured `CypherPlan` with filters quoting the
   original question. The model must not invent canonical names or IDs.
2. `resolve_entities` reads public id/name records for each required Product,
   Category or Supplier label from the configured dataset. A complete set of at
   most 2,000 records per label is required; truncation/invalid data fails closed.
   These are read-only metadata lookups with the executor's usual snapshot,
   EXPLAIN, timeout and plan-budget checks, not a category-products template.
3. A deterministic matcher normalizes Unicode/case and conservative regular
   English plurals, then compares complete token sequences. Full-name matches
   take priority. For example, `speakers` uniquely matches the actual category
   `Smart Speaker` in the current dataset.
4. Unique matches become request-local, server-owned entity bindings. The compiler
   uses a parameterized ID equality predicate; every binding retains its original
   surface term, label, canonical name, ID and dataset. The original question is
   never rewritten to grant new scope. Unknown terms retain their original name
   filter, so they cannot silently become a different product.
5. Multiple matches produce `clarify / entity_ambiguous` with candidate names.
   The adapter propagates this to the supervisor and assistant as a user question,
   not a provider failure. It does not run the final business query or Map/Reduce.
6. Unique/unknown matches continue through semantic review, server recompilation,
   fingerprint validation, read-only EXPLAIN and bounded query execution. The
   reviewer sees the verified name/ID mapping. Public callers cannot supply it.

The frontend's Execution details displays successful entity bindings alongside
the actual Cypher and query mode. Candidate lookups are internal operations,
not separate autonomous agents or additional supervisor tool calls.

## Deliberate limits

- No arbitrary synonyms, typo correction, multilingual translation, semantic
  embeddings or product-specific keyword exceptions are introduced.
- Multiple partial-name matches request clarification, including cases where a
  user might have intended a broad substring search. Explicit collection/search
  semantics need a separate contract before bypassing ambiguity resolution.
- Catalog reads are bounded scans suitable for this prototype. Larger catalogs
  need a complete indexed candidate lookup, not a larger silent result cap.
- Model planning/classification and empty-result follow-up behavior can still
  fail. This change does not claim to fix every previously observed planner issue.
- Clarification uses the existing conversation history flow; no new pending-choice
  state or conversation migration is introduced. Users should reply with a full
  candidate name and their intended question.

## Verification

Deterministic backend suite: **563 passed**; frontend inspector suite: **18 passed**.
Real classifier/supervisor smoke tests for the exact plural question
`what type of speakers do you have` and a camera catalog question passed, with one
scope call each, Text-to-Cypher execution, verified category bindings, and cited
answers. No conversation messages or business records were written by these tests.
An additional real Text-to-Cypher request for the `smart` category returned
`entity_ambiguous` with existing category candidates and no final query execution.
These are smoke tests, not a guarantee of universal model accuracy.

```bash
.venv-langchain/bin/python -m pytest tests/test_entity_resolution.py -q
.venv-langchain/bin/python -m app.cli.evaluate_graph_products --case speakers --case camera
```
