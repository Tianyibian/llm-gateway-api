# One scope gate per business branch

"Scope guardrail" means a model-backed decision about whether a business question
belongs to a branch. It is distinct from authentication, ownership, upload limits,
Pydantic validation, evidence checks and read-only database enforcement.

| Branch | Scope calls per entry | Owner |
| --- | --- | --- |
| `graph_rag_search` | 1 | `GraphRAGGuardrail` at branch entry |
| `policy_search` | 1 | `PolicyGuardrail` before ensemble retrieval |
| `additional_search` | 1 | `ClarificationService.assess`, combining scope and missing-information assessment |
| `general_search` | 0 | No separate scope classifier |
| `file_query` | 0 | Request-scoped attachment handling and grounded answer prompt |
| Image/vision branch | 0 | Image validation and vision answer prompt |
| `analytics_search` (optional Snowflake) | 0 | Structured analytical plan and fixed read-only query compilation |

The router chooses a branch; it is not an additional allow/reject scope gate.
Clarification can hand off to another branch: additional assessment once, then the
destination's own assessment once. This is **per branch**, not one scope call
across an entire conversation or a blanket approval shared by unrelated branches.
No approval is cached across user turns or accepted from API input.

## GraphRAG entry paths

- Assistant: `assess_graph_scope -> supervisor.run_approved -> planner -> tools`.
- Standalone `/api/graphrag/query`: `supervisor.run -> root assessment -> run_approved`.
- `/api/graphrag/guardrail`: one diagnostic assessment, no planner or retrieval.

Mode selection and multiple planning rounds do not cause another scope call.
The former `task_guardrail` dependency and per-task evaluation are removed. The
retained legacy hierarchy also does not recheck scope for assignments/specialists.
Planner clarification is not a second scope guardrail: it indicates missing input.

## What remains enforced

- Registered tools only; bounded batches, rounds, timeouts and result counts.
- Known parent evidence IDs, duplicate detection and task coverage at synthesis.
- Planner-declared `entity_mentions` must occur in the task and come from the
  original question or explicitly cited adapter evidence, with approved types.
- Cypher plan/template validation, parameterization, question-alignment review,
  EXPLAIN read-only checks, dataset filters and database write restrictions.
- Microsoft index verification, fixed retrieval adapters and source/citation checks.
- Policy public-document filters; file/image validation; conversation ownership.

No secondary classifier independently verifies every planner question. The
planner must preserve user intent, and entity declarations are not proof of
exhaustive entity extraction or semantic alignment. The removal reduces duplicate
scope decisions and latency but trades away that independent model check. Prompt
instructions cannot replace the remaining execution and data-access controls.

## Where to change scope rules

- GraphRAG business domain/capabilities: `app/graphrag_policy.json`.
- GraphRAG assessment instructions/decision handling: `app/services/graphrag_guardrail.py`.
- Policy scope: `app/services/policy_guardrail.py`.
- Clarification scope and required fields: `app/services/clarification_service.py`.

Changing these does not grant new database fields or tools. Query schemas, adapters
and read-only permissions remain separate. Removing the duplicate gate also does
not implement category aliases such as `speaker` -> `Smart Speaker`, change the
root domain, or establish that a catalog item is currently in stock. Product
discovery/details now belong to GraphRAG business scope; missing fields are a
planner data-capability outcome, not grounds for scope rejection. See
[product queries](graph-product-queries.md).

## Verification

```bash
.venv-langchain/bin/python -m pytest tests/test_branch_scope_counts.py tests/test_graph_supervisor.py tests/test_graph_agents.py tests/test_policy_ensemble.py -q
```

These use explicitly fake providers to assert exact call counts, not live model
classification quality. Opt-in live checks (model charges; public graph reads;
no conversation writes):

```bash
.venv-langchain/bin/python -m app.cli.evaluate_single_scope
```
