# Project map

The application is a stateful support assistant with separate HTTP, orchestration,
retrieval, model-adapter and persistence layers. Run commands from the repository
root; folder organization does not change API URLs or Python package imports.

For upstream Microsoft GraphRAG internals, see the
[local source checkout guide](../third_party/README.md). The editable checkout
lives in `third_party/graphrag/`; our integration remains in `graphrag_runtime/`.

## Read the application in this order

| Step | File | Question it answers |
| --- | --- | --- |
| 1 | [main.py](../app/main.py) | Where are the application, routers and UI registered? |
| 2 | [api/routes.py](../app/api/routes.py) | How does a request enter, load history and return SSE? |
| 3 | [models/schemas.py](../app/models/schemas.py) | What input and classification shapes are accepted? |
| 4 | [services/factory.py](../app/services/factory.py) | How are configured providers and services assembled? |
| 5 | [services/assistant_service.py](../app/services/assistant_service.py) | How do state, nodes and conditional edges execute a request? |
| 6 | [services/query_classifier.py](../app/services/query_classifier.py) | Which prompt selects the business route? |
| 7 | [services/conversation_service.py](../app/services/conversation_service.py) | How are owned conversations and complete turns persisted? |
| 8 | [static/app.js](../app/static/app.js) | How does the UI send questions and render stream events? |

The HTTP router dispatches URLs; the model classifier selects a business branch.
LangGraph state belongs to an execution, while persisted messages survive across
requests. The client-supplied `user_id` is not production authentication.

## Services grouped by responsibility

Service modules remain in `app/services/` so imports and runtime entry points stay
stable. Use these groups as a reading guide rather than opening every file at once.

| Responsibility | Modules in `app/services/` |
| --- | --- |
| Model adapters | `base.py`, `openai_service.py`, `ollama_service.py`, `langchain_service.py`, `vision_service.py` |
| Assembly and shared behavior | `factory.py`, `errors.py`, `streaming.py` |
| Routing and clarification | `query_classifier.py`, `clarification_service.py`, `assistant_service.py` |
| Conversation storage | `conversation_service.py`; ORM definitions live in `app/db/` |
| Policy/support vector RAG | `knowledge_loader.py`, `embedding_service.py`, `knowledge_service.py` |
| Graph scope and task planning | `graphrag_guardrail.py`, `graphrag_service.py`, `graph_supervisor.py` (`graph_agents.py` is legacy) |
| Evidence synthesis | `graph_answer.py` |
| Neo4j retrieval | `neo4j_service.py`, `cypher_templates.py`, `cypher_compiler.py`, `cypher_checks.py` |
| Microsoft GraphRAG adapter | `microsoft_graphrag.py`; official API calls live in `graphrag_runtime/worker.py` |
| Data preparation | `graph_data.py`, `graph_review_data.py`, `neo4j_data.py` |
| Optional warehouse analytics | `analytics_planner.py`, `snowflake_analytics.py`, `csv_analytics.py`, `analytics_comparison.py` |

## Routing and specialist boundaries

- `general_search`: general conversation.
- `additional_search`: guarded clarification for essential missing information.
- `file_query`: request-scoped TXT/Markdown/PDF questions; see [file-query setup](file-query.md).
- Catalog questions use `graph_rag_search`; the standalone CSV product route has been removed. Current price and stock lookup is unavailable.
- `policy_search`: unified return/refund and help-center vector RAG.
- `graph_rag_search`: guarded decomposition and direct tool-task dispatch.
- `analytics_search`: optional configured Snowflake reporting.
- Attached images enter the vision branch before text classification.

Inside the graph branch, the planner dispatches subtasks directly to explicit
predefined Cypher, constrained Text-to-Cypher and Microsoft GraphRAG tools.
It does not execute arbitrary model-written Cypher. Microsoft Local is the default;
Global is selected per request, while DRIFT remains a low-level experiment.
Dependent tasks use prior evidence. The final MapReduce stage combines source-backed
evidence; neither subtasks nor map calls require fixed business-role agents.

## Data locations are intentionally stable

- `Business_data/`: synthetic source CSVs. Structured Neo4j sales data and the
  sampled Microsoft GraphRAG review corpus use different preparation pipelines.
- `Knowledge Base/`: policy/support documents for the separate pgvector RAG.
- `.local/`: generated indexes and diagnostics. Do not edit verified artifacts.
- `.env` and provider-specific local environment files: private configuration.
- `.venv-langchain/`: application environment; `.venv-graphrag/`: isolated vendor runtime.

Moving these directories can invalidate source paths, executable shebangs,
verification hashes or local configuration. This organization pass preserves them.
PostgreSQL/Neo4j container volumes and existing local database files are untouched.

## Reference material and validation

Exploration notebooks now live in `notebooks/`. The original MySQL preprocessing
reference is in `scripts/reference/`, not the active indexing path. See
[optional requirements](../requirements/README.md) for relocated dependency lists.

Automated tests are under `tests/`, browser-inspector tests under `tests/frontend/`,
and live API collections under `postman/`. Passing fake-model tests is not proof
that live providers, database services or retrieval accuracy have been verified.
