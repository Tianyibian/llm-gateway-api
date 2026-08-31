# LLM Gateway

**A production-style, provider-agnostic gateway for large-language-model APIs — streaming, stateful, and persistence-backed.**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-async-009688)](https://fastapi.tiangolo.com/)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-async%20ORM-d71f00)](https://www.sqlalchemy.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-17-4169E1)](https://www.postgresql.org/)
[![Alembic](https://img.shields.io/badge/Alembic-migrations-6BA81E)](https://alembic.sqlalchemy.org/)
[![Tests](https://img.shields.io/badge/tests-pytest%20%2B%20postman-0A9EDC)](#testing)
[![License](https://img.shields.io/badge/license-MIT-black)](LICENSE)

LLM Gateway puts a single, stable HTTP contract in front of multiple inference
backends. Clients always speak the same `messages` format and receive the same
Server-Sent Event stream, whether the tokens come from the **OpenAI API** or a
**self-hosted Ollama** model. Conversations are durable: every completed turn is
written to a relational database, so a client can resume a thread by ID instead
of replaying history on every request.

---

## Highlights

| | |
|---|---|
| **Streaming by default** | Both endpoints emit Server-Sent Events (`metadata` → `delta` → `done`), so tokens render as they are generated. |
| **Provider-agnostic** | `LLMServiceFactory` returns an adapter chosen from configuration. Swapping OpenAI ⇄ Ollama requires no client or endpoint changes. |
| **Optional orchestration** | Use the direct provider adapters by default, or enable LangChain through configuration without changing routes, persistence, or clients. |
| **LangGraph assistant** | Classifies each query, follows a conditional graph branch, and streams a grounded answer with visible route and source metadata. |
| **Knowledge Base RAG** | Loads CSV, HTML, PDF, and DOCX sources, creates local or OpenAI embeddings, retrieves with pgvector HNSW search, and returns citations. |
| **Image understanding** | Validates an uploaded image locally, then streams analysis from OpenAI vision without persisting the upload. |
| **Stateful conversations** | Server-side history: send only the new user turn and the service prepends stored context. |
| **Durable & transactional** | Async SQLAlchemy + PostgreSQL persistence; a turn is committed atomically, and a conversation auto-created by a failed first stream is cleaned up rather than left orphaned. |
| **Versioned schema** | Alembic migrations are tracked independently of application startup — no implicit schema mutation on boot. |
| **Ownership enforced** | Conversation reads and mutations are scoped to a `user_id`; another user's thread is indistinguishable from a missing one (404). |
| **Tested two ways** | Deterministic `pytest` suite (no API credits, no Ollama required) plus a Postman collection that asserts against *live* providers. |

---

## Architecture

```
                    ┌──────────────────────────────┐
   POST /api/chat      │        FastAPI  layer        │
   POST /api/reason    │  routes · Pydantic schemas   │
   POST /api/assistant │   SSE · browser frontend    │
        ──────────► │        SSE  response         │
                    └───────────────┬──────────────┘
                                    │
                    ┌───────────────▼──────────────┐
                    │     ConversationService      │   load history
                    │  (async SQLAlchemy · Alembic)│ ◄─────────────►  DB
                    └───────────────┬──────────────┘   persist turn
                                    │
                    ┌───────────────▼──────────────┐
                    │      LLMServiceFactory       │
                    │   provider × service type    │
                    └───────┬──────────────┬───────┘
                            │              │
                   ┌────────▼───────┐ ┌────▼───────────┐
                   │ OpenAIService  │ │ OllamaService  │
                   │ Responses API  │ │ local /api/chat│
                   └────────────────┘ └────────────────┘
```

**Request flow:** `POST /api/chat` → load stored history → merge with incoming
messages → factory selects an adapter → provider stream → completed turn saved
atomically → `[DONE]`.

The factory keys on *provider*, *service type* (`chat`, `reason`, and a reserved
`recommendation` slot), and *orchestrator* (`native` or `langchain`). New model
runtimes plug in without touching routing code — an application of the factory +
adapter patterns.

---

## API

### Inference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat` | Stateful multi-turn conversation. Returns a `conversation_id`; reuse it and send only the new turn. |
| `POST` | `/api/reason` | Stateless reasoning-oriented response — a reasoned final answer plus a concise explanation (not hidden chain of thought). |
| `POST` | `/api/classify` | Structured routing: `general_search`, `product_search`, `return_search`, or `knowledge_search`. |
| `POST` | `/api/assistant` | Stateful multimodal LangGraph assistant. Accepts text JSON or multipart image input and persists completed turns. |
| `GET` | `/api/knowledge/status` | Show compatible indexed document/chunk counts and embedding configuration. |

### Conversation management

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/conversations` | Create a thread explicitly |
| `PATCH` | `/api/conversations/{id}` | Rename (`{"user_id": "...", "title": "..."}`) |
| `DELETE` | `/api/conversations/{id}?user_id=` | Delete (cascades messages, `204`) |
| `GET` | `/api/users/{user_id}/conversations` | List a user's threads |
| `GET` | `/api/conversations/{id}/messages?user_id=` | Read stored turns in order |

### Streamed response format

```
event: metadata
data: {"provider":"openai","model":"...","service":"chat","conversation_id":"..."}

event: delta
data: {"content":"FastAPI"}

event: done
data: "[DONE]"
```

Interactive docs are served at `/docs`; liveness at `/health`.

---

## Quick start

```bash
python3.13 -m venv .venv-langchain && source .venv-langchain/bin/activate
python -m pip install -r requirements-langchain.txt

cp .env.example .env          # then set LLM_PROVIDER (and a key, if using OpenAI)
docker compose up -d postgres  # start PostgreSQL 17 on localhost:5432
alembic upgrade head          # create / update the schema
ollama pull embeddinggemma    # local 768-dimensional embeddings
python -m app.cli.ingest_knowledge
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000` for the local Aster assistant interface. It is
served by FastAPI and requires no separate JavaScript build process.

```bash
curl -N -X POST http://127.0.0.1:8000/api/chat \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo-user","messages":[{"role":"user","content":"Explain FastAPI in two sentences."}]}'
```

`-N` disables curl buffering so deltas appear as they stream.

---

## Configuration

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` | `openai` · `ollama` · `auto` (uses OpenAI when a valid key is present, otherwise Ollama) |
| `LLM_ORCHESTRATOR` | `native` (default) · `langchain` (optional model abstraction) |
| `OPENAI_API_KEY` | Required only for the OpenAI path |
| `OPENAI_CHAT_MODEL` / `OPENAI_REASON_MODEL` | Model per service type |
| `OLLAMA_BASE_URL` / `OLLAMA_CHAT_MODEL` / `OLLAMA_REASON_MODEL` | Local inference, no key required |
| `DATABASE_URL` | Async PostgreSQL URL using the `postgresql+asyncpg` driver |
| `DATABASE_POOL_SIZE` / `DATABASE_MAX_OVERFLOW` | Base and burst capacity for SQLAlchemy's async connection pool |
| `DATABASE_POOL_TIMEOUT_SECONDS` | Maximum wait for an available pooled connection |
| `DATABASE_POOL_RECYCLE_SECONDS` | Maximum lifetime before a pooled connection is replaced |
| `BUSINESS_DATA_DIR` | Directory containing `Products.csv`, `Categories.csv`, and `Suppliers.csv` |
| `KNOWLEDGE_BASE_DIR` | Directory containing supported RAG source files |
| `RAG_EMBEDDING_PROVIDER` | `ollama` (local) or `openai`; independent of the answer-model provider |
| `OLLAMA_EMBEDDING_MODEL` / `OPENAI_EMBEDDING_MODEL` | Embedding model selected by the RAG provider |
| `RAG_CHUNK_SIZE` / `RAG_CHUNK_OVERLAP` | Character-based chunking settings |
| `RAG_RETRIEVAL_K` | Maximum number of chunks supplied to the grounded answer prompt |
| `OPENAI_VISION_MODEL` | OpenAI model used only for explicit image-analysis requests |
| `OPENAI_VISION_DETAIL` | `low`, `high`, `original`, or `auto`; defaults to `high` |
| `VISION_MAX_IMAGE_BYTES` | Local upload limit; defaults to 10 MB |

Both providers share one request schema and one SSE contract, so switching is a
configuration change — clients and tests stay identical. Restart Uvicorn after
editing `.env`.

### Optional LangChain runtime

LangChain is an orchestration layer, not another model provider. With
`LLM_ORCHESTRATOR=langchain`, the factory still selects OpenAI or Ollama, then
wraps that provider's LangChain chat model behind the same local `LLMService`
interface. Conversation persistence, ownership checks, SSE, and API contracts
do not change.

LangChain 1.x and the assistant graph require Python 3.10 or newer. This project
uses the Python 3.13 `.venv-langchain` environment for application development:

```bash
python3.13 -m venv .venv-langchain
source .venv-langchain/bin/activate
python -m pip install -r requirements-langchain.txt
```

Then set these values in the local `.env` and restart Uvicorn:

```dotenv
LLM_ORCHESTRATOR=langchain
LLM_PROVIDER=ollama
```

Use `LLM_PROVIDER=openai` instead to run the same adapter through OpenAI. Set
`LLM_ORCHESTRATOR=native` to return to the direct SDK/HTTP implementations.

The `/api/classify` endpoint always uses the LangChain runtime because
it combines `ChatPromptTemplate` with the provider model's structured-output
interface. Classification completes before routing and therefore returns one
validated JSON object rather than an SSE stream.

```bash
curl -X POST http://127.0.0.1:8000/api/classify \
  -H 'Content-Type: application/json' \
  -d '{"query":"Can I return a smart lock after installation?"}'
```

### LangGraph assistant

The browser UI and `/api/assistant` use a compiled `StateGraph`:

```text
START -> detect_modality
              |
              +-> vision_answer (OpenAI) ------------> END
              +-> classify_query
                        |
                        +-> general_answer ---------------------> END
                        +-> search_products -> product_answer -> END
                        +-> retrieve_knowledge -> knowledge_answer -> END
```

The graph has separate input, output, and overall state schemas. Every node
returns only its state update. The conditional edge reads `state["route"]` and
selects the next node. Product search joins the local products, categories, and
suppliers CSV files in memory, then supplies only the best matching records to
the model. SSE events expose `metadata`, `route`, optional `sources`, real model
`delta` tokens, and `done`. Before execution, the endpoint loads the owned
conversation history from PostgreSQL. After a successful stream, it atomically
saves the user query and complete assistant answer as one turn.

The CSV search is intentionally a local demonstration implementation. In a
production system, exact prices, stock, and availability should come from the
transactional product service or PostgreSQL rather than embeddings or static
files. PostgreSQL full-text search or trigram indexes work well for catalog
names; OpenSearch/Elasticsearch becomes useful for larger faceted catalogs.
Use pgvector or another vector index for semantic descriptions and support
documents, while preserving relational filters and source/version metadata.
Policies belong in a versioned RAG knowledge pipeline with citations. A hybrid
router can therefore send exact business facts to SQL/service APIs and
unstructured policy questions to retrieval.

### Knowledge Base RAG

The ingestion command loads each FAQ row, HTML help article, non-empty PDF page,
and DOCX document as a versioned source unit. It normalizes text, creates
overlapping chunks, generates 768-dimensional embeddings, and stores them in
PostgreSQL with a pgvector HNSW cosine index. Checksums make ingestion
idempotent: unchanged sources are skipped, changed sources are replaced, and
removed sources are deleted.

```bash
docker compose up -d postgres
alembic upgrade head
ollama pull embeddinggemma
python -m app.cli.ingest_knowledge
curl http://127.0.0.1:8000/api/knowledge/status
```

Set `RAG_EMBEDDING_PROVIDER=openai` to use `text-embedding-3-small` instead.
The OpenAI key stays server-side; the local Ollama default requires no key or
embedding API credits. Run ingestion again after changing the embedding model
or provider. Retrieval refuses to mix vectors from a different configured
embedding space.

`return_search` and `knowledge_search` share the retrieval branch. Only
`visibility=public` documents are eligible for assistant retrieval. Internal
support DOCX files are indexed with `visibility=internal` as preparation for
future role-based access, but the public assistant's SQL filter cannot return
them. Each SSE `sources` event includes title, file, type, category, PDF page,
chunk index, and similarity score; the answer prompt cites those excerpts as
`[1]`, `[2]`, and so on.

### Image analysis

Attaching an image in the browser sends a multipart request to the same
`/api/assistant` endpoint used for text. The `detect_modality` node routes it to
the graph's `vision_answer` branch. This branch is intentionally OpenAI-only
even when `LLM_PROVIDER=ollama`; the
OpenAI key remains server-side in `.env`, and the browser never receives it.
The backend verifies the actual image content with Pillow, permits PNG, JPEG,
WEBP, and non-animated GIF, enforces a 10 MB local limit, converts the bytes to
a Base64 data URL, and streams only response text. The application does not
save the image bytes in PostgreSQL or the project directory. It saves the
question and textual image analysis, so later turns can use that text as
conversation context. Image inputs consume
billable tokens, so the API is called only after the user explicitly attaches
and sends an image.

```bash
curl -N -X POST http://127.0.0.1:8000/api/assistant \
  -F 'query=Read the visible text in this image.' \
  -F 'user_id=demo-user' \
  -F 'image=@/absolute/path/to/image.png;type=image/png'
```

**Secrets:** `.env` is git-ignored and only `.env.example` is committed; keys
never appear in source, docs, or requests. Verify with `git check-ignore -v .env`.
The Compose credentials are intentionally local-development defaults; replace
them with managed secrets in any shared or deployed environment.

---

## PostgreSQL database

The application uses PostgreSQL 17 with pgvector through SQLAlchemy's async
`asyncpg` driver.
The local service is defined in `compose.yaml`; its data survives container
restarts in the `postgres_data` Docker volume.

```bash
docker compose up -d postgres
docker compose ps
alembic upgrade head
alembic current
alembic check
```

Stop the local database without deleting its volume:

```bash
docker compose stop postgres
```

### Migrate existing SQLite data

Apply the schema to an empty PostgreSQL database first, then validate and run
the copy:

```bash
docker compose up -d postgres
alembic upgrade head
python -m scripts.migrate_sqlite_to_postgres --dry-run
python -m scripts.migrate_sqlite_to_postgres
```

The source defaults to `./llm_gateway.db`; override it with
`SQLITE_SOURCE_URL=sqlite+aiosqlite:////absolute/path/to/source.db` when needed.
The migration copies IDs, timestamps, conversations, and messages in one target
transaction, updates the PostgreSQL message-ID sequence, refuses a non-empty
target, masks the target password in output, and never modifies or deletes the
SQLite source. To roll back before removing the source, point `DATABASE_URL`
back to the preserved SQLite URL and restart Uvicorn.

---

## Testing

```bash
pytest -q
```

The `pytest` suite runs against temporary SQLite databases with the provider
replaced by a deterministic fake, covering routing, request validation, factory
selection, SSE formatting, ownership rules, atomic turn persistence,
failed-first-turn cleanup, cascading deletes, history ordering, and Alembic
upgrade/downgrade, multi-format knowledge loading, chunking, embeddings, and
grounded citation routing. It consumes no API credits and needs no local model.

The Postman collection (`postman/`) is deliberately **not** mocked: it drives a
running Uvicorn process against real providers and asserts status, SSE content
type, the expected provider and service type, streamed output, `[DONE]`, and the
absence of an error event — including a seven-request stateful conversation flow.

Passing unit tests proves the application logic; the Postman run proves the
integration.

---

## Project layout

```
app/
├── api/routes.py                 # LLM endpoints + SSE
├── api/conversation_routes.py    # conversation CRUD
├── core/config.py                # environment configuration
├── db/models.py                  # conversation, message, vector tables
├── db/session.py                 # async engine & session factory
├── models/schemas.py             # Pydantic request & response schemas
├── services/base.py              # abstract service + service types
├── services/factory.py           # provider × service-type factory
├── services/langchain_service.py # optional LangChain adapter
├── services/openai_service.py    # OpenAI adapter
├── services/ollama_service.py    # Ollama streaming adapter
├── services/query_classifier.py # prompt + structured-output routing
├── services/assistant_service.py # StateGraph + conditional branches
├── services/product_catalog.py   # local CSV search adapter
├── services/knowledge_loader.py  # CSV / HTML / PDF / DOCX loaders
├── services/embedding_service.py # Ollama / OpenAI embedding adapters
├── services/knowledge_service.py # ingestion + pgvector retrieval
├── services/vision_service.py   # OpenAI Responses image analysis
├── services/conversation_service.py
├── cli/ingest_knowledge.py       # idempotent indexing command
├── static/                       # local assistant web interface
└── main.py
compose.yaml                      # local PostgreSQL 17 service
migrations/                       # Alembic revisions
scripts/                          # SQLite-to-PostgreSQL data migration
tests/                            # deterministic provider + temp SQLite
postman/                          # live end-to-end collection
requirements-langchain.txt        # optional LangChain dependency set
```

---

## Notes & limits

- `user_id` is a demonstration ownership boundary, not authentication. A real
  deployment should derive identity from a verified token (e.g. JWT) rather than
  trusting client input.
- `LLM_PROVIDER=auto` resolves at service-construction time; it does not fail
  over mid-request after quota, billing, or network errors.
- `/api/reason` returns a reasoned answer and a short explanation by design — it
  does not expose a model's internal reasoning trace.

## License

MIT
