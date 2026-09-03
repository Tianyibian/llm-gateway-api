# Smart AI Support API: Postman live integration tests

This collection makes real model requests for every provider/service combination.
It does not use the fake services from the pytest suite.

## 1. Start both provider-specific servers

From the project root, open Terminal 1:

```bash
source .venv-langchain/bin/activate
env LLM_PROVIDER=openai uvicorn app.main:app --port 8001
```

From the project root, open Terminal 2:

```bash
source .venv-langchain/bin/activate
env LLM_PROVIDER=ollama uvicorn app.main:app --port 8002
```

Keep Ollama App running. The local model is `qwen3:4b`.

## 2. Import the collection

1. In Postman, click **Import**.
2. Choose **File**.
3. Select `postman/LLM_API.postman_collection.json`.
4. Open the imported **Smart AI Support API - Live Integration** collection.

The collection already contains these variables:

```text
openai_base_url = http://127.0.0.1:8001
ollama_base_url = http://127.0.0.1:8002
router_base_url = http://127.0.0.1:8002
router_user_id = postman-router-user
conversation_base_url = http://127.0.0.1:8002
conversation_user_id = postman-user
conversation_id = (set automatically)
```

Do not put the OpenAI API key in Postman. The port-8001 server reads it from the local `.env` file.

## 3. Run the provider matrix

You can open each request and click **Send**, or run the complete collection:

1. Click the collection's **...** menu.
2. Select **Run collection**.
3. Select the four requests under **OpenAI** and **Ollama**.
4. Click **Run Smart AI Support API - Live Integration**.

Each request reaches the real provider through FastAPI and has seven automatic
assertions:

- HTTP status is 200
- response Content-Type is SSE
- expected provider is present
- expected service type is present
- at least one delta was streamed
- the stream ended with `[DONE]`
- no SSE error event was returned

Expected matrix:

| Case | URL | Expected metadata |
| --- | --- | --- |
| OpenAI Chat | `http://127.0.0.1:8001/api/chat` | `openai` + `chat` |
| OpenAI Reason | `http://127.0.0.1:8001/api/reason` | `openai` + `reason` |
| Ollama Chat | `http://127.0.0.1:8002/api/chat` | `ollama` + `chat` |
| Ollama Reason | `http://127.0.0.1:8002/api/reason` | `ollama` + `reason` |

## 4. Run the live router tests

Open the **Router - Live Model** folder and run its five requests in order. The
first four call `/api/classify` through the real Ollama model and verify every
supported structured-output route:

| Case | Expected route | Capability |
| --- | --- | --- |
| Greeting and joke | `general_search` | No company data required |
| Product price and inventory | `product_search` | Structured catalog lookup required |
| Return eligibility and window | `return_search` | Return-policy RAG required |
| Password reset | `knowledge_search` | Help-center RAG required |

Each classifier request asserts HTTP 200, the exact route, the complete
`route`/`reason`/`confidence` schema, a non-empty reason, and confidence within
the valid 0-to-1 range.

The fifth request is deliberately more comprehensive. It calls
`/api/assistant` and verifies the actual LangGraph `return_search` branch,
pgvector document sources, an SSE answer containing at least one numbered
citation, `[DONE]`, and no error event. These are live integration tests: they
do not use the deterministic fake services from pytest.

Before running the fifth request, build the local RAG index once:

```bash
alembic upgrade head
ollama pull embeddinggemma
python -m app.cli.ingest_knowledge
```

## 5. Run the stateful multi-turn flow

Run the seven requests under **Stateful Conversation** in their numbered order:

1. **Create Conversation** creates a database record and stores its ID in the
   `conversation_id` collection variable.
2. **First Turn** tells the real model that the user's preferred backend
   framework is FastAPI. The completed user and assistant messages are saved.
3. **Second Turn Recalls History** sends only a new question. Its assertion
   reconstructs the SSE deltas and verifies that the answer recalls FastAPI.
4. **Verify Persisted Messages** checks that the database contains exactly two
   complete turns ordered as user, assistant, user, assistant.
5. **Verify User Conversation List** checks that the new conversation appears in
   the user's conversation list.
6. **Update Conversation Title** renames the conversation and verifies the
   returned title and ID.
7. **Delete Conversation** removes the conversation and its stored messages and
   verifies HTTP 204.

The default `conversation_base_url` uses the Ollama server on port 8002, so this
stateful flow does not consume OpenAI API credits. Set it to
`http://127.0.0.1:8001` only when you intentionally want to repeat the same flow
through OpenAI. Creating a new conversation at the beginning makes repeated
collection runs independent.
