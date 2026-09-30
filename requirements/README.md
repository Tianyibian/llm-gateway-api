# Application and optional dependency sets

Run these installation commands from the repository root. The base requirements
live in `requirements.txt`, including LangChain, LangChain Core, LangGraph and
the OpenAI/Ollama model integrations, the Neo4j driver and local key-pair
utilities. Local setup, Docker and CI use this same entry point. Use Python 3.13
for the documented setup; CI also covers 3.11.

| File | Purpose | Environment |
| --- | --- | --- |
| `langchain.txt` | Compatibility alias for the complete base app | `.venv-langchain` |
| `neo4j.txt` | Compatibility alias; Neo4j still needs explicit database setup | `.venv-langchain` |
| `snowflake.txt` | LangChain app plus Snowflake support | `.venv-langchain` |
| `reranker.txt` | Local policy Cross-Encoder inference | `.venv-langchain` |
| `graphrag.txt` | Version-pinned Microsoft GraphRAG | `.venv-graphrag` only |
| `graphrag-source.txt` | Editable GraphRAG source packages; provision dependencies first | `.venv-graphrag` only |

```bash
.venv-langchain/bin/python -m pip install -r requirements/neo4j.txt
.venv-langchain/bin/python -m pip install -r requirements/reranker.txt
.venv-langchain/bin/python -m app.cli.prepare_policy_reranker
.venv-graphrag/bin/python -m pip install -r requirements/graphrag.txt
```

The root install supports application imports and deterministic tests without
a running Neo4j/Snowflake service, reranker weights, or a Microsoft GraphRAG index.
Snowflake and reranking need optional packages; all retrieval backends need data
and configuration. Do not combine the
GraphRAG and application environments. Version ranges are not a complete lockfile;
run `python -m pip check` and the tests after installing.

For the optional official Git checkout, follow the
[source installation guide](../docs/microsoft-graphrag.md#source-installation).
