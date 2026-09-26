# Optional dependency sets

Run these installation commands from the repository root. The base requirements
remain in `requirements.txt` for the existing Docker and CI entry points.

| File | Purpose | Environment |
| --- | --- | --- |
| `langchain.txt` | Base app plus LangChain/LangGraph | `.venv-langchain` |
| `neo4j.txt` | LangChain app plus Neo4j driver | `.venv-langchain` |
| `snowflake.txt` | LangChain app plus Snowflake support | `.venv-langchain` |
| `graphrag.txt` | Version-pinned Microsoft GraphRAG | `.venv-graphrag` only |
| `graphrag-source.txt` | Editable GraphRAG source packages; provision dependencies first | `.venv-graphrag` only |

```bash
.venv-langchain/bin/python -m pip install -r requirements/neo4j.txt
.venv-graphrag/bin/python -m pip install -r requirements/graphrag.txt
```

These are relocated dependency lists, not version upgrades. Do not combine the
GraphRAG and application environments. Transitive dependencies are not fully locked.

For the optional official Git checkout, follow the
[source installation guide](../docs/microsoft-graphrag.md#source-installation).
