# Reference preprocessing

`preprocess_data.py` is the legacy MySQL-based review-export example that inspired
the current CSV preparation pipeline. It is not called by FastAPI or GraphRAG.
Its default output remains `output_data/` at the repository root.

For maintained preparation commands, use:

- `python -m app.cli.prepare_graph_data` for reviewed Microsoft GraphRAG text.
- `python -m app.cli.load_neo4j` for the structured business-graph snapshot.

The legacy reference can include customer fields. Do not run it against private
data or submit its exports to a model without reviewing the data and permissions.
