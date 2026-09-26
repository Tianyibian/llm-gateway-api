# Exploration notebooks

These notebooks are learning and experimentation references, not application
entry points or an automated test suite. Existing cells and outputs are retained.

- `chapter04.ipynb`: model interfaces, prompts, structured output and LCEL.
- `chapter08.ipynb`: RAG and GraphRAG concepts and experiments.

Select the project's `.venv-langchain/bin/python` for LangChain exploration in
VS Code. Saved kernel display names may refer to a previous environment; choose
the interpreter explicitly. Some cells demonstrate older APIs or contain
partial examples, so read them before executing instead of using Run All.
Model calls can incur charges. Keep credentials in the root ignored `.env`,
never in notebook cells or saved outputs.

Run application commands from the repository root. If a new experiment needs
project-relative paths, first resolve the project root explicitly:

```python
from pathlib import Path

cwd = Path.cwd().resolve()
project_root = next(
    path for path in (cwd, *cwd.parents)
    if (path / "app/main.py").is_file()
)
```

The installed Microsoft GraphRAG runtime is intentionally separate in
`.venv-graphrag`. Use `graphrag_runtime/worker.py` for the maintained indexing and
query workflow, rather than treating demonstration cells as its implementation.

See [the code reading guide](../docs/project-map.md) for the application itself.
