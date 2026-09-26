# Third-party source checkouts

`graphrag/` is a Git submodule of the official Microsoft GraphRAG repository:

- Remote: https://github.com/microsoft/graphrag.git
- Tag: `v3.1.2`
- Commit: `243637c4eb94e34c3a5e5c7d871a725e8d6b77fc`
- License: MIT; the upstream checkout retains its license and notices.

The parent repository tracks its URL in `.gitmodules` and its exact commit as a
Git submodule pointer, rather than copying upstream files into our history.
The source is still excluded from the application Docker build context.
After cloning this project, initialize the pinned source with:

```bash
git submodule update --init --recursive third_party/graphrag
```

Alternatively, clone this project with `git clone --recurse-submodules <project-url>`.
Then follow the [source installation instructions](../docs/microsoft-graphrag.md#source-installation).
Initialization downloads source; it does not install Python dependencies.
Do not use `git submodule update --remote` for normal setup: it can select a
different upstream revision instead of the version tested by this project.

Source edits belong to the submodule repository. Committing the parent project
does not back up uncommitted edits inside the submodule. To share custom upstream
changes, commit and push them to your own fork, then update the submodule URL and
commit pointer in the parent repository. Never publish a pointer to a commit
that other users cannot fetch.

All eight GraphRAG workspace packages are installed in editable mode in
`.venv-graphrag`. Other dependencies remain regular installed packages. Keep the
checkout in place: deleting or moving it breaks those editable imports. Restart
running processes after changing source code. Source changes may require a new
index; do not modify the active index to bypass compatibility checks.

Useful entry points inside `graphrag/packages/graphrag/graphrag/`:

- `api/`: public indexing and search APIs.
- `index/operations/extract_graph/graph_extractor.py`: entity and relationship extraction.
- `prompts/query/`: default Local, Global, and DRIFT search prompts.
- `query/`: retrieval and answer-generation implementation.

Project-specific integration and prompt overrides remain in
`../graphrag_runtime/` and `../app/services/microsoft_graphrag.py`.
