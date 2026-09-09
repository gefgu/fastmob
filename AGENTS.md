## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).

## Git operations

The workspace permits edits to the working tree but exposes `.git` as read-only
inside the sandbox. For user-requested staging, commits, or other Git metadata
writes, request elevated permission directly before running the Git command;
do not first attempt the command in the sandbox.

## Documentation notebooks

Documentation notebooks under `docs/src/` are source files. Convert them to
normal Markdown pages, including code blocks and saved outputs, with:

```bash
uv run --group docs python scripts/build_notebook_docs.py
```

The generated `.md` files and `*_files/` output directories are ignored build
artifacts. Run the conversion before documentation tests or a local Zensical
build. Each generated page includes a Binder badge linking to the original
notebook on `main`.
