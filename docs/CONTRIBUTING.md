# Documentation contribution guide

## Choose the right page type

- **Recipe:** a short task with prerequisites, a minimal example, expected
  output/interpretation, and links to the authoritative API reference.
- **Reference page:** a compact overview table plus generated public API
  docstrings; do not duplicate every parameter in prose.
- **Release note:** user-visible `Added`, `Changed`, `Fixed`, `Deprecated`,
  or `Removed` items and a migration note for public API changes.

## Writing rules

- State the user outcome first and use stable, deterministic examples.
- Describe units, schemas, backend/optional-extra requirements, and important
  fallback or ordering behavior.
- Use relative Markdown links and meaningful link text.
- Add alt text to every image. Keep diagram source next to its rendered asset.
- Mark an intentional future visual with `<!-- Visual placeholder: ... -->`;
  do not leave TODO links in published content.

## Before opening a pull request

```bash
uv run --group docs python scripts/build_notebook_docs.py
uv run --group docs zensical build --strict
pytest tests/docs
```

Documentation notebooks in `docs/src/` are the source of truth. The conversion
script generates ignored Markdown pages and output assets, and adds a Binder
badge to each page. Do not edit generated files; rerun the script after
changing a notebook.

To serve the site with notebook pages available automatically:

```bash
bash scripts/serve_docs.sh
```

Public API changes must update the generated docstring contract and the
appropriate recipe/reference/release note, or state why no user-facing
documentation change is needed.

## Dependency environments

Use `uv sync --group dev` for the normal contributor environment and
`uv sync --group docs` for documentation. Third-party comparison suites are
intentionally isolated because their dependency stacks conflict with the main
environment. Create one with, for example,
`bash scripts/setup_benchmark_env.sh pymove --python 3.10`; the supported
environment names and pins live in `benchmarks/environments/`.
