# MkDocs Documentation System

> Add MkDocs with Material theme and mkdocstrings to give skmob2 a hosted, auto-generated API reference driven entirely by existing Python docstrings.

## Purpose & Motivation

skmob2 currently has no published documentation. All knowledge about the public API lives in docstrings and internal markdown files (`CLAUDE.md`, `SETUP_GUIDE.md`, `BENCHMARK_RESULTS.md`) that are not surfaced to users. Anyone evaluating the library must read source code to understand inputs, outputs, and semantics.

MkDocs with mkdocstrings converts the existing docstrings — which are already well-structured (NumPy-style Parameters / Returns / Examples sections) — into a navigable HTML site without requiring new prose content to be written up front. The Material theme is the de-facto standard for Python library documentation and is widely recognised by users of the scientific Python ecosystem.

The motivation to do this now is that the public API is stable enough (8 exported functions across 6 measure files) that a documentation site is immediately useful, and the docstrings are rich enough to produce a meaningful reference without additional writing work.

## Success Criteria

- Running `uv run mkdocs serve` from the repo root starts a local docs site with no errors.
- Running `uv run mkdocs build --strict` produces a `site/` directory with zero warnings.
- Every function listed in `skmob2/__init__.py:8-17` appears on an API reference page with its parameters, return type, and example rendered.
- A GitHub Actions workflow builds the docs on every push to `main` and deploys them to GitHub Pages (or at minimum builds without error to act as a CI gate).
- A new developer can follow the Getting Started page to install skmob2 and run their first `jump_lengths` call.

## Scope

**In scope:**
- `mkdocs.yml` configuration file at the repo root.
- `docs/` directory restructured to hold MkDocs source pages (`index.md`, `getting-started.md`, `api/` subdirectory).
- `pyproject.toml` updated with a `docs` optional-dependency group listing `mkdocs`, `mkdocs-material`, `mkdocstrings[python]`.
- One API reference page per public measure module, auto-populated via `mkdocstrings` `::: skmob2.module` directives.
- A Getting Started page that covers installation and a minimal usage example using `jump_lengths`.
- A `.github/workflows/docs.yml` CI job that builds (and optionally deploys) the docs.
- Moving the existing `docs/features/` sub-directory so it does not collide with MkDocs page discovery.

**Out of scope:**
- Writing new narrative content (tutorials, conceptual guides, architecture deep-dives).
- Documenting the Rust extension (`src/lib.rs`) — Rust doc comments are not exposed via mkdocstrings.
- Versioned documentation (mike plugin).
- Search customisation or analytics.
- Improving existing docstrings — the plan uses them as-is.

## Constraints

- `uv` is the only allowed Python installer; no `pip install` in CI or developer instructions.
- `import pandas` and `import polars` are forbidden inside `skmob2/` source files; this constraint does not affect docs infrastructure.
- The project already has a `CI.yml` workflow (`/.github/workflows/CI.yml`) that handles wheel builds across many platforms. The docs workflow must be a separate file and must not touch that job.
- `docs/features/` currently contains planning documents that are not MkDocs pages; they must not appear in the built site nav.
- Python minimum is 3.8 (`pyproject.toml:7`), but MkDocs itself only needs to run in the developer/CI environment, not in the distributed package.

## Architecture & Integration Points

**pyproject.toml** (`/home/gustavo/skmob2/pyproject.toml:20-36`) — the `[project.optional-dependencies]` table is where `docs` extras will be added alongside `dev`, `fitting`, and `diversity`. The `docs` group will list `mkdocs`, `mkdocs-material`, and `mkdocstrings[python]`.

**skmob2/__init__.py** (`/home/gustavo/skmob2/skmob2/__init__.py:1-17`) — the canonical list of 8 public symbols that mkdocstrings must resolve. Each symbol maps back to a concrete file:

- `jump_lengths` — `skmob2/measures/jump_lengths.py:11`
- `radius_of_gyration` — `skmob2/measures/radius_of_gyration.py:11`
- `od_matrix`, `od_metrics_per_area` — `skmob2/measures/od.py:36,78`
- `log_truncated_powerlaw`, `fit_values_to_truncated_powerlaw` — `skmob2/measures/mobility_laws.py:12,45`
- `activity_transition_matrix` — `skmob2/measures/activity.py:22`
- `intermittance_and_degree_of_return` — `skmob2/measures/individual.py:23`

**mkdocs.yml** (new file at `/home/gustavo/skmob2/mkdocs.yml`) — central config; references the `docs/` source directory and declares nav, plugins, and theme.

**docs/** (restructured from `/home/gustavo/skmob2/docs/`) — MkDocs source root. `docs/features/` will be excluded from the nav so it does not appear in the published site; it stays on disk for internal planning use.

**`.github/workflows/docs.yml`** (new file) — separate from `CI.yml`; triggers on push to `main` and on pull requests (build-only on PRs, deploy on `main`).

The system works as follows: mkdocstrings resolves `:::` directives by importing the Python module at build time, extracting docstrings, and rendering them as Markdown. This means the Rust extension (`skmob2/_core.*.so`) must be built before `mkdocs build` runs. In CI, that means calling `maturin build` before `mkdocs build`. In local development, the extension is already present after `maturin develop`.

## Similar Patterns & Reuse

**Existing narrative content:**

- `SETUP_GUIDE.md` (`/home/gustavo/skmob2/SETUP_GUIDE.md`) — contains installation steps and troubleshooting that can be adapted verbatim into `docs/getting-started.md`. The Getting Started page is not new content; it is a reorganisation of this file.
- `BENCHMARK_RESULTS.md` (`/home/gustavo/skmob2/BENCHMARK_RESULTS.md`) — performance tables that can optionally be linked from the home page or a performance page in a future iteration.
- `CLAUDE.md` column auto-detection table — the table of accepted column names per semantic role is directly usable as a reference section on the Getting Started page.

**Existing docstring style:** All public functions use NumPy-style docstrings with `Parameters`, `Returns`, `Raises`, and `Examples` sections (confirmed in `jump_lengths.py:22-65`, `radius_of_gyration.py:21-63`, `od.py:42-62`, `_common.py:39-116`). The `mkdocstrings` `python` handler with `docstring_style: numpy` will render these correctly without reformatting.

**Existing GitHub Actions pattern:** `CI.yml` (`/home/gustavo/skmob2/.github/workflows/CI.yml:1-187`) uses `actions/checkout@v6`, `actions/setup-python@v6`, and `astral-sh/setup-uv@v7`. The docs workflow should reuse the same action versions for consistency.

## Implementation Strategy

### Step 1 — Add docs dependencies to pyproject.toml

**Before:** `/home/gustavo/skmob2/pyproject.toml:20-36` has optional-dependency groups `fitting`, `diversity`, `dev`, `dev-skmob`, `dev-movingpandas`.

**After:** A new `[project.optional-dependencies]` entry `docs` is added:

```
docs = [
    "mkdocs>=1.6",
    "mkdocs-material>=9.5",
    "mkdocstrings[python]>=0.25",
]
```

This keeps docs tooling installable via `uv sync --extra docs` without polluting the core install.

### Step 2 — Create the mkdocs.yml config

New file at `/home/gustavo/skmob2/mkdocs.yml`. Key decisions:

- `site_name: skmob2`
- `docs_dir: docs/src` (see Step 3 for why a `src/` subdirectory is used)
- `theme: name: material` with `palette` for light/dark toggle
- `plugins:` block enabling `search` and `mkdocstrings` with `python` handler, `docstring_style: numpy`
- `nav:` declaring Home, Getting Started, and API Reference sections
- `docs/features/` is NOT listed in `nav:` and therefore will not appear in the built site

The `docs_dir` is set to `docs/src/` rather than `docs/` directly because `docs/features/` already contains planning files that should not be surfaced. Keeping MkDocs source pages in `docs/src/` cleanly separates published content from internal documents.

### Step 3 — Create the docs/src/ directory structure

New files to create:

```
docs/src/
  index.md               -- home page: one-paragraph description + install snippet
  getting-started.md     -- derived from SETUP_GUIDE.md; covers install, maturin develop, first call
  api/
    index.md             -- brief intro to the API reference section
    measures.md          -- ::: directives for all 8 public functions
```

The single `api/measures.md` page uses one `:::` block per public function. Example:

```
## jump_lengths

::: skmob2.measures.jump_lengths.jump_lengths
    options:
      show_source: false
```

Grouping all 8 functions on one page keeps navigation simple at this stage. When the measure count grows, splitting by module (one page per file) is straightforward.

### Step 4 — Create the GitHub Actions docs workflow

New file at `/home/gustavo/skmob2/.github/workflows/docs.yml`.

The job sequence for a push to `main`:

1. `actions/checkout@v6`
2. `astral-sh/setup-uv@v7`
3. `actions/setup-python@v6` with `python-version: '3.x'`
4. Install Rust toolchain (required because mkdocstrings imports the package, which needs `_core.*.so`)
5. `uv sync --extra docs` — installs MkDocs toolchain
6. `maturin develop` — builds the Rust extension into the venv so mkdocstrings can import `skmob2`
7. `uv run mkdocs build --strict` — validates the site
8. On push to `main` only: `uv run mkdocs gh-deploy --force` — pushes `site/` to the `gh-pages` branch

For pull requests, steps 1-7 run (build-only, no deploy) to catch documentation regressions.

The workflow needs `permissions: contents: write` on the deploy step to push to `gh-pages`. The CI.yml workflow uses `contents: read` only; the docs workflow must declare its own permissions block.

### Step 5 — Wire up local dev commands

Add a note to `CLAUDE.md` (the project instructions file) documenting the new commands:

```bash
# Install docs toolchain
uv sync --extra docs

# Serve locally with live reload
uv run mkdocs serve

# Build and validate (strict mode)
uv run mkdocs build --strict
```

No shell scripts are created. `uv run` is the standard invocation pattern consistent with the existing project.

## Trade-Offs

**Single `api/measures.md` vs one page per module:** One page is simpler to navigate at the current scale (8 functions). The downside is that the page will grow long as the library expands. Splitting is trivial later: create one `.md` file per measure module and move the `:::` directives.

**`docs/src/` subdirectory vs `docs/` as MkDocs root:** Using `docs/src/` prevents `docs/features/` planning files from appearing in the site nav. The alternative — listing `docs/features/` in `.gitignore` or a `mkdocs.yml` `exclude_docs:` glob — is more fragile and less obvious to future contributors. The `src/` nesting is a one-time structural cost.

**Building the Rust extension in CI docs job:** mkdocstrings must import `skmob2` to resolve docstrings, which requires `_core.*.so` to exist. This means the docs CI job is slower than a pure-Python docs build. The alternative — mocking `_core` during the docs build — is complex and fragile. Accepting the build cost is the correct trade-off.

**No versioned docs (mike):** The `mike` plugin for version-stamped documentation would add complexity (a `versions.json` file on `gh-pages`, extra CI logic). The library is pre-1.0 and has one stable branch; versioned docs are premature. This can be added later without structural changes.

## Rejected Approaches

**Sphinx:** The Python ecosystem default for C-extension libraries. Rejected because: MkDocs + Material produces better-looking sites with less configuration; mkdocstrings handles NumPy-style docstrings natively; the project has no existing `.rst` content; and the team's stated preference is MkDocs.

**Placing MkDocs pages directly in `docs/`:** Simpler directory structure, but the existing `docs/features/` planning subdirectory would appear in the `nav` unless explicitly excluded. MkDocs `exclude_docs:` requires a glob pattern and is easy to get wrong. Moving MkDocs source to `docs/src/` is a cleaner separation of concerns.

**Auto-generating one page per module with `gen-files` + `literate-nav`:** The `mkdocs-gen-files` and `mkdocs-literate-nav` plugins can dynamically create one page per module. This is powerful for large libraries but adds two extra plugins and a Python script. At 6 module files, manually written `:::` directives in a single `api/measures.md` are easier to understand and maintain.

**Mocking `skmob2._core` during docs build:** Would avoid the Rust build step in CI. Rejected because it requires maintaining a stub module that mirrors the real Rust API; any drift becomes a silent documentation bug. Paying the build cost in CI is safer.

**Inline documentation in `CLAUDE.md` only:** The project already uses `CLAUDE.md` for agent instructions, not user-facing documentation. Rejected because `CLAUDE.md` is not a substitute for a searchable, rendered, publicly hosted API reference.

## Assumptions & Open Questions

- **GitHub Pages is enabled** on the repository. If it is not, the deploy step in `docs.yml` will fail silently or with a permissions error. The repository owner must enable GitHub Pages (Settings > Pages > Source: gh-pages branch) before the first deploy runs.
- **The `gh-pages` branch does not already exist** with conflicting content. If it does, `mkdocs gh-deploy --force` will overwrite it.
- **mkdocstrings can successfully import `skmob2`** after `maturin develop` in CI. This assumption holds as long as the Rust toolchain version in the docs CI job is compatible with the PyO3 version in `Cargo.toml:11` (`pyo3 = "0.28.2"`).
- **The `activity.py` and `individual.py` modules have explicit `import pandas`** (`activity.py:9`, `individual.py:9`). These imports violate the stated Narwhals-only rule in CLAUDE.md but exist in the current codebase. mkdocstrings will import these files and will require pandas to be installed in the docs build environment. The `docs` optional-dependency group should include `pandas` (it is already a core dependency in `pyproject.toml:10`, so `uv sync` will install it automatically).
- **Python version for the docs CI job:** The docs workflow uses `python-version: '3.x'` (latest stable). This is consistent with `CI.yml` and will resolve to Python 3.13+ by the time this runs. Narwhals and mkdocstrings both support Python 3.13.

## Code That Could Be Refactored *(informational)*

- `/home/gustavo/skmob2/skmob2/measures/activity.py:9` and `individual.py:9` — both `import pandas as pd` directly, violating the Narwhals-only rule documented in `CLAUDE.md`. This is not a blocker for the docs feature but should be addressed in a follow-up. If mkdocstrings triggers an import error because pandas is absent, adding `pandas` to the `docs` extras group is a safe workaround.
- `/home/gustavo/skmob2/SETUP_GUIDE.md` — contains installation instructions that partially duplicate what the Getting Started page will contain. Once `docs/src/getting-started.md` exists, `SETUP_GUIDE.md` could be removed or reduced to a one-liner pointing at the docs site.
- `/home/gustavo/skmob2/BENCHMARK_RESULTS.md` and `/home/gustavo/skmob2/BENCHMARK_RESULTS.md` — these are good candidates for a `docs/src/performance.md` page in a future iteration.
- `/home/gustavo/skmob2/skmob2/measures/mobility_laws.py:12` — `log_truncated_powerlaw` has no `Examples` section in its docstring; it will render correctly but the API page will lack a usage example for this function.

## Proposed Next Steps

1. Add the `docs` optional-dependency group to `/home/gustavo/skmob2/pyproject.toml` under `[project.optional-dependencies]`.
2. Create `/home/gustavo/skmob2/mkdocs.yml` with the configuration described in Step 2.
3. Create the `docs/src/` directory and the three page files (`index.md`, `getting-started.md`, `api/index.md`, `api/measures.md`) described in Step 3.
4. Run `uv sync --extra docs && maturin develop && uv run mkdocs build --strict` locally to verify zero warnings.
5. Create `/home/gustavo/skmob2/.github/workflows/docs.yml` as described in Step 4.
6. Enable GitHub Pages on the repository (Settings > Pages > Source: gh-pages branch).
7. Push to `main` and confirm the docs workflow deploys successfully.
