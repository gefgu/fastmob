# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`skmob2` is a high-performance reimplementation of the [`skmob`](https://github.com/scikit-mobility/scikit-mobility) mobility-analysis library. It exposes the same measure API but replaces the Python/pandas internals with a Rust extension (via PyO3) for compute-heavy kernels, and wraps the Python layer with [Narwhals](https://narwhals-dev.github.io/) so any eager dataframe (pandas, polars, …) is accepted as input.

## Architecture

```
src/lib.rs           ← Rust extension compiled to skmob2/_core.*.so
skmob2/_core         ← compiled artifact (do not edit manually)
skmob2/measures/     ← Python measure implementations; each calls into _core
skmob2/__init__.py   ← re-exports public API
tests/correctness/   ← correctness tests (no optional deps required)
tests/benchmarks/    ← pytest-benchmark tests comparing skmob vs skmob2
```

**Data flow for a measure (e.g. `jump_lengths`):**
1. Python wrapper (`skmob2/measures/individual.py`) receives any native dataframe.
2. Narwhals wraps it into a backend-agnostic `nw.DataFrame`.
3. Column names are auto-detected from a priority list (see below).
4. Data is sorted by datetime, split per user, then handed as plain Python lists to the Rust function.
5. Rust computes Haversine distances via the `geo` crate and returns a `Vec<f64>`.
6. Python assembles a result dataframe in the original backend.

## Build & develop

Always use `uv` for python installation procedures.

Requires: Rust toolchain + `maturin` + Python ≥ 3.8.

```bash
# Development build (editable install of the Rust extension)
maturin develop

# Release build (produces a wheel in dist/)
maturin build --release
```

After `maturin develop` the compiled `.so` is placed directly in `skmob2/`, so the package is importable from the repo root without a pip install.

## Test environment setup

First-time setup (creates `.venv` at repo root, builds the Rust extension, installs all dev deps):

```bash
bash tests/setup_env.sh
source .venv/bin/activate
```

Dev dependencies (`pytest`, `pytest-benchmark`, `skmob`, `polars`, `tqdm`) are declared under `[project.optional-dependencies] dev` in `pyproject.toml` and are not required to use the library.

## Tests

```bash
# Correctness only — fast, no network, no optional deps (default)
bash tests/run_correctness.sh

# Include skmob-comparison tests (requires skmob installed)
bash tests/run_correctness.sh -m skmob

# Or run directly with pytest
pytest tests/correctness/ -m "not skmob"
```

## Documentation

```bash
# Install docs toolchain
uv sync --extra docs

# Serve locally with live reload
uv run mkdocs serve

# Build and validate (strict mode, zero warnings)
uv run mkdocs build --strict
```

MkDocs source pages live in `docs/src/`. The `docs/features/` subdirectory holds internal planning files and is intentionally excluded from the published site nav.

## Benchmarks

```bash
# Benchmark table printed to stdout
bash tests/run_benchmarks.sh

# Save JSON report
bash tests/run_benchmarks.sh --benchmark-json=results.json

# Save named snapshot for later comparison
bash tests/run_benchmarks.sh --benchmark-save=baseline

# Compare two snapshots
pytest-benchmark compare baseline 0001
```

Benchmarks are parametrized over three dataset sizes (1k / 10k / 100k / 1M / 4M rows) and run both `skmob` and `skmob2` side by side. The Brightkite check-in dataset (~4M rows) is downloaded on first run and cached to `tests/shared/data/`.

## Profiling

```bash
# CPU flamegraph with py-spy and native Rust frames
bash tests/run_py_spy_profiles.sh --rows 10000 --workload radius_of_gyration

# Memory flamegraph with pytest-memray and native Rust frames
bash tests/run_memray_profiles.sh --rows 10000 --workload radius_of_gyration

# Direct pytest-memray usage for one workload
pytest tests/profiling/test_brightkite_memray.py::test_memray_brightkite_workload \
  --profile-workload radius_of_gyration \
  --profile-rows 10000 \
  --memray \
  --native \
  --memray-bin-path .profiles/memray/manual
```

Profiling outputs are written to `.profiles/py-spy/` and `.profiles/memray/`. Run `maturin develop` first when invoking pytest directly so `skmob2._core` and native symbols are available.

## Narwhals API notes

When writing new measures, use `nw.from_native(traj, eager_only=True)` to accept any dataframe. Use `.to_native()` to return a result in the caller's original backend.

**Narwhals-only enforcement:** `import pandas` and `import polars` are **forbidden** inside any file under `skmob2/`. All DataFrame operations must go through the Narwhals API. This keeps every function backend-agnostic (pandas, polars, and any future backend). Dropping this guarantee for a specific function requires **explicit written approval from the user**; it is not a judgment call.

Preserve `backend=nw_df.implementation` when constructing output dicts so the result backend matches the input.

## Column name conventions

The library auto-detects required columns by scanning a priority list:

| Semantic role | Accepted names (in order) |
|---|---|
| datetime | `datetime`, `check-in_time`, `timestamp`, `time` |
| latitude | `lat`, `latitude` |
| longitude | `lng`, `lon`, `longitude` |
| user ID (optional) | `uid`, `user`, `user_id` |

If a user-ID column is absent, the entire dataframe is treated as a single individual.

## Adding a new measure

1. Add the Rust kernel to `src/lib.rs` and expose it via `m.add_function(...)` in the `_core` pymodule.
2. Run `maturin develop` to rebuild.
3. Create the Python wrapper in `skmob2/measures/individual.py` (or a new file), following the pattern of `jump_lengths`.
4. Re-export from `skmob2/measures/__init__.py` and `skmob2/__init__.py`.
5. Add a correctness test in `tests/correctness/test_individual.py` and a benchmark in `tests/benchmarks/bench_individual.py`.


# Project Structure

- Each functionality should have its own small file, like jump lenghts belong to the jump_lenghts.py file. 
- Each file should have a mirror in the test folder that tests its functionality.
