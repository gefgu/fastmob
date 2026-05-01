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

## Formatting

Leave formatter output as the tools produce it. If `cargo fmt`, Ruff, or another project formatter rewrites nearby code or import ordering, keep that formatting instead of manually undoing it to minimize diff shape. Prefer a clean, tool-formatted tree over hand-preserving previous style.

## Completion workflow

At the end of each completed assigned task, review the final diff, run the relevant verification, and create a focused git commit for the completed work. Do not include unrelated user changes in that commit. If the auto-review finds a blocking issue, fix it and re-check before committing; if verification cannot be run, note that in the final response.

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

# Or run directly with pytest
pytest tests/correctness/ -m "not skmob"
```

### skmob comparison correctness tests

Run the `@pytest.mark.skmob` comparison tests from the dedicated `.venv-skmob` environment, not the normal `.venv`. `scikit-mobility` depends on older geo packages; the known-good local stack is Python 3.10 with `scikit-mobility 1.3.1`, `geopandas 0.10.2`, and `Shapely 1.8.5.post1`. Shapely 2.x breaks `skmob` imports because `shapely.ops.cascaded_union` was removed.

If your shell has Conda active, unset `CONDA_PREFIX` before invoking `maturin`; otherwise `maturin` can refuse to choose between Conda and the virtualenv.

```bash
# Build skmob2._core for the skmob comparison environment
source .venv-skmob/bin/activate
unset CONDA_PREFIX
maturin develop

# Run only the skmob comparison correctness tests
python -m pytest tests/correctness -m skmob -vv
```

Avoid `bash tests/run_correctness.sh -m skmob` for these comparisons unless it has been updated to use `.venv-skmob`; that helper currently activates `.venv`, whose modern Shapely stack is intended for normal development and can make `skmob` fail to import.

The skmob comparison tests should usually be strict, but the Brightkite tests for `filter`, `distance_straight_line`, and `jump_lengths` intentionally allow a narrow tolerance. `skmob` computes distances with `skmob.utils.gislib.getDistanceByHaversine` and `earthradius = 6371.0`, while `skmob2` uses the Rust `geo::Haversine` kernel. This can move threshold-adjacent filtering points across the speed cutoff and can create metre-scale differences on long jumps. Keep that relaxation limited to those tests unless another comparison shows the same distance-kernel-only cause.

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
# CPU flamegraph and Speedscope JSON with py-spy and native Rust frames.
# By default, only the target function call and result materialization are profiled.
bash tests/run_py_spy_profiles.sh --rows 10000 --workload radius_of_gyration

# Memory flamegraph with memray and native Rust frames; defaults to function-only profiling.
bash tests/run_memray_profiles.sh --rows 10000 --workload radius_of_gyration

# Compare skmob2 and skmob where a skmob equivalent exists.
bash tests/run_py_spy_profiles.sh --rows 10000 --workload radius_of_gyration --implementation both

# Legacy whole-process profiling, including imports and data loading.
bash tests/run_memray_profiles.sh --rows 10000 --workload radius_of_gyration --scope full
```

Profiling outputs are written to implementation-specific folders under `.profiles/py-spy/` and `.profiles/memray/`. Py-spy writes both `<workload>.svg` and `<workload>.speedscope.json`. Run `maturin develop` first when invoking profiling modules directly so `skmob2._core` and native symbols are available.

## Narwhals API notes

When writing new measures, use `nw.from_native(traj, eager_only=True)` to accept any dataframe. Use `.to_native()` to return a result in the caller's original backend.

**Narwhals-only enforcement:** `import pandas` and `import polars` are **forbidden** inside any file under `skmob2/`. All DataFrame operations must go through the Narwhals API. This keeps every function backend-agnostic (pandas, polars, and any future backend). Dropping this guarantee for a specific function requires **explicit written approval from the user**; it is not a judgment call.

Preserve `backend=nw_df.implementation` when constructing output dicts so the result backend matches the input.

## Radius of gyration performance pattern

`skmob2.measures.spatial.radius_of_gyration` is the current reference implementation for a high-throughput, low-memory measure. Treat it as the standard to copy when building or refactoring other measures.

What makes it fast:

- **Do not sort the full dataframe unless the metric truly needs row order.** Radius of gyration is order-independent, so the wrapper calls `_prepare_trajectory(..., sort=False)` and preserves the input rows after null dropping and coordinate casting. This avoids the large temporary memory spike caused by sorting all columns.
- **Group with sorted indexes instead of sorted rows.** For user-level results, build a sorted row-index vector and user ranges, then let the Rust kernel read coordinates through those indexes. Sorting a skinny index representation is much cheaper than materializing a fully sorted trajectory dataframe.
- **Keep Python out of the hot loop.** Python should detect columns, prepare arrays, choose the backend route, and assemble the final dataframe. Per-row grouping, range validation, index validation, and numeric computation belong in Rust.
- **Use backend-specific zero-copy-ish routes.** Polars-backed Narwhals data should go through Arrow (`to_arrow()` + Arrow Rust kernels). Other eager backends should go through NumPy (`to_numpy()` + NumPy Rust kernels). Avoid converting entire columns to Python lists on performance paths.
- **Batch across all users in one Rust call.** Build all user ranges once and call the kernel once for the whole dataframe. Avoid per-user Python calls into Rust; boundary crossings are cheap individually but expensive at scale.
- **Extract result labels vectorized when possible.** For NumPy-backed data, gather user labels with array indexing (`uid_series.to_numpy()[start_indices].tolist()`). For Arrow-backed data, keep Arrow conversion to one call and gather only the needed unique labels.
- **Preserve backend and API compatibility.** Keep Narwhals as the public dataframe boundary, return the caller's backend with `backend=df.implementation`, and keep required trajectory column handling consistent with the rest of the API even when a metric does not use chronological order.

When adding a new measure, first ask whether the metric is order-dependent:

- If order-dependent, use the shared preparation pipeline with sorting, then pass contiguous ranges to Rust.
- If order-independent but grouped by user, prefer the radius-of-gyration pattern: clean without sorting, sort indexes/ranges only, and use an indexed Rust kernel.
- If there is no user column, call a single contiguous Rust kernel over the whole coordinate array.

The target shape is: small Narwhals wrapper, no pandas/polars imports, no Python per-row loops on large data, one batched Rust call, explicit validation in Rust, and correctness tests covering pandas and polars.

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
