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
benchmarks/    ← pytest-benchmark tests comparing skmob vs skmob2
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
# Normal development and correctness tests
bash scripts/setup_env.sh
source .venv/bin/activate
```

Dev dependencies (`pytest`, `pytest-benchmark`, `skmob`, `polars`, `tqdm`) are declared under `[project.optional-dependencies] dev` in `pyproject.toml` and are not required to use the library.

## Tests

Use the shell scripts under `tests/` for normal verification. They activate `.venv`, rebuild `skmob2._core` when needed, and forward any extra arguments to pytest or the underlying tool.

```bash
# Correctness only, excluding skmob comparisons by default
bash scripts/run_correctness.sh

# A focused correctness subset
bash scripts/run_correctness.sh tests/correctness/spatial/test_radius_of_gyration.py -v

# Lint and format checks
bash scripts/run_lint.sh

# Coverage over correctness tests
bash scripts/run_coverage.sh

# Benchmarks
bash scripts/run_benchmarks.sh
```

### skmob comparison correctness tests

Run the `@pytest.mark.skmob` comparison tests from the dedicated `.venv-skmob` environment, not the normal `.venv` and not `scripts/run_correctness.sh`. The normal correctness script intentionally uses `.venv`; that environment carries the modern Shapely stack and `skmob` fails to import there because `shapely.ops.cascaded_union` was removed.

The known-good local skmob comparison stack is Python 3.10 with `scikit-mobility 1.3.1`, `geopandas 0.10.2`, and `Shapely 1.8.5.post1`.

If your shell has Conda active, unset `CONDA_PREFIX` before invoking `maturin`; otherwise `maturin` can refuse to choose between Conda and the virtualenv.

```bash
# Build skmob2._core for the skmob comparison environment.
# Use the normal .venv maturin binary if .venv-skmob does not have maturin installed.
unset CONDA_PREFIX
VIRTUAL_ENV="$PWD/.venv-skmob" \
PATH="$PWD/.venv-skmob/bin:$PATH" \
    .venv/bin/maturin develop

# Run only the skmob comparison correctness tests
.venv-skmob/bin/python -m pytest tests/correctness -m skmob -vv
```

The skmob comparison tests should usually be strict, but the Brightkite tests for `filter`, `distance_straight_line`, and `jump_lengths` intentionally allow a narrow tolerance. `skmob` computes distances with `skmob.utils.gislib.getDistanceByHaversine` and `earthradius = 6371.0`, while `skmob2` uses the Rust `geo::Haversine` kernel. This can move threshold-adjacent filtering points across the speed cutoff and can create metre-scale differences on long jumps. Keep that relaxation limited to those tests unless another comparison shows the same distance-kernel-only cause.

## Documentation

```bash
# Install docs toolchain
uv sync --extra docs

# Serve locally with live reload
uv run zensical serve

# Build
uv run zensical build
```

Documentation source pages live in `docs/src/`. The `docs/features/` subdirectory holds internal planning files and is intentionally excluded from the published site nav. `docs/DESIGN.md` is the Zensical design system reference.

API reference pages under `docs/src/reference/` should begin with a compact summary table immediately after the H1. Use the table shape `API | Description`, link each public object to its generated anchor, and keep descriptions short, factual, and consistent with the object's first docstring sentence when possible. Skip navigation-only index pages.

## Benchmarks

`scripts/run_benchmarks.sh` is environment-aware. It rebuilds `skmob2._core` in `.venv`, runs skmob2 pandas/Polars benchmarks there, runs `@pytest.mark.skmob` comparison benchmarks in `.venv-skmob`, and runs `@pytest.mark.movingpandas` benchmarks only from an environment where `movingpandas` imports. Keep skmob comparisons out of `.venv`; that environment uses the modern Shapely stack and cannot import scikit-mobility.

```bash
# Benchmark table printed to stdout
bash scripts/run_benchmarks.sh

# Save JSON report
bash scripts/run_benchmarks.sh --benchmark-json=results.json

# Save named snapshot for later comparison
bash scripts/run_benchmarks.sh --benchmark-save=baseline

# Run one workload family across all compatible comparison environments
bash scripts/run_benchmarks.sh -k radius_of_gyration

# Compare two snapshots
pytest-benchmark compare baseline 0001
```

Benchmarks are parametrized over five dataset sizes (1k / 10k / 100k / 1M / 4M rows) and run `skmob2`, `skmob`, and movingpandas comparisons where the matching environment is available. When using `--benchmark-json` or `--benchmark-save`, the script suffixes the output names per environment (`skmob2`, `skmob`, `movingpandas`) so repeated pytest invocations do not overwrite each other. The Brightkite check-in dataset (~4M rows) is downloaded on first run and cached to `tests/shared/data/`.

## Profiling

Start with a small row count, then scale to 4M only after the profiler path works. Brightkite workloads are registered in `tests/profiling/brightkite_workloads.py`; use `--workload filter`, `--workload radius_of_gyration`, etc. The Brightkite dataset is cached at `tests/shared/data/loc-brightkite_totalCheckins.txt.gz`.

```bash
# Python CPU and memory profile with Scalene.
bash scripts/run_scalene_profiles.sh --rows 10000 --workload radius_of_gyration --implementation skmob2

# Native Rust sampling profile (Firefox Profiler format) with samply.
# Function scope: samply attaches after data prep, so only the Rust kernel is sampled.
bash scripts/run_samply_profiles.sh --rows 10000 --workload radius_of_gyration
samply load .profiles/samply/skmob2/radius_of_gyration.json.gz

# Compare skmob2 and skmob where a skmob equivalent exists.
bash scripts/run_scalene_profiles.sh --rows 10000 --workload radius_of_gyration --implementation both

# Legacy whole-process profiling, including imports and data loading.
bash scripts/run_samply_profiles.sh --rows 10000 --workload radius_of_gyration --scope full
```

Profiling outputs are written to implementation-specific folders under `.profiles/scalene/` and `.profiles/samply/`. Scalene writes `<workload>.json` plus `<workload>.html` unless reduced output is requested; samply writes `<workload>.json.gz` (Firefox Profiler JSON, viewable via `samply load`). Run `maturin develop` first when invoking profiling modules directly so `skmob2._core` and native symbols are available. samply requires `kernel.perf_event_paranoid <= 1` on Linux (`sudo sysctl kernel.perf_event_paranoid=1`).

### Profiling from constrained Codex/sandbox environments

The wrapper scripts are the preferred path because they activate `.venv`, check tools, rebuild the Rust extension in release mode, and write manifests. If they fail before profiling because the sandbox cannot run Snap `uv`, the `.venv` has no `pip`, or `maturin develop --uv` cannot resolve `uv`, do not spend time reinstalling Python tooling. If `skmob2._core` is already importable, run the profiler modules directly with `.venv/bin/python` and `.venv/bin/scalene`.

Check the two prerequisites first:

```bash
cat /proc/sys/kernel/perf_event_paranoid
.venv/bin/python -c 'import skmob2._core as c; print(c.__file__)'
```

If `perf_event_paranoid` is greater than `1`, ask the user to run:

```bash
sudo sysctl kernel.perf_event_paranoid=1
```

If the extension import fails, rebuild with the approved local build path:

```bash
env -u CONDA_PREFIX uv run maturin develop --release
```

For a direct Scalene profile of a Brightkite workload:

```bash
.venv/bin/scalene run --profile-only skmob2 --memory --off \
  -o /tmp/filter_4M_scalene.json \
  tests/profiling/brightkite_workloads.py --- \
  --workload filter --rows 4000000 --backend pandas \
  --implementation skmob2 --scalene-function-profile

.venv/bin/python .agents/skills/profile-python-scalene/scripts/reduce_scalene_json.py \
  /tmp/filter_4M_scalene.json --top 25
```

For a direct function-scope Samply profile of the same workload:

```bash
.venv/bin/python scripts/profile_brightkite_samply.py \
  --samply-bin /home/gustavo/.cargo/bin/samply \
  --rows 4000000 --workload filter --implementation skmob2 \
  --backend pandas --scope function --rate 1000 \
  --output-dir /tmp/samply_filter_4M

.venv/bin/python .agents/skills/profile-rust-samply/scripts/reduce_samply_json.py \
  /tmp/samply_filter_4M/skmob2/filter.json.gz --top 20 \
  -o /tmp/filter_4M_samply_reduced.json
```

Use `--scope function` for Samply unless startup, imports, or data loading are the target. Function scope prepares Brightkite first, then attaches to the child process before releasing the workload, so samples represent the API call rather than CSV loading. Use `/tmp` for exploratory profile artifacts unless the user asks to keep them in `.profiles/`.

The Rust build flags in `.cargo/config.toml` must keep each `-C` option as a separate flag/value pair. The correct form is:

```toml
[build]
rustflags = ["-C", "force-frame-pointers=yes", "-C", "symbol-mangling-version=v0"]
```

## Narwhals API notes

When writing new measures, use `nw.from_native(traj, eager_only=True)` to accept any dataframe. Use `.to_native()` to return a result in the caller's original backend.

**Narwhals-only enforcement:** `import pandas` and `import polars` are **forbidden** inside any file under `skmob2/`. All DataFrame operations must go through the Narwhals API. This keeps every function backend-agnostic (pandas, polars, and any future backend). Dropping this guarantee for a specific function requires **explicit written approval from the user**; it is not a judgment call.

Preserve `backend=nw_df.implementation` when constructing output dicts so the result backend matches the input.

## Radius of gyration performance pattern

`skmob2.measures.individual.radius_of_gyration` is the current reference implementation for a high-throughput, low-memory measure. Treat it as the standard to copy when building or refactoring other measures.

What makes it fast:

- **Do not sort the full dataframe unless the metric truly needs row order.** Radius of gyration is order-independent, so the wrapper calls `_prepare_trajectory(..., sort=False)` and preserves the input rows after null dropping and coordinate casting. This avoids the large temporary memory spike caused by sorting all columns.
- **Group with sorted indexes instead of sorted rows.** For user-level results, build a sorted row-index vector and user ranges, then let the Rust kernel read coordinates through those indexes. Sorting a skinny index representation is much cheaper than materializing a fully sorted trajectory dataframe.
- **Filter invalid rows while building grouped indexes.** If a metric ignores rows with null/invalid coordinates, prefer backend-specific Rust helpers that produce valid row indexes and user ranges in one pass (for example `radius_of_gyration_valid_user_indices_*`). Keep a narrow Narwhals fallback only for unsupported user dtypes; do not materialize a cleaned, fully sorted dataframe on the main performance path.
- **Keep Python out of the hot loop.** Python should detect columns, prepare arrays, choose the backend route, and assemble the final dataframe. Per-row grouping, range validation, index validation, and numeric computation belong in Rust.
- **Use backend-specific zero-copy-ish routes.** Polars-backed Narwhals data should go through Arrow (`to_arrow()` + Arrow Rust kernels). Other eager backends should go through NumPy (`to_numpy()` + NumPy Rust kernels). Avoid converting entire columns to Python lists on performance paths.
- **Return kernel results in the same array backend.** Arrow routes should return Arrow-compatible arrays and NumPy routes should return NumPy arrays, then the Narwhals wrapper can build `nw.from_dict(..., backend=df.implementation)` without converting result values across backends.
- **Batch across all users in one Rust call.** Build all user ranges once and call the kernel once for the whole dataframe. Avoid per-user Python calls into Rust; boundary crossings are cheap individually but expensive at scale.
- **Extract result labels vectorized when possible.** For NumPy-backed data, gather user labels with array indexing (`uid_series.to_numpy()[start_indices].tolist()`). For Arrow-backed data, keep Arrow conversion to one call and gather only the needed unique labels.
- **Preserve backend and API compatibility.** Keep Narwhals as the public dataframe boundary, return the caller's backend with `backend=df.implementation`, and keep required trajectory column handling consistent with the rest of the API even when a metric does not use chronological order.

When adding a new measure, first ask whether the metric is order-dependent:

- If order-dependent, use the shared preparation pipeline with sorting, then pass contiguous ranges to Rust.
- If order-independent but grouped by user, prefer the radius-of-gyration pattern: clean only what the metric requires, build valid sorted indexes/ranges without sorting the full dataframe, and use an indexed Rust kernel.
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

## Variable-length-per-user output kernel pattern

Measures that emit a variable number of output rows per user (e.g. `location_frequency`,
`frequency_rank`, `recency_rank`) use a **flat output + user-boundary** convention:

- The Rust kernel returns `(out_vals..., out_user_starts: Vec<usize>, out_user_ends: Vec<usize>)`.
- `out_user_starts[i]` / `out_user_ends[i]` are slice boundaries into the flat output arrays for user `i`.
- The Python wrapper reconstructs uid labels:
  ```python
  for i, (s, e) in enumerate(zip(out_starts.tolist(), out_ends.tolist())):
      uid_vals_all.extend([uid_values[i]] * (e - s))
  ```
- For `uid_col=None`, `_build_indexed_user_ranges_fast` emits a single range `[0, N)`; the uid column is omitted from the result dict.
- User-boundary arrays are always NumPy `usize` even in the Arrow route — they are index metadata, not data values.
- Rust sorts per-user results before returning (e.g. by `-count, lat, lng` for frequency measures) so ranks are implicit (1, 2, … in output order) and computed in Python.
- See `src/location_frequency.rs` and `src/recency_rank.rs` as reference implementations. Compare with `src/spatial_counts.rs` (scalar-per-user) and `src/jump_lengths.rs` (single flat value column + boundaries).

## Adding a new measure

When adding a measure, choose the output pattern based on cardinality:

- **Scalar per user** (e.g. `radius_of_gyration`, `number_of_locations`): use `_dispatch_kernel`, return one value per user. See `src/spatial_counts.rs`.
- **Variable rows per user** (e.g. `location_frequency`, `recency_rank`): return flat arrays + `(out_user_starts, out_user_ends)` usize arrays from Rust. See `src/location_frequency.rs`.

1. Add the Rust kernel to `src/lib.rs` and expose it via `m.add_function(...)` in the `_core` pymodule.
2. Run `maturin develop` to rebuild.
3. Create the Python wrapper in `skmob2/measures/individual.py` (or a new file), following the pattern of `jump_lengths`.
4. Re-export from `skmob2/measures/__init__.py` and `skmob2/__init__.py`.
5. Add a correctness test in `tests/correctness/test_individual.py` and a benchmark in `benchmarks/bench_individual.py`.


# Project Structure

- Each functionality should have its own small file, like jump lenghts belong to the jump_lenghts.py file. 
- Each file should have a mirror in the test folder that tests its functionality.
