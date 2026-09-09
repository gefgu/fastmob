# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`fastmob` is a high-performance reimplementation of the [`skmob`](https://github.com/scikit-mobility/scikit-mobility) mobility-analysis library. It exposes the same measure API but replaces the Python/pandas internals with a Rust extension (via PyO3) for compute-heavy kernels, and wraps the Python layer with [Narwhals](https://narwhals-dev.github.io/) so any eager dataframe (pandas, polars, …) is accepted as input.

## Architecture

```
src/lib.rs           ← Rust extension compiled to fastmob/_core.*.so
fastmob/_core         ← compiled artifact (do not edit manually)
fastmob/measures/     ← Python measure implementations; each calls into _core
fastmob/__init__.py   ← re-exports public API
tests/correctness/   ← correctness tests (no optional deps required)
benchmarks/    ← pytest-benchmark tests comparing skmob vs fastmob
```

**Data flow for a measure (e.g. `jump_lengths`):**
1. Python wrapper (`fastmob/measures/individual.py`) receives any native dataframe.
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

After `maturin develop` the compiled `.so` is placed directly in `fastmob/`, so the package is importable from the repo root without a pip install.

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

Dev dependencies are declared in the `dev` dependency group in `pyproject.toml` and are not required to use the library. External-library comparison stacks live under `benchmarks/environments/` and must stay isolated from the normal development environment.

## Tests

Use the shell scripts under `tests/` for normal verification. They activate `.venv`, rebuild `fastmob._core` when needed, and forward any extra arguments to pytest or the underlying tool.

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
# Build fastmob._core for the skmob comparison environment.
# Use the normal .venv maturin binary if .venv-skmob does not have maturin installed.
unset CONDA_PREFIX
VIRTUAL_ENV="$PWD/.venv-skmob" \
PATH="$PWD/.venv-skmob/bin:$PATH" \
    .venv/bin/maturin develop

# Run only the skmob comparison correctness tests
.venv-skmob/bin/python -m pytest tests/correctness -m skmob -vv
```

The skmob comparison tests should usually be strict, but the Brightkite tests for `filter`, `distance_straight_line`, and `jump_lengths` intentionally allow a narrow tolerance. `skmob` computes distances with `skmob.utils.gislib.getDistanceByHaversine` and `earthradius = 6371.0`, while `fastmob` uses the Rust `geo::Haversine` kernel. This can move threshold-adjacent filtering points across the speed cutoff and can create metre-scale differences on long jumps. Keep that relaxation limited to those tests unless another comparison shows the same distance-kernel-only cause.

## Release process

`.github/workflows/CI.yml`'s wheel-build jobs (`linux-wheels`, `musllinux-wheels`, `macos-wheels`, `macos-x86_64-wheels`, `windows-wheels`, `sdist`) only run on a `vX.Y.Z` tag push or a manual `workflow_dispatch`; ordinary PRs and pushes to `main` only run the cheap `test` job. This keeps the expensive macOS (10x Linux runner cost) and Windows (2x) runners off the normal development loop. Use `workflow_dispatch` with the `platform` input (`all`, `linux`, `musllinux`, `macos`, `windows`, `sdist`) from the Actions tab to debug a single platform's wheel build without running the whole matrix.

Before tagging a release, validate everything locally so the tag-triggered run succeeds on the first try instead of idling in the runner queue on a retry:

```bash
# Fast local Linux-CI-equivalent check (lint + pytest, via act + Docker)
bash scripts/act_test.sh
# or, without act:
bash scripts/run_lint.sh
bash scripts/run_correctness.sh

# Mirror CI's Linux wheel/sdist jobs locally (manylinux + musllinux x86_64
# via Docker, aarch64 via zig cross-compile, sdist)
bash scripts/build_release_wheels.sh
```

macOS and Windows wheels cannot be built locally (no genuine Apple/Microsoft toolchain on this machine) and stay CI-only.

To cut a release:

1. Bump the version in `fastmob-py/Cargo.toml` — that's the authoritative version; `pyproject.toml` reads it dynamically via `[tool.maturin] manifest-path`. Commit the bump.
2. `git tag vX.Y.Z && git push origin main --tags`.
3. The tag push triggers the full wheel matrix, then `release` (gated on `startsWith(github.ref, 'refs/tags/v')`) generates artifact attestations, creates the GitHub release, and publishes to PyPI via OIDC Trusted Publishing — no stored token, and it must run inside GitHub Actions regardless of what was built locally.

## Documentation

```bash
# Install docs toolchain
uv sync --group docs

# Serve locally with live reload
uv run --group docs zensical serve

# Build
uv run --group docs zensical build
```

Documentation source pages live in `docs/src/`. The `docs/features/` subdirectory holds internal planning files and is intentionally excluded from the published site nav. `docs/DESIGN.md` is the Zensical design system reference.

### Notebook documentation pages

Notebooks under `docs/src/` are converted into normal Markdown pages before
the docs build. The converter preserves notebook prose, code blocks, and
captured outputs, writes the Markdown and output assets beside the source
notebook, and adds a Binder badge linking to the original notebook:

```bash
uv run --group docs python scripts/build_notebook_docs.py
```

For local serving, use `bash scripts/serve_docs.sh`; it regenerates notebook
pages first and serves Zensical at `http://127.0.0.1:5178` by default.

Generated notebook `.md` files and `*_files/` directories are ignored and
must not be edited directly. Run the converter before `pytest tests/docs` or
`uv run --group docs zensical build`; docs CI runs it automatically.

API reference pages under `docs/src/reference/` should begin with a compact summary table immediately after the H1. Use the table shape `API | Description`, link each public object to its generated anchor, and keep descriptions short, factual, and consistent with the object's first docstring sentence when possible. Skip navigation-only index pages.

### Measures sidebar

The Measures dropdown uses Zensical's native nested `nav` list plus the `navigation.indexes` theme feature. Keep the overview page first in the section so it becomes the clickable index; do not use the unsupported `children = [...]` field. Measure category icons come from each page's `icon: lucide/...` front matter.

Do not replace this with a client-side DOM rewrite: external scripts miss Zensical's initial navigation event and are unreliable with instant navigation. `mkdocs-awesome-nav`/`.nav.yml` was tested and ignored by Zensical. Validate any change with `uv run --group docs zensical build --strict` and `uv run --group docs --with pytest pytest tests/docs -q`.

## Benchmarks

`scripts/run_benchmarks.sh` is environment-aware. It rebuilds `fastmob._core` in `.venv`, runs fastmob pandas/Polars benchmarks there, runs `@pytest.mark.skmob` comparison benchmarks in `.venv-skmob`, and runs `@pytest.mark.movingpandas` benchmarks only from an environment where `movingpandas` imports. Keep skmob comparisons out of `.venv`; that environment uses the modern Shapely stack and cannot import scikit-mobility.

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

Benchmarks are parametrized over five dataset sizes (1k / 10k / 100k / 1M / 4M rows) and run `fastmob`, `skmob`, and movingpandas comparisons where the matching environment is available. When using `--benchmark-json` or `--benchmark-save`, the script suffixes the output names per environment (`fastmob`, `skmob`, `movingpandas`) so repeated pytest invocations do not overwrite each other. The Brightkite check-in dataset (~4M rows) is downloaded on first run and cached to `tests/shared/data/`.

## Profiling

Python measure wrappers must not contain ad-hoc timing state or `print` calls.
For opt-in, stage-level diagnostics, use `fastmob.utils._profiling.StageTimer`
with a clearly named `FASTMOB_PROFILE_*` environment variable. The helper keeps
profiling output on stderr and is a no-op unless that variable is set.

Start with a small row count, then scale to 4M only after the profiler path works. Brightkite workloads are registered in `tests/profiling/brightkite_workloads.py`; use `--workload filter`, `--workload radius_of_gyration`, etc. The Brightkite dataset is cached at `tests/shared/data/loc-brightkite_totalCheckins.txt.gz`.

```bash
# Python CPU and memory profile with Scalene.
bash scripts/run_scalene_profiles.sh --rows 10000 --workload radius_of_gyration --implementation fastmob

# Native Rust sampling profile (Firefox Profiler format) with samply.
# Function scope: samply attaches after data prep, so only the Rust kernel is sampled.
bash scripts/run_samply_profiles.sh --rows 10000 --workload radius_of_gyration
samply load .profiles/samply/fastmob/radius_of_gyration.json.gz

# Compare fastmob and skmob where a skmob equivalent exists.
bash scripts/run_scalene_profiles.sh --rows 10000 --workload radius_of_gyration --implementation both

# Legacy whole-process profiling, including imports and data loading.
bash scripts/run_samply_profiles.sh --rows 10000 --workload radius_of_gyration --scope full
```

Profiling outputs are written to implementation-specific folders under `.profiles/scalene/` and `.profiles/samply/`. Scalene writes `<workload>.json` plus `<workload>.html` unless reduced output is requested; samply writes `<workload>.json.gz` (Firefox Profiler JSON, viewable via `samply load`). Run `maturin develop` first when invoking profiling modules directly so `fastmob._core` and native symbols are available. samply requires `kernel.perf_event_paranoid <= 1` on Linux (`sudo sysctl kernel.perf_event_paranoid=1`).

### Profiling from constrained Codex/sandbox environments

The wrapper scripts are the preferred path because they activate `.venv`, check tools, rebuild the Rust extension in release mode, and write manifests. If they fail before profiling because the sandbox cannot run Snap `uv`, the `.venv` has no `pip`, or `maturin develop --uv` cannot resolve `uv`, do not spend time reinstalling Python tooling. If `fastmob._core` is already importable, run the profiler modules directly with `.venv/bin/python` and `.venv/bin/scalene`.

Check the two prerequisites first:

```bash
cat /proc/sys/kernel/perf_event_paranoid
.venv/bin/python -c 'import fastmob._core as c; print(c.__file__)'
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
.venv/bin/scalene run --profile-only fastmob --memory --off \
  -o /tmp/filter_4M_scalene.json \
  tests/profiling/brightkite_workloads.py --- \
  --workload filter --rows 4000000 --backend pandas \
  --implementation fastmob --scalene-function-profile

.venv/bin/python .agents/skills/profile-python-scalene/scripts/reduce_scalene_json.py \
  /tmp/filter_4M_scalene.json --top 25
```

For a direct function-scope Samply profile of the same workload:

```bash
.venv/bin/python scripts/profile_brightkite_samply.py \
  --samply-bin /home/gustavo/.cargo/bin/samply \
  --rows 4000000 --workload filter --implementation fastmob \
  --backend pandas --scope function --rate 1000 \
  --output-dir /tmp/samply_filter_4M

.venv/bin/python .agents/skills/profile-rust-samply/scripts/reduce_samply_json.py \
  /tmp/samply_filter_4M/fastmob/filter.json.gz --top 20 \
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

**Narwhals-only enforcement:** `import pandas` and `import polars` are **forbidden** inside any file under `fastmob/`. All DataFrame operations must go through the Narwhals API. This keeps every function backend-agnostic (pandas, polars, and any future backend). Dropping this guarantee for a specific function requires **explicit written approval from the user**; it is not a judgment call.

Preserve `backend=nw_df.implementation` when constructing output dicts so the result backend matches the input.

### Core dataframe hierarchy wrappers

Files under `fastmob/core/*_dataframe.py` must not extract Narwhals columns with direct `.to_numpy()` calls. These wrappers should follow the same shape as high-throughput functions such as `jump_lengths`:

- Use Narwhals for validation, joins, sorting, timestamp normalization, and final dataframe assembly.
- Use `TrajectoryDispatcher.get_ops(df)["extract_data"]` to pass backend-native column buffers into Rust; Polars/PyArrow-backed data should route through Arrow, while pandas/NumPy-backed data may route through NumPy inside the dispatcher.
- Move stateful, non-dataframe work into Rust kernels with PyO3 adapters that accept the input buffer format and dispatch NumPy vs Arrow internally.
- Keep Python out of per-row scans, grouped forward/backward fills, trip/tour boundary loops, and distance attribution loops.
- Do not add direct pandas or polars imports to core wrappers; backend-specific handling belongs in shared dispatch/prep helpers or Rust adapters.

## Radius of gyration performance pattern

`fastmob.measures.individual.radius_of_gyration` is the current reference implementation for a high-throughput, low-memory measure. Treat it as the standard to copy when building or refactoring other measures.

What makes it fast:

- **Do not sort the full dataframe unless the metric truly needs row order.** Radius of gyration is order-independent, so the wrapper avoids the large temporary memory spike caused by sorting all columns. (This implementation pre-dates `TrajectoryDispatcher`; new measures should follow the conventions in "Backend dispatch and null-handling conventions" instead.)
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

> `radius_of_gyration` pre-dates `TrajectoryDispatcher` and is intentionally kept as-is; new measures must follow the conventions below.

## Backend dispatch and null-handling conventions

Two mandatory rules apply to every file under `fastmob/` that calls a Rust kernel.

### Rule 1 — TrajectoryDispatcher (index/uid-metadata builders only)

This rule is superseded for *measure data* (coordinates, timestamps, and any other dense
numeric array that is a direct argument to a Rust kernel) by
["Arrow-only Rust bindings for dense numeric arrays"](#arrow-only-rust-bindings-for-dense-numeric-arrays)
below — new measures should follow that convention, not this one. `TrajectoryDispatcher`
remains the right tool only for the index/uid-metadata builders in `fastmob/utils/_common.py`
(`_build_indexed_user_ranges_fast`, `_build_time_ordered_user_ranges`,
`_build_presorted_user_ends`) and their Rust counterparts (`indexed_user_indices`,
`time_ordered_user_indices`, `presorted_user_starts_ends_*`) — these move small per-user
boundary arrays, not measure data, and the NumPy-vs-Arrow choice there is about avoiding a
copy on the input dataframe's own uid column, not about the output array's numeric type.

Never write `if _is_polars_backed(df): ... else: ...` or `if use_arrow: ...` branching inline.
Use a **module-level `TrajectoryDispatcher`** instance from `fastmob.core.dispatch` instead.
The dispatcher injects `extract_data` automatically and holds all backend-specific ops.

```python
from fastmob.core.dispatch import TrajectoryDispatcher
from fastmob._core import my_kernel_indexed_arrow as _kia, my_kernel_indexed_numpy as _kin
from ..measures._common import _arrow_result_values
import numpy as np

MY_DISPATCHER = TrajectoryDispatcher(
    arrow_ops={
        "kernel_indexed": _kia,
        "unpack": lambda r: (
            np.asarray(_arrow_result_values(r[0]), dtype=np.float64),
            # … one entry per output array
        ),
    },
    numpy_ops={
        "kernel_indexed": _kin,
        "unpack": lambda r: r,
    },
)
```

Inside the public function:

```python
ops       = MY_DISPATCHER.get_ops(df)
use_arrow = MY_DISPATCHER.get_backend_key(df) == "arrow"

lats_data = ops["extract_data"](df.get_column(lat_col))
lngs_data = ops["extract_data"](df.get_column(lng_col))
raw    = ops["kernel_indexed"](lats_data, lngs_data, sorted_indices, ends, ...)
result = ops["unpack"](raw)
```

Do **not** import `_is_polars_backed` in new or refactored files.
Canonical references: `fastmob/preprocessing/_filter.py` and `fastmob/preprocessing/_compress.py`.

### Rule 2 — Indexed path handles nulls natively; no global drop_nulls

Do **not** call `_prepare_trajectory(..., drop_nulls=True)` on the default (non-`sorted`) path.
Instead:

1. Keep only the explicit coordinate cast:
   ```python
   df = df.with_columns(
       nw.col(lat_col).cast(nw.Float64),
       nw.col(lng_col).cast(nw.Float64),
   )
   ```
2. The **Arrow**-indexed Rust binding uses `as_nullable_f64_array` on each coordinate/timestamp
   array, then `arrow_valid_rows(&[&lat, &lng, ...])`, and passes `valid_rows.as_deref()` to
   the core `_indexed_impl`.
3. The **NumPy**-indexed Rust binding passes `None` for `valid_rows`; the core checks
   `is_finite()` on each index before using it.
4. The `presorted=True` fast path retains its assumption that the caller supplies pre-cleaned data;
   `_prepare_trajectory(sort=True)` is acceptable there.

`arrow_valid_rows` is already implemented in `fastmob-py/src/utils/py_helpers.rs`.

### Rust template for adding `valid_rows` to an indexed core function

In `fastmob-core/src/…/<measure>.rs`, add a validity helper and filter per-user indices:

```rust
fn is_valid_indexed_row(
    lats: &[f64], lngs: &[f64], valid_rows: Option<&[bool]>, idx: usize,
) -> bool {
    valid_rows.is_none_or(|v| v[idx]) && lats[idx].is_finite() && lngs[idx].is_finite()
}

// In the per-user loop, replace direct iteration with a validity filter:
let valid: Vec<usize> = user_indices.iter().copied()
    .filter(|&i| is_valid_indexed_row(lats, lngs, valid_rows, i))
    .collect();
```

Add `valid_rows: Option<&[bool]>` after `ends` in the `_indexed_impl` signature.

In `fastmob-py/src/…/<measure>.rs`:
- **NumPy binding** (only for bindings that still keep a NumPy path — new Arrow-only bindings
  skip this step entirely, see the section below): keep `PyReadonlyArray1<f64>` inputs; pass
  `None` to core.
- **Arrow binding**: change `as_f64_array` → `as_nullable_f64_array` for each coordinate /
  timestamp array; add `let valid_rows = arrow_valid_rows(&[&lats, &lngs, ...]);`; pass
  `valid_rows.as_deref()` to core.
- Add `as_nullable_f64_array, arrow_valid_rows` to the `use crate::utils::` import.

See `fastmob-core/src/preprocessing/compress_traj.rs` and
`fastmob-py/src/preprocessing/compress_traj_py.rs` as the canonical Rust reference.

## Arrow-only Rust bindings for dense numeric arrays

New measures (and measures being migrated off Rule 1 above) should use this convention
instead of a NumPy/Arrow dual-dispatch binding, whenever the Rust kernel's inputs and outputs
are dense `f64`/`u64`/`i64`/`u8`/`bool` arrays with no strings, nested, or list types — i.e.
virtually all coordinate, timestamp, and other measure-data arrays. For that shape, NumPy and
Arrow are both just a contiguous buffer: converting a non-null NumPy array to Arrow (or back)
is a buffer-level operation on both sides, not a real copy, so a runtime NumPy-vs-Arrow branch
buys nothing but code complexity — and, measured on `radius_of_gyration`/`jump_lengths` at 4M
rows, actively costs 2.5–4x in wall time versus going Arrow-only end to end (pandas and polars
inputs alike; the polars/Arrow path was the *slower* one before this change, since the dual
dispatch's own branching overhead dominated). This does **not** apply to the index/uid-metadata
builders carved out in Rule 1 above — those stay dual-path deliberately.

### Rust side

- Type `#[pyfunction]` parameters as `pyo3_arrow::PyArray` (aliased `ArrowPyArray` in most
  binding files) directly — not `&Bound<'py, PyAny>` plus a runtime `is_arrow_array()` check
  and dual `.extract::<PyReadonlyArray1<f64>>()` / `.extract::<ArrowPyArray>()` branches.
  PyO3's argument extraction handles the conversion, and `pyo3_arrow::PyArray`'s
  `FromPyObject` implementation is lenient enough to also accept a plain NumPy array
  transparently (verified empirically) — so there is no NumPy-specific code path to write or
  keep, on either side of the FFI boundary.
- Non-nullable inputs (the `presorted` fast path, which assumes pre-cleaned data): extract via
  `as_f64_array(arr, name)` — errors if the array contains nulls.
- Nullable inputs (the `indexed` path): extract via `as_nullable_f64_array(arr, name)`, then
  `arrow_valid_rows(&[&arr1, &arr2, ...])` to get `Option<Vec<bool>>`, and pass
  `valid_rows.as_deref()` to the core `_indexed_impl` exactly as Rule 2 above already
  describes — that null-handling convention is unchanged by this section, only the
  NumPy-vs-Arrow *extraction* branching goes away.
- Get the raw `&[f64]` (or `&[u64]`/`&[i64]`/etc) slice via `arrow_values(&array)`.
- Wrap output via `f64_results_into_arrow`/`u64_results_into_arrow`/`u32_results_into_arrow`/
  `bool_results_into_arrow` (in `fastmob-py/src/utils/py_helpers.rs`), then
  `Py::new(py, array)?.into_any()`. Each binding file keeps its own tiny private
  `arrow_f64_output(py, values)` wrapper around this — that one-off duplication across binding
  files is a pre-existing, accepted repo convention; don't try to consolidate it.
- For the common "presorted contiguous ranges" / "indexed row-order + ends" shapes, reuse the
  shared generic adapters in `fastmob-py/src/adapters/trajectory.rs`:
  `run_presorted_coordinate_arrow<T, F>` / `run_indexed_coordinate_arrow<T, F>` (and their
  timed/group/two-sequence-shaped siblings, where present). Both are generic over the
  closure's return type `T`, so callers with different output shapes — a plain `Vec<f64>`, a
  `(Vec<f64>, Vec<bool>)` validity pair, a fallible `Result<_, String>` — all reuse the same
  extraction/validation/`py.detach` boilerplate and just wrap `T` into Python types themselves
  afterward. Reference implementations:
  `fastmob-py/src/measures/individual/radius_of_gyration.rs` and `jump_lengths.rs`.

### Python side

- Extract measure-data columns via `.to_arrow()` unconditionally, regardless of the input
  dataframe's backend. Do not instantiate a `TrajectoryDispatcher` for this — that class is
  reserved for the index/uid-metadata builders (Rule 1 above).
- **Preserve existing public output-type contracts.** If a public measure function's docstring
  already promises a raw-array return that matches the input backend (e.g. `merge=True` on
  `jump_lengths`/`waiting_times`: "NumPy array for NumPy-backed inputs... PyArrow array for
  Arrow-backed inputs"), keep a thin `TrajectoryDispatcher` instance around purely to call
  `.get_backend_key(df) == "numpy"` at that one output boundary and convert the Arrow result
  back with `.to_numpy(zero_copy_only=False)`. This is presentation logic at the API edge, not
  the kind of routing branch Rule 1 forbids.
- Once a file's Rust results are always Arrow, audit any remaining `import numpy as np`: a
  NumPy array returned directly by Rust (e.g. a per-user validity mask — that's index
  metadata, and intentionally stays NumPy) needs no `numpy` import to call methods on: `.sum()`,
  `.any()`, `.size`, and iteration all work on an existing NumPy array instance without ever
  writing `import numpy`. Only filtering/masking an actual *value* array now needs
  `pyarrow.compute` (`pc.filter(...)`) instead of `numpy` (`np.asarray(...)[mask]`).

### A gotcha worth remembering when touching existing tests

Because `pyo3_arrow::PyArray` extraction transparently coerces plain NumPy arrays, a test that
asserts `TypeError`/`ValueError` when a Rust binding is called with mismatched NumPy/Arrow
arguments will start failing with "did not raise" once that binding goes Arrow-only — the call
now just works, with the correct (identical) result either way. Convert such a test into an
"accepts mixed backends, produces identical output" test instead of trying to preserve the old
rejection behavior. See `git show 1d444d2` for the exact template already used once.

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
3. Create the Python wrapper in `fastmob/measures/individual.py` (or a new file), following the pattern of `jump_lengths`.
4. Re-export from `fastmob/measures/__init__.py` and `fastmob/__init__.py`.
5. Add a correctness test in `tests/correctness/test_individual.py` and a benchmark in `benchmarks/bench_individual.py`.
6. Follow **Rule 1**: instantiate a module-level `TrajectoryDispatcher`; do not use `_is_polars_backed`.
7. Follow **Rule 2**: do not call `_prepare_trajectory(drop_nulls=True)` on the indexed path; add the explicit `Float64` cast; update the Arrow-indexed Rust binding with `as_nullable_f64_array` + `arrow_valid_rows`.

See the "Backend dispatch and null-handling conventions" section above for templates.

## Trajectory hierarchy and mobility-analysis feature modules

On top of the flat `TrajDataFrame`/`FlowDataFrame` model, `fastmob/core/` also has a trackintel-style typed hierarchy, built by composing already-Rust-backed operations rather than new kernels for most levels:

- `positionfixes_dataframe.py` — `Positionfixes`, a semantic alias for `TrajDataFrame` (the raw-GPS-fix level). `.generate_staypoints()`/`.generate_triplegs()` start the hierarchy.
- `staypoints_dataframe.py` — `Staypoints` (wraps `preprocessing.stay_locations`); `.generate_locations()`, `.create_activity_flag()`.
- `triplegs_dataframe.py` — `Triplegs` (movement segments, derived from `preprocessing.segment(method="stop")` minus the stop-windows matched against `Staypoints`); `.predict_transport_mode()`, `.calculate_modal_split()`, `.generate_trips()`. See this file's module docstring for why a tripleg's `started_at`/`finished_at`/`length_km` must be recovered from the *bracketing* stop segments rather than the moving segment's own rows — `segment(method="stop")` attributes a stop's entry/leaving transition rows to the stop's own segment, not the moving one before/after it.
- `trips_dataframe.py` / `tours_dataframe.py` — `Trips` (consecutive triplegs merged at non-activity staypoints) and `Tours` (consecutive trips returning to the same `Location`). Both build their per-user timeline scan in NumPy after extracting columns via Narwhals (`.to_numpy()`/`nw.from_dict(..., backend=...)` at the boundary), since Narwhals has no "collect to list per group" aggregation and this operates on tables far smaller than raw positionfixes — not a performance-critical path, so **do not** treat this as license to import pandas/polars elsewhere; the Narwhals-only rule above still applies everywhere else.
- `locations_dataframe.py` — `Locations` (DBSCAN-clustered recurring stops via `preprocessing.cluster`); `.identify()` for home/work/other labeling.

Related feature modules, each with its own correctness tests under the matching `tests/correctness/` subtree:

- `fastmob/trajectory/_smooth.py` — `smooth()`, a Kalman constant-velocity filter/RTS smoother (same named-method dispatch shape as `_interpolate.py`, but replaces existing points' positions at the same cardinality instead of inserting new ones).
- `fastmob/trajectory/_shape_cluster.py` — `cluster_trajectory_shapes()`/`cluster_trajectory_shapes_from_segments()`, DBSCAN over a rotation/translation-invariant distance-geometry signature (`fastmob-core/src/trajectory/shape_signature.rs`); a different notion of "clustering" than `preprocessing.cluster`'s stop-point clustering.
- `fastmob/preprocessing/_transport_mode.py` / `_activity.py` — `predict_transport_mode`, `calculate_modal_split`, `create_activity_flag`, `identify_locations`. Plain Narwhals, no new Rust kernel, since they classify/aggregate the small per-tripleg/per-staypoint/per-location tables the hierarchy above already produces.
- `fastmob/models/next_location.py` — `NextLocationPredictor`, an order-k Markov chain (with backoff) fit over a location-id sequence, backed by a stateful PyO3 handle (`fastmob._core.NextLocationModels`, prepare-once/query-many). Deliberately separate from `markov_diary_generator.py`'s `MarkovDiaryGenerator` (a fixed-48-state hour/home-away diary *generator*, not a next-location *predictor*).

None of these features have a wired-up comparison-library `.venv-*`/cached-reference setup yet (trackintel/tracktable/humobi/movingpandas-Kalman comparisons were all deliberately deferred) — correctness tests use hand-built synthetic fixtures with known ground truth plus real-dataset (Brightkite/GeoLife) structural-invariant checks instead.

# Project Structure

- Each functionality should have its own small file, like jump lenghts belong to the jump_lenghts.py file. 
- Each file should have a mirror in the test folder that tests its functionality.
