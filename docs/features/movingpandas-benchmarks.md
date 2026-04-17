# movingpandas Benchmarks
> Add pytest-benchmark comparisons of skmob2 against movingpandas for the two measures that have a meaningful overlap: jump lengths (per-segment distances) and radius of gyration.

## Purpose & Motivation

skmob2 already benchmarks against the original `skmob` library to validate its performance story. movingpandas is a separate, widely-used Python mobility library that also computes per-point distances (`add_distance`) on trajectories. Adding movingpandas as a third competitor in the benchmark suite gives skmob2 a broader performance context — positioning it not just as a faster `skmob` but as a generally faster mobility-analysis toolkit. It also surfaces any regression risk: if skmob2 is ever slower than a pure-Python GeoPandas loop, that is worth knowing.

## Success Criteria

1. `bash tests/run_benchmarks.sh` runs without error whether or not movingpandas is installed; movingpandas benchmarks are auto-skipped when absent (matching the existing skmob skip pattern).
2. `bash tests/setup_env.sh --movingpandas` installs movingpandas and its geospatial dependencies into `.venv`.
3. `pytest tests/benchmarks/ -m movingpandas` runs only movingpandas benchmark tests and exits 0.
4. The benchmark table produced by pytest-benchmark includes movingpandas rows alongside skmob and skmob2 rows so all three are visible in the same comparison.
5. The movingpandas benchmark exercises the same Brightkite slice sizes (1k / 10k / 100k / 1M / 4M) as the skmob benchmark.

## Scope

**In scope:**
- A new `dev-movingpandas` extras group in `pyproject.toml`.
- A `--movingpandas` flag in `tests/setup_env.sh` that installs it.
- A new session-scoped fixture `bench_slice_movingpandas` in `tests/benchmarks/conftest.py` that constructs a `movingpandas.TrajectoryCollection` from the existing `brightkite_raw` pandas DataFrame.
- A movingpandas benchmark case in `tests/benchmarks/bench_jump_lengths.py` (calling `tc.add_distance(units="km", overwrite=True)` to measure per-segment distances — the closest functional analog to `jump_lengths`).
- A movingpandas benchmark case in `tests/benchmarks/bench_radius_of_gyration.py` (no native equivalent; the benchmark must implement radius-of-gyration on top of movingpandas' geometry layer — see Implementation Strategy).
- A `movingpandas` pytest marker registered in `tests/benchmarks/conftest.py` (mirroring the `skmob` marker in `tests/correctness/conftest.py:137`).

**Out of scope:**
- Correctness cross-validation between skmob2 and movingpandas (the two libraries compute slightly different things; see Trade-Offs).
- Benchmarking movingpandas measures that have no skmob2 analog (stop detection, trajectory clipping, etc.).
- Changes to the Rust extension or to any measure implementation.
- Polars fixtures for movingpandas (movingpandas is GeoPandas-based; Polars input is meaningless here).
- CI integration (adding movingpandas to `.github/workflows/CI.yml`) — that is a separate decision.

## Constraints

1. movingpandas requires GeoPandas, Shapely ≥ 2.0, and pyproj — a heavier dependency stack than skmob. They are installable via `pip` on Linux (the package ships a `py3-none-any.whl`) but installation of GeoPandas may pull in GDAL/GEOS binaries. The extras group must be kept opt-in.
2. The Brightkite dataset column names (`user`, `check-in_time`, `latitude`, `longitude`) must be mapped to the movingpandas constructor API, which expects a geometry column (Shapely Points) or explicit `x`/`y` kwargs plus a datetime index or `t=` kwarg, plus a trajectory-ID column.
3. movingpandas does **not** implement radius of gyration natively. Any RoG benchmark for movingpandas must be hand-rolled on top of its geometry layer, which adds authorship complexity and makes the comparison less apples-to-apples.
4. The `bench_slice_movingpandas` fixture will be substantially slower to construct than `bench_slice_skmob` because building a `TrajectoryCollection` requires creating Shapely Point geometries and a GeoDataFrame — this setup cost must be isolated into the fixture (session-scoped) and not included in the timed benchmark body.

## Architecture & Integration Points

The feature touches four files and adds one new optional-deps group:

- `pyproject.toml:23-33` — `[project.optional-dependencies]` block where the new `dev-movingpandas` group is added alongside the existing `dev-skmob` group.
- `tests/setup_env.sh:16-49` — the `--skmob` flag pattern is replicated for `--movingpandas`.
- `tests/benchmarks/conftest.py:1-88` — all benchmark fixtures live here. The new `brightkite_tc` (TrajectoryCollection) and `bench_slice_movingpandas` fixtures follow the `brightkite_tdf` / `bench_slice_skmob` pattern exactly.
- `tests/benchmarks/bench_jump_lengths.py:1-37` — one new test function `test_jump_lengths_movingpandas` is added following the `test_jump_lengths_skmob` pattern.
- `tests/benchmarks/bench_radius_of_gyration.py:1-36` — one new test function `test_radius_of_gyration_movingpandas` is added.

The benchmark runner (`tests/run_benchmarks.sh:29`) calls `pytest tests/benchmarks/ -v "$@"` with no filtering, so new benchmark tests are automatically picked up without changes to the shell script.

Marker registration: the existing `skmob` marker is registered in `tests/correctness/conftest.py:136-140`. The new `movingpandas` marker must be registered in `tests/benchmarks/conftest.py` because that is the conftest that applies to benchmark tests (the correctness conftest does not apply to `tests/benchmarks/`).

## Similar Patterns & Reuse

Every pattern below is reused verbatim — no new abstractions are needed.

- **`bench_slice_skmob` fixture**: `tests/benchmarks/conftest.py:68-78 — bench_slice_skmob(brightkite_tdf, dataset_size)`
  Constructs a `skmob.TrajDataFrame` slice at the parametrized size. The movingpandas fixture follows the same shape: take `brightkite_raw`, build a `TrajectoryCollection` at full size once (session scope), then slice it per test.

- **`pytest.importorskip` pattern**: `tests/benchmarks/bench_jump_lengths.py:8-11 — pytest.importorskip("skmob.measures.individual", reason="...")`
  The movingpandas benchmark uses `pytest.importorskip("movingpandas", reason="Install movingpandas to run this benchmark")` at the top of each test function. The fixture itself also calls `importorskip` so both the fixture and the test guard independently.

- **`dataset_size` parametrize fixture**: `tests/benchmarks/conftest.py:57-59 — @pytest.fixture(params=[1_000, 10_000, 100_000, 1_000_000, 4_000_000])`
  The movingpandas slice fixture accepts `dataset_size` as a parameter just like `bench_slice_pandas` and `bench_slice_skmob` do.

- **`dev-skmob` extras group**: `pyproject.toml:33-35 — dev-skmob = ["scikit-mobility"]`
  The new group mirrors this: `dev-movingpandas = ["movingpandas", "geopandas"]`.

- **`--skmob` flag in setup_env.sh**: `tests/setup_env.sh:18-19 — INSTALL_SKMOB=false / [[ "$arg" == "--skmob" ]]`
  The `--movingpandas` flag uses the same pattern: a boolean variable, a loop over `$@`, and a conditional `uv pip install` call with a fallback warning.

## Implementation Strategy

### Step 1 — Add the `dev-movingpandas` extras group in `pyproject.toml`

**Before**: `pyproject.toml:33` ends with `dev-skmob = ["scikit-mobility"]`.

**After**: A new stanza `dev-movingpandas = ["movingpandas", "geopandas"]` is appended in the same `[project.optional-dependencies]` block. `geopandas` is listed explicitly because it is movingpandas' primary hard dependency and its installation on Linux sometimes requires a binary wheel that pip must resolve cleanly.

### Step 2 — Add `--movingpandas` flag to `tests/setup_env.sh`

**Before**: `tests/setup_env.sh:18-49` handles only `--skmob`.

**After**: A parallel `INSTALL_MOVINGPANDAS=false` variable, a loop branch for `--movingpandas`, and a conditional `uv pip install -e ".[dev-movingpandas]"` block with the same fallback warning structure as the `--skmob` block.

### Step 3 — Add fixtures in `tests/benchmarks/conftest.py`

**Before**: The conftest ends at line 88 with `bench_slice_polars`.

**After**: Two new session-scoped fixtures are appended.

`brightkite_tc(brightkite_raw)` — skips if movingpandas is absent; constructs the full `TrajectoryCollection` once per session. The construction sequence is:

```
1. mpd = pytest.importorskip("movingpandas")
2. gpd = pytest.importorskip("geopandas")
3. from shapely.geometry import Point
4. gdf = gpd.GeoDataFrame(
       brightkite_raw,
       geometry=gpd.points_from_xy(brightkite_raw["longitude"], brightkite_raw["latitude"]),
       crs="EPSG:4326",
   )
5. gdf["check-in_time"] = pd.to_datetime(gdf["check-in_time"])
6. gdf = gdf.set_index("check-in_time")
7. return mpd.TrajectoryCollection(gdf, traj_id_col="user")
```

`bench_slice_movingpandas(brightkite_tc, dataset_size)` — slices the trajectory collection to the first `dataset_size` rows by filtering on `tc.to_point_gdf()` then re-constructing a smaller `TrajectoryCollection`. This matches the `.head(dataset_size)` slicing used for pandas and skmob fixtures.

### Step 4 — Add movingpandas case to `tests/benchmarks/bench_jump_lengths.py`

**Before**: `bench_jump_lengths.py:37` ends with `test_jump_lengths_skmob2_polars`.

**After**: A new function:

```python
def test_jump_lengths_movingpandas(benchmark, bench_slice_movingpandas):
    """Benchmark movingpandas.TrajectoryCollection.add_distance (km) — analog of jump_lengths."""
    mpd = pytest.importorskip(
        "movingpandas",
        reason="Install movingpandas to run this benchmark",
    )
    benchmark(
        bench_slice_movingpandas.add_distance,
        overwrite=True,
        units="km",
    )
```

`TrajectoryCollection.add_distance(overwrite=True, units="km")` iterates over each `Trajectory` in the collection and calls `Trajectory.add_distance`, which computes pairwise Haversine distances between consecutive points. This is the closest functional equivalent to `jump_lengths`: it produces a distance column on the trajectory's GeoDataFrame containing the distance from each point to the previous point, in km. The semantic difference (movingpandas mutates the object in-place; skmob2 returns a new DataFrame) is acceptable for a benchmark because we are measuring compute time, not API style.

### Step 5 — Add movingpandas case to `tests/benchmarks/bench_radius_of_gyration.py`

**Before**: `bench_radius_of_gyration.py:36` ends with `test_radius_of_gyration_skmob2_polars`.

**After**: A new function that wraps a pure-Python RoG computation over a `TrajectoryCollection`. Since movingpandas has no native RoG, the benchmark calls a helper that, for each trajectory, extracts `(lat, lng)` from the GeoDataFrame and computes RoG in plain NumPy. This makes the comparison intentionally unfavorable to movingpandas — the point is to show how much skmob2's Rust kernel gains. The helper must be defined in the benchmark file (not in `skmob2/`) to avoid polluting the library with benchmark-only code.

```python
def _rog_for_tc(tc):
    """Compute RoG for all trajectories in a TrajectoryCollection using NumPy."""
    import numpy as np
    results = []
    for traj in tc.trajectories:
        gdf = traj.df
        lats = gdf.geometry.y.to_numpy()
        lngs = gdf.geometry.x.to_numpy()
        # RMS Haversine from centroid
        ...
    return results
```

The test:

```python
def test_radius_of_gyration_movingpandas(benchmark, bench_slice_movingpandas):
    pytest.importorskip("movingpandas", reason="...")
    benchmark(_rog_for_tc, bench_slice_movingpandas)
```

## Trade-Offs

| Gained | Sacrificed / risked |
|---|---|
| Three-way benchmark table (skmob2 vs skmob vs movingpandas) gives skmob2 a richer performance story | movingpandas brings a heavy transitive dependency chain (GeoPandas, Shapely, pyproj, GEOS binaries) that increases setup friction |
| Demonstrates skmob2 performance beyond the original skmob comparison | The `add_distance` / `jump_lengths` comparison is semantically close but not identical (movingpandas mutates; skmob2 returns; movingpandas also supports arbitrary CRS reprojection, skmob2 assumes WGS84) |
| Benchmark infrastructure for a new competitor is reusable for future movingpandas measures | movingpandas has no native RoG, so the RoG "benchmark" hand-rolls NumPy logic, making the comparison less rigorous than the jump-lengths comparison |
| The `bench_slice_movingpandas` fixture construction (GeoDataFrame + TrajectoryCollection) is expensive at 4M rows — isolating it as session-scoped prevents benchmark noise, but cold-start time is still visible to anyone running `setup_env.sh` | |

## Rejected Approaches

**Approach: Add movingpandas to the top-level `dev` extras group**
Why rejected: `dev` currently installs with a single `uv pip install -e ".[dev]"` in `tests/setup_env.sh:37`. Folding in movingpandas would force every developer to install GeoPandas and its binary dependencies just to run the correctness tests, breaking the lightweight default setup. The `dev-skmob` precedent (`pyproject.toml:33-35`) establishes that heavy optional comparators go in their own extras group.

**Approach: Use conda/mamba to install movingpandas instead of pip**
Why rejected: The entire project toolchain uses `uv` for Python environment management (`tests/setup_env.sh:24 — uv venv .venv`). Introducing conda would require a second environment manager, complicating both local dev and any future CI. movingpandas ships a `py3-none-any.whl` on PyPI (version 0.22.4 as of July 2025) and `geopandas` ships binary wheels for Linux on PyPI, so pip installation is viable on Linux without conda.

**Approach: Put the `movingpandas` marker in `tests/correctness/conftest.py`**
Why rejected: `tests/correctness/conftest.py` is scoped to the correctness suite. Benchmark tests live under `tests/benchmarks/` and use `tests/benchmarks/conftest.py`. Placing the marker in the wrong conftest would work by pytest inheritance but would be misleading and fragile if the test directories are ever decoupled.

**Approach: Implement a movingpandas-native radius-of-gyration using trajectory geometry operations**
Why rejected: movingpandas exposes per-point geometry via `Trajectory.df.geometry` but has no aggregate statistical method for RoG. Any "movingpandas-native" implementation would still be the same NumPy loop — just accessed through the movingpandas GeoDataFrame layer. The added complexity of navigating movingpandas' geometry API in the benchmark function is not worth the marginal authenticity gain.

**Approach: Benchmark only jump lengths, skip RoG**
Why rejected: Both existing benchmark files (`bench_jump_lengths.py`, `bench_radius_of_gyration.py`) follow the convention of having one benchmark file per measure. Skipping movingpandas for RoG would leave the benchmark table asymmetric and give the misleading impression that movingpandas has no comparable RoG capability (it does, via manual NumPy). The explicit "hand-rolled" benchmark is more honest than silence.

**Approach: Slice `TrajectoryCollection` by re-filtering its `to_point_gdf()` and reconstructing**
Kept as the plan, but flagged: at large sizes (1M, 4M rows), rebuilding a `TrajectoryCollection` from a filtered GeoDataFrame will be slow inside the fixture. An alternative is to simply use the full `brightkite_tc` for all size tiers and vary the number of trajectories rather than the row count. This would make the movingpandas size parametrization incomparable to the pandas/skmob size parametrization. The current plan preserves comparability at the cost of fixture construction time — which is acceptable since fixtures are session-scoped.

## Assumptions & Open Questions

1. **Assumed**: `geopandas.points_from_xy(longitude, latitude)` is sufficient to build a valid WGS84 GeoDataFrame from the Brightkite columns without additional CRS conversion — no reprojection needed.
2. **Assumed**: `movingpandas.TrajectoryCollection(gdf, traj_id_col="user")` accepts `"user"` as the trajectory ID column and groups correctly. The constructor documentation states it takes the column name; the Brightkite column is named `"user"` (`tests/benchmarks/conftest.py:36`).
3. **Open question**: Does movingpandas' `add_distance(units="km")` use the same Haversine formula as skmob2's Rust kernel, or does it use Vincenty or a projected approximation? The documentation says "meters per second for geographic CRS (EPSG:4326)"; the `units="km"` parameter converts after the fact. If the underlying formula differs, the per-segment distances will be numerically different from skmob2's output — this is fine for a benchmark (we are measuring speed, not correctness) but should be noted in a comment in the benchmark file.
4. **Open question**: At 4M rows, constructing the `TrajectoryCollection` inside a session-scoped fixture may take 10–60 seconds. This is acceptable for a benchmark run but should be documented in the fixture docstring so future contributors understand why session scope is mandatory.
5. **Assumed**: `pip install movingpandas geopandas` succeeds on the target Linux environment without conda. This should be validated the first time `bash tests/setup_env.sh --movingpandas` is run.
6. **Open question**: Should the `bench_slice_movingpandas` fixture construct a new `TrajectoryCollection` per size tier (slow, but exact row count) or iterate on the existing full collection and stop early (fast, but row count is approximate)? The current plan uses reconstruction; this decision can be revisited if fixture setup proves prohibitively slow at 1M+ rows.

## Code That Could Be Refactored *(informational)*

- `tests/benchmarks/conftest.py:62-88` — `bench_slice_pandas`, `bench_slice_skmob`, and `bench_slice_polars` share the same `brightkite_raw.head(dataset_size).copy()` slicing logic wrapped in different constructors. If a fourth or fifth competitor is added, this pattern becomes repetitive. A helper function `_head_slice(raw, n)` that returns a pandas slice could be extracted. Not a blocker; this is purely cosmetic.

- `tests/setup_env.sh:18-49` — The `--skmob` install block will be duplicated by the new `--movingpandas` block. If three or more optional comparators are eventually supported, the shell script would benefit from a loop over optional extras. Not a blocker at two comparators.

## Proposed Next Steps

1. Run `uv pip install movingpandas geopandas` in the `.venv` to verify pip-installability on the current Linux host before writing any code.
2. Add `dev-movingpandas = ["movingpandas", "geopandas"]` to `pyproject.toml:33-35` (after the `dev-skmob` group).
3. Add the `--movingpandas` flag block to `tests/setup_env.sh`, mirroring the `--skmob` block at lines 38-45.
4. Add the `brightkite_tc` and `bench_slice_movingpandas` session-scoped fixtures to `tests/benchmarks/conftest.py`, after the existing `bench_slice_polars` fixture (line 88). Register the `movingpandas` pytest marker via `pytest_configure` in the same file.
5. Add `test_jump_lengths_movingpandas` to `tests/benchmarks/bench_jump_lengths.py`.
6. Add `test_radius_of_gyration_movingpandas` (with the inline `_rog_for_tc` helper) to `tests/benchmarks/bench_radius_of_gyration.py`.
7. Run `bash tests/run_benchmarks.sh -k 1k` to verify all three competitors run without error on the smallest dataset slice before full-scale runs.
