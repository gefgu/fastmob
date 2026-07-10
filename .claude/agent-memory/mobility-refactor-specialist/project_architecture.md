---
name: fkmob project architecture and conventions
description: Core architectural rules, file layout, Narwhals constraint, column-detection design, and test conventions for fkmob
type: project
---

## Layout
- `src/lib.rs` — Rust kernels compiled to `fkmob/_core.*.so` via maturin
- `fkmob/preprocessing/` — preprocessing functions (filter, compress, stay_locations, cluster); each in its own file mirrored by `tests/correctness/preprocessing/`
  - `filter.py` — `filter_trajectory_batch` Rust kernel via `src/filter_traj.rs` (uses Rayon)
  - `compress.py` — `compress_trajectory_batch` Rust kernel via `src/compress_traj.rs`
  - `stay_locations.py` — `detect_stay_locations_batch` Rust kernel via `src/stay_locations_rs.rs`
  - `cluster.py` — pure Python using scikit-learn DBSCAN with Haversine metric
- `fkmob/measures/` — measures organized into subfolders:
  - `fkmob/measures/_common.py` — shared utilities: candidate lists, `_pick_existing_column`, `_detect_trajectory_columns`, `_prepare_trajectory`
  - `fkmob/measures/spatial/` — `jump_lengths.py`, `radius_of_gyration.py`
  - `fkmob/measures/visits/` — `activity.py`, `intermittance.py` (was `individual.py`), `motifs.py`
  - `fkmob/measures/flows/` — `od.py`
  - `fkmob/measures/fitting/` — `mobility_laws.py`
- Each subfolder has an `__init__.py` that re-exports all public symbols from its files.
- `fkmob/measures/__init__.py` — imports directly from subfolders, re-exports full public API.
- `fkmob/__init__.py` — same public API re-exported at package level.
- `tests/correctness/` — test files mirror the source subfolder structure:
  - `tests/correctness/spatial/`, `visits/`, `flows/`, `fitting/`
  - `tests/correctness/test_common.py` stays at root (no matching subfolder for `_common.py`)

**Why:** CLAUDE.md mandates "each functionality should have its own small file" with a mirror test file. Subfolders group by domain (spatial/visits/flows/fitting) for discoverability.

## Narwhals constraint
- Never import pandas or polars directly in measure code — use `nw.from_native(traj, eager_only=True)`.
- Exception: `activity.py` and `intermittance.py` legitimately convert to pandas for row-iterative loops (noted in comments).
- `od.py` intentionally returns pandas (pivot_table requires it); `import pandas as pd` is deferred inside the function body, not at module level.
- Preserve `backend=df.implementation` when constructing output dicts so result backend matches input.

## Shared pure-Python helpers in `_common.py`
- `_shannon_entropy(counts: list[int]) -> float` — Shannon entropy in bits; shared by `uncorrelated_entropy.py` and `uncorrelated_location_entropy.py`.
- `import math` added to `_common.py`.

## Column auto-detection
All authoritative candidate lists live in `fkmob/measures/_common.py`:
- `DATETIME_CANDIDATES`, `LAT_CANDIDATES`, `LNG_CANDIDATES`, `UID_CANDIDATES` — trajectory measures
- `ACTIVITY_CANDIDATES`, `TIMESTAMP_CANDIDATES`, `DAY_CANDIDATES`, `USER_ID_CANDIDATES`, `LOCATION_CANDIDATES`, `DURATION_CANDIDATES`, `PURPOSE_CANDIDATES` — visit/activity measures
- `ORIGIN_CANDIDATES`, `DEST_CANDIDATES` — OD matrix

`_detect_trajectory_columns(nw_df, ...)` — validate + auto-detect all four trajectory column roles; raises ValueError with actionable message listing candidates.
`_prepare_trajectory(traj, ...)` — full preprocessing pipeline: wrap → row_index → detect → drop_nulls → sort → cast lat/lng Float64. Returns `(df, datetime_col, lat_col, lng_col, uid_col)`.

**Why:** Candidate lists were duplicated (with subtle differences) in jump_lengths.py, radius_of_gyration.py, activity.py, and individual.py.

## Data flow for trajectory measures
1. `_prepare_trajectory` handles all boilerplate (Narwhals wrap, column detect, sort, cast).
2. Extract numpy/list arrays from the clean df.
3. Call Rust kernel.
4. Reassemble result dataframe in original backend.

## Rust/Python boundary conventions
- Rust kernels receive flat Python lists (Vec<String>, Vec<u32>, etc.) + per-user index ranges (Vec<(usize, usize)>) — same pattern as `waiting_times_seconds` and `radius_of_gyration_batch_km`.
- `compute_daily_motifs` signature: `(unique_ids, purposes, start_hours, end_hours, date_ids, durations, user_ranges, user_id_labels) → (Vec<String>, Vec<i32>, Vec<String>)` — parallel flat-column output.
- date_id encoding: days since Unix epoch as Int32. Convert: `.dt.truncate("1d").cast(Int64) // (86400 * 1_000_000)`. Reverse: `date_id * 86400 * 1_000_000 → cast(Datetime("us"))`.
- `canonical_adjacency_form_internal` is kept as a non-PyO3 fn in motifs.rs so it can be called by `compute_daily_motifs` without overhead. `canonical_adjacency_form` is the thin `#[pyfunction]` wrapper around it.

## Test strategy
- Run: `source .venv/bin/activate && pytest tests/correctness/ -m "not skmob" -q`
- Current baseline: 256 passed, 5 pre-existing failures (pyarrow missing — unrelated to our code), 1 skipped, 17 deselected (skmob marker).
- `@pytest.mark.skmob` tests require `skmob` package and Brightkite dataset download.
- `conftest.py` fixtures: `synthetic_tdf` (pandas, 3 users × 5 pts), `synthetic_tdf_polars` (same in Polars), `brightkite_skmob`.
