---
name: skmob2 project architecture and conventions
description: Core architectural rules, file layout, Narwhals constraint, column-detection design, and test conventions for skmob2
type: project
---

## Layout
- `src/lib.rs` — Rust kernels compiled to `skmob2/_core.*.so` via maturin
- `skmob2/measures/` — one file per measure (jump_lengths.py, radius_of_gyration.py, etc.)
- `skmob2/measures/_common.py` — shared utilities: candidate lists, `_pick_existing_column`, `_detect_trajectory_columns`, `_prepare_trajectory`
- `tests/correctness/` — one test file per source file (test_jump_lengths.py, test_common.py, etc.)
- `tests/benchmarks/` — pytest-benchmark tests comparing skmob vs skmob2

**Why:** CLAUDE.md mandates "each functionality should have its own small file" with a mirror test file.

**How to apply:** When adding a new measure, create a new file (not adding to individual.py). Add a mirror test file immediately.

## Narwhals constraint
- Never import pandas or polars directly in measure code — use `nw.from_native(traj, eager_only=True)`.
- Exception: `activity.py` and `individual.py` legitimately convert to pandas for row-iterative loops (noted in comments).
- `od.py` intentionally returns pandas (pivot_table requires it); `import pandas as pd` is deferred inside the function body, not at module level.
- Preserve `backend=df.implementation` when constructing output dicts so result backend matches input.

## Column auto-detection
All authoritative candidate lists live in `skmob2/measures/_common.py`:
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

## Test strategy
- Run: `source .venv/bin/activate && pytest tests/correctness/ -m "not skmob" -q`
- 58 tests (as of 2026-04-17) — all must pass before any commit.
- `@pytest.mark.skmob` tests require `skmob` package and Brightkite dataset download.
- `conftest.py` fixtures: `synthetic_tdf` (pandas, 3 users × 5 pts), `synthetic_tdf_polars` (same in Polars), `brightkite_skmob`.
