---
name: fkmob codebase architecture and patterns
description: Key patterns for Rust kernels, Python wrappers, Narwhals usage, column detection, and test structure in fkmob
type: project
---

## Rust kernels (src/lib.rs)
- Single-user kernel: accepts `Vec<(f64, f64)>` (lat/lng pairs), returns `f64` or `Vec<f64>`
- Batch kernel: accepts flat `Vec<f64>` lat + `Vec<f64>` lng arrays + `Vec<(usize, usize)>` ranges, returns `Vec<f64>` — one result per range
- Use `geo::Haversine.distance(p1, p2) / 1000.0` for km distances. Point::new(lon, lat) — NOTE: geo uses (lon, lat) order
- Use `rayon::prelude::*` + `.par_iter()` for parallel iteration over large slices
- Expose via `#[pyfunction]` + `m.add_function(wrap_pyfunction!(..., m)?)` in `_core` pymodule
- `cargo check` mindset: no warnings allowed

## Python wrappers (fkmob/measures/individual.py)
- Column detection via `_pick_existing_column(nw_df.columns, priority_list)`
- Priority lists: datetime→["datetime","timestamp","time","check-in_time"], lat→["latitude","lat"], lng→["longitude","lon","lng"], uid→["user_id","uid","user"]
- Sort df by `[uid_col, datetime_col, _ROW_ORDER_COL]` (with _ROW_ORDER_COL as tiebreaker)
- For batch operations: extract full lat/lng arrays once, compute group boundaries by linear scan on already-sorted uid column, call batch Rust kernel once
- `nw.from_dict({...}, backend=df.implementation).to_native()` preserves input backend
- Never import pandas/polars directly

## Test patterns (tests/correctness/test_individual.py)
- `pytest.importorskip("fkmob._core")` guard at top of each test
- Fixtures: `synthetic_tdf` (pandas), `synthetic_tdf_polars` — 3 users, 5 pts each in conftest.py
- `@pytest.mark.skmob` for tests that require skmob installed
- Expected values: compute by calling the Rust kernel directly (not reimplementing Haversine in Python) to avoid floating-point formula mismatch (~1e-5 km)
- Haversine tolerance between geo crate and skmob's Python: up to 0.02 km for long trajectories

## Non-Rust measure pattern (pure Python measures)
- Pure Python measures live in `fkmob/measures/<name>.py` — each feature gets its own file
- Use `nw.from_native(visits, eager_only=True)` then convert to pandas for row-iterative loops
- `_pick_existing_column` for all auto-detection; keep all candidate lists local in the module
- For OD/pivot output: use narwhals for filtering/groupby, then convert to pandas for `pivot_table`
- For transition matrices / inner loops: convert to pandas early, iterate with `iterrows()` or `groupby()`
- Candidate lists in TDD migration: user_id→["user_id","uid","agent_id","user","ID"], location→["location_id","area","venueId","location"], duration→["duration_steps","duration_minutes","duration"], purpose→["purpose","activity","location_type"]
- od.py: `od_matrix` returns plain `pd.DataFrame` (not Narwhals-wrapped); `od_metrics_per_area` takes the wide pivot as input
- mobility_laws.py: guard `_scipy_curve_fit = None` when scipy absent; raise `ImportError` with install hint
- activity.py: `day_filter` requires `day_col` resolvable — raise `ValueError` with "day_col" in message
- individual.py: `intermittance_and_degree_of_return` — each user group runs `_compute_single_idr`; `degree_of_return = pi/2` when `mean_exploration == 0`

## Build/test workflow
- `maturin develop` after any Rust change (rebuilds .so in place)
- `bash tests/run_correctness.sh` — correctness only
- `bash tests/run_correctness.sh -m skmob` — includes skmob comparison
- `bash tests/run_benchmarks.sh` — full benchmarks
- `.venv` at repo root; activate with `source .venv/bin/activate`
