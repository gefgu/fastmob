---
name: skmob2 codebase architecture and patterns
description: Key patterns for Rust kernels, Python wrappers, Narwhals usage, column detection, and test structure in skmob2
type: project
---

## Rust kernels (src/lib.rs)
- Single-user kernel: accepts `Vec<(f64, f64)>` (lat/lng pairs), returns `f64` or `Vec<f64>`
- Batch kernel: accepts flat `Vec<f64>` lat + `Vec<f64>` lng arrays + `Vec<(usize, usize)>` ranges, returns `Vec<f64>` — one result per range
- Use `geo::Haversine.distance(p1, p2) / 1000.0` for km distances. Point::new(lon, lat) — NOTE: geo uses (lon, lat) order
- Use `rayon::prelude::*` + `.par_iter()` for parallel iteration over large slices
- Expose via `#[pyfunction]` + `m.add_function(wrap_pyfunction!(..., m)?)` in `_core` pymodule
- `cargo check` mindset: no warnings allowed

## Python wrappers (skmob2/measures/individual.py)
- Column detection via `_pick_existing_column(nw_df.columns, priority_list)`
- Priority lists: datetime→["datetime","timestamp","time","check-in_time"], lat→["latitude","lat"], lng→["longitude","lon","lng"], uid→["user_id","uid","user"]
- Sort df by `[uid_col, datetime_col, _ROW_ORDER_COL]` (with _ROW_ORDER_COL as tiebreaker)
- For batch operations: extract full lat/lng arrays once, compute group boundaries by linear scan on already-sorted uid column, call batch Rust kernel once
- `nw.from_dict({...}, backend=df.implementation).to_native()` preserves input backend
- Never import pandas/polars directly

## Test patterns (tests/correctness/test_individual.py)
- `pytest.importorskip("skmob2._core")` guard at top of each test
- Fixtures: `synthetic_tdf` (pandas), `synthetic_tdf_polars` — 3 users, 5 pts each in conftest.py
- `@pytest.mark.skmob` for tests that require skmob installed
- Expected values: compute by calling the Rust kernel directly (not reimplementing Haversine in Python) to avoid floating-point formula mismatch (~1e-5 km)
- Haversine tolerance between geo crate and skmob's Python: up to 0.02 km for long trajectories

## Build/test workflow
- `maturin develop` after any Rust change (rebuilds .so in place)
- `bash tests/run_correctness.sh` — correctness only
- `bash tests/run_correctness.sh -m skmob` — includes skmob comparison
- `bash tests/run_benchmarks.sh` — full benchmarks
- `.venv` at repo root; activate with `source .venv/bin/activate`
