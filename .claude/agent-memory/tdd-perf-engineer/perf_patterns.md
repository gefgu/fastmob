---
name: skmob2 performance lessons from radius_of_gyration implementation
description: Bottleneck analysis and optimization patterns discovered while implementing radius_of_gyration
type: project
---

## Key finding: batch Rust calls beat per-user Python loops

**Why:** With 162 users and 100k rows, calling the Rust kernel once per user (162 crossings) is slower than computing group boundaries in Python and making a single batch Rust call. The per-user loop with Narwhals group_by also incurs Python-side grouping overhead.

**How to apply:** For any measure that computes one value per user (radius_of_gyration, max_distance, etc.):
1. Sort df by [uid, datetime] so same-uid rows are contiguous
2. Extract full lat/lng arrays once with `.to_list()`
3. Linear scan over uid column to build `(start, end)` ranges
4. Single batch Rust call with `ranges: Vec<(usize, usize)>`
5. Use `rayon::par_iter()` inside Rust to parallelize over users

**Measured speedup (100k rows, Brightkite dataset):**
- Per-user loop: pandas 185ms, polars 96ms vs skmob 308ms
- Batch call: pandas 123ms (1.5x improvement), polars 36ms (2.6x improvement)
- Final vs skmob: pandas 2.5x faster, polars 8.4x faster

## Python overhead dominates at small scale (1k rows)
- At 1k rows, skmob2 pandas is roughly equal to skmob (both ~3.8ms) — Narwhals/pandas overhead dominates
- Polars backend is always faster (3-8x) due to less overhead in the narwhals/polars path
- The Narwhals `sort()` on pandas is expensive (categorical factorization) — accounts for ~36% of 100k runtime

## geo crate vs skmob Haversine: systematic difference
- The `geo` crate and skmob's `getDistanceByHaversine` both implement spherical Haversine
- Due to floating-point evaluation order differences, results can differ by up to 0.02 km for long trajectories
- Do NOT use `atol=1e-3` for skmob comparison tests — use `atol=0.02`
- For unit tests of RoG values, compute expected values by calling the Rust kernel directly, not reimplementing in Python
