# fastmob-rs benchmarks

Dataset: the cached Brightkite check-in file (`tests/shared/data/loc-brightkite_totalCheckins.txt.gz`), first N usable rows.
Machine: local dev box, `--release`.

Reproduce:

```bash
cargo bench -p fastmob-rs                      # Rust vs. vendored citybehavex baseline
.venv/bin/python scripts/bench_fastmob_rs.py   # Rust vs. fastmob Python path (+ parity check)
```

## Phase 1 — jump lengths + radius of gyration

Both gates the migration requires are met at every size.

### vs. citybehavex (criterion, typed `datetime` column)

End-to-end `jumps_rog`: clean, arrange, then compute jump lengths **and** radius of gyration.

| rows | fastmob-rs | citybehavex | speedup |
|---:|---:|---:|---:|
| 100,000 | 3.00 ms | 5.19 ms | 1.73x |
| 1,000,000 | 24.3 ms | 38.1 ms | 1.57x |
| 4,000,000 | 127.0 ms | 206.3 ms | 1.62x |

### vs. the fastmob Python path (`scripts/bench_fastmob_rs.py`)

| rows | fastmob-rs | Python | speedup | values |
|---:|---:|---:|---:|---|
| 100,000 | 6.7 ms | 28.2 ms | 4.22x | exact match |
| 1,000,000 | 27.8 ms | 56.4 ms | 2.03x | exact match |
| 4,000,000 | 132.9 ms | 221.5 ms | 1.67x | exact match |

"Exact match" is bit-for-bit equality on all 1,705,587 jump lengths and 25,378 radii at 4M rows, not agreement within a tolerance.

### Where the time goes, and what reuse buys

| stage (4M rows) | time |
|---|---:|
| `prepare` (clean + arrange) | 72.3 ms |
| `jump_lengths_flat` over an existing `PreparedTrajectory` | 6.6 ms |

This is the point of the `PreparedTrajectory` split. The first measure over a frame costs ~127 ms; every additional measure costs ~7 ms instead of a fresh ~206 ms, because it reuses the arrangement. citybehavex's `jumps_rog_for_filters` re-cleans and re-sorts once per filter, so the saving there scales with the filter count.

## Two findings worth keeping

**Bit-parity depends on a cargo feature, not just the formula.** `fastmob-core`'s Haversine has a `numkong`-accelerated path and a pure-Rust fallback that disagree in the last bits. Building `fastmob-rs` against `fastmob-core` with `default-features = false` produced jump lengths that differed from Python by up to 9.1e-12 on 56,293 of 57,551 values at 100k rows. That is small in absolute terms but not harmless: citybehavex has a documented case where last-bit Haversine differences reclassified 503 near-zero jumps and visibly shifted a jump-length ECDF. `fastmob-rs` therefore mirrors `fastmob-py`'s feature wiring (`default = ["simd"]`) so both resolve to the same kernel. Radius of gyration was unaffected — it never took the accelerated path — which is why a parity check on a single measure would have missed this.

**String datetime columns are parse-bound, not sort-bound.** With a raw string `datetime` column, 4M rows take 828 ms in fastmob-rs and 843 ms in the baseline — a ~2% difference, because Polars' datetime string parsing (~700 ms) dominates both sides equally. The 1.6x figures above use a typed `datetime` column, which is what citybehavex actually has: its frames are read from parquet, where the column arrives already typed. Callers holding string timestamps should parse once and reuse the frame rather than expect this crate to make parsing faster.
