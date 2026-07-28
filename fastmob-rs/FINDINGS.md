# fastmob-rs jump_lengths findings

## Benchmark setup

- Dataset: first 4,000,000 Brightkite rows from `loc-brightkite_totalCheckins.txt.gz`, materialized as `/tmp/brightkite_4m.tsv`.
- Comparison target: current citybehavex jump-length implementation shape, copied into a temporary benchmark harness to avoid unrelated citybehavex build failures.
- Benchmark style: single release run with simple section `println!` timing.
- Output count matched in both paths: `1,705,638` positive jump lengths.

## Final one-shot timings

| section | citybehavex current | fastmob-rs |
|---|---:|---:|
| clean/select/sort orchestration | 74.103 ms | 54.789 ms |
| prepare/extract coordinates | 20.135 ms | 0.004 ms |
| group boundaries | 1.280 ms | 1.388 ms |
| core compute | 3.931 ms | 4.157 ms |
| output/filter/handoff | 6.105 ms | 18.488 ms |
| total internal call | 114.273 ms | 64.078 ms |
| total including output Vec extraction | 114.273 ms | 80.641 ms |

`fastmob-rs` is about `1.78x` faster for the internal call and about `1.42x` faster when the benchmark also extracts the returned flat `Series` into a `Vec<f64>`.

## What changed

- `fastmob-rs` now accepts a Rust Polars `DataFrame` and exposes a public `jump_lengths` API.
- The Python `jump_lengths` path was left unchanged; this crate is Rust-only orchestration.
- The Polars path cleans/selects required columns, checks whether the cleaned frame is grouped and monotonic, and only sorts when that check fails.
- Coordinate preparation now borrows contiguous Polars `Float64` buffers using `Cow<'_, [f64]>` instead of copying when possible.
- Timestamp materialization was removed from preparation; timestamps are needed for ordering, but the distance kernel does not consume them.
- UID boundary detection avoids the previous per-row `AnyValue`/`String` allocation path and scans typed buffers or borrowed string values.

## Main finding

The biggest remaining bottleneck is still sorting. The no-sort fast path did not fire on the 4M Brightkite prefix because the data is not actually monotonic within every user group after cleaning. A raw scan found a timestamp direction change for user `637` around line `297458`.

That means the largest theoretical win, deleting the global sort, is not available for this exact input without changing ordering semantics.

## Current bottlenecks

- Polars orchestration plus sort is still the dominant section in `fastmob-rs`: `54.789 ms` out of the `64.078 ms` internal call.
- The actual distance computation is small: `4.157 ms`.
- Preparing coordinate buffers is effectively gone after borrowing contiguous buffers: `0.004 ms`.
- The caller-side flat output extraction is still expensive in the benchmark: `16.556 ms`. Rust callers should prefer consuming the returned `Series`/Arrow buffer directly instead of converting to a `Vec`.

## Practical conclusion

For citybehavex, centralizing the Rust Polars orchestration in `fastmob-rs` is useful and already removes duplicated extraction/boundary work. The big architectural win still depends on callers being able to provide data that is already grouped and monotonic, or on adding a smarter arrange step than a global Polars sort.

The next high-leverage step is not optimizing the haversine kernel. It is replacing the fallback global sort with a trajectory-aware arrange path: encode or partition by user, then sort timestamps within each group.
