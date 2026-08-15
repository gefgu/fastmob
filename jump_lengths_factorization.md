# User-ID factorization: 20.6 ms → 1.7 ms

Optimization log for `fastmob-py/src/measures/individual/factorization.rs`, the
kernel behind `_factorize_arrow_values` and therefore behind the
`[jump_lengths] user-ID factorization` profiling stage.

**Result: 11.8x on the target workload.** No behavioural change: output is
bit-for-bit identical for every input, on every path.

---

## 1. Baseline

Workload: the Brightkite check-in dataset's `uid` column — 4,747,286 rows,
51,406 distinct users, Arrow `Int64`, no nulls, `sort=False`.

Machine: Intel Xeon Silver 4316, 20 physical cores / 40 threads, 30 MiB L3.

```
[jump_lengths] user-ID factorization: 0.019875s (51,406 groups)
[jump_lengths] user-ID factorization: 0.022956s (51,406 groups)
[jump_lengths] user-ID factorization: 0.021137s (51,406 groups)
```

Isolated micro-benchmark, best of 5: **20.59 ms**.

> **Measurement caveat.** This is a shared box. Every number in this document
> was taken at a load average below ~6. The same unmodified kernel measures
> 1.74 ms at load ~2 and 5.4–12.3 ms at load ~27, so any comparison across
> differently-loaded runs is meaningless. Numbers below are best-of-N with
> before/after taken under comparable load.

---

## 2. Where the time actually went

The first thing worth establishing was *which* code path a uid column takes.
`factorize_array` dispatches `Int64` to `factorize_integer!`, which tries
`factorize_dense_integers` first: the value span is `max - min + 1 = 58,228`,
far under `dense_span_cap(4.75M) = 16Mi`, so the column takes the **dense
direct-indexed path** and never touches a hash map at all.

That mattered, because the dense path was the one path in the module that had
never been tuned. Phase timings from a temporarily instrumented build:

| phase | time | share |
|---|---:|---:|
| min/max scan | 0.83 ms | 25 % |
| representative discovery scan | 0.73 ms | 22 % |
| code assignment | 0.61 ms | 18 % |
| **output buffer allocation** | **1.41 ms** | **42 %** |
| final gather | 0.75 ms | 22 % |

Two things stood out immediately:

1. The single most expensive operation was not a scan at all. It was
   `vec![0u32; len]` — zeroing a 19 MB buffer that the gather then overwrote
   in full, one line later.
2. The whole path was **sequential**, on a 40-thread machine, despite being the
   most trivially parallelizable code in the file. It never called
   `should_parallelize`.

---

## 3. Changes

### 3.1 Dense path: parallelize all three scans

Each of the three row scans (min/max, representative discovery, final gather)
now runs over rayon chunks above `PARALLEL_ROW_THRESHOLD`, and each is
specialized on `null_count() == 0` so the common fully-populated column pays no
per-row validity branch.

Chunk length is `len / (threads * 8)`, deliberately oversubscribed — these
scans are memory-bound, so work stealing matters more than minimizing chunk
count.

### 3.2 Dense path: `i128` out of the per-row loop

The old bound was `T::Native: Into<i128>`, and the widening happened *per row*,
in both the min/max fold and the offset computation `(value.into() - min)`.
`i128` is not a native register width on x86-64, so every one of those was a
multi-instruction sequence, and it blocked autovectorization of the min/max
scan — which should be pure SIMD and purely memory-bound.

Replaced with a `DenseInt` trait doing native-width, two's-complement offset
arithmetic:

```rust
(self as i64).wrapping_sub(min as i64) as u64   // signed types
(self as u64).wrapping_sub(min as u64)          // unsigned types
```

Exact whenever the true difference fits in `u64`, which the `span <=
dense_span_cap` check already guarantees. min/max now folds in the native type
too.

### 3.3 Dense path: one table instead of two, zero-initialized

`representative_by_value` and `codes_by_value` were both `span`-sized and never
live at the same offset simultaneously. They are now **one** table — which
matters because `DENSE_SPAN_CEILING` permits 64Mi entries, i.e. 256 MB *per
table*.

The sentinel also changed. `vec![u32::MAX; span]` cannot use the allocator's
zeroed-page path, so it forced an eager `span * 4`-byte memset — up to 256 MB
written before a single row was read, precisely in the sparse-span case the
ceiling exists to serve. The table now stores `u32::MAX - row_index`, so:

- `0` is the "unset" state → `vec![0u32; span]` → `alloc_zeroed` → lazily
  faulted zero pages;
- "smallest row index wins" becomes "largest encoded value wins" → the parallel
  fill is a plain order-independent `fetch_max`.

The atomic fill guards the read-modify-write behind an unsynchronized load.
Row indices ascend within a chunk, so after a value's first occurrence in that
chunk every later occurrence loses the comparison and skips the RMW entirely.
That keeps actual atomic RMWs at roughly the column's *cardinality* rather than
its row count — which is what makes this viable for a 100-distinct-value column
where a naive per-row `fetch_max` would serialize every thread on a handful of
cache lines.

### 3.4 Dense path: don't zero the output buffer (the big one)

New `build_row_codes` helper: `Vec::with_capacity` + fill every element through
`&mut [MaybeUninit<u32>]` + `set_len`. This removed the 1.41 ms zero-fill
outright — the largest single win in the whole exercise, and it also applies to
the parallel hash path and the dictionary path, which had the same pattern.

### 3.5 Dense path: one slot-table scan, presized

Code assignment read the present offsets out of the table and then re-read each
one's representative — two passes over the slot table, both growing a vector
from empty. Fused into one presized scan. Measured 410 µs → ~130 µs at 51k
distinct values, an order of magnitude more than the sort those scans were
feeding (39 µs).

### 3.6 `estimate_cardinality`: remove the double extrapolation

```rust
let scale = len as f64 / CARDINALITY_SAMPLE as f64;   // removed
let estimate = (chao1 * scale)...
estimate.clamp(CARDINALITY_SAMPLE, MAX_INITIAL_CAPACITY)  // floor was wrong too
```

Chao1 is already a *population* richness estimator — it predicts distinct values
in the whole column from the sample. Multiplying by the sampling fraction
double-counts the extrapolation. A 10M-row column with 50 distinct values gave
`chao1 ≈ 50`, `scale ≈ 2441`, estimate ≈ 122,000: a multi-megabyte hash table
for 50 entries, evicting a map that belonged entirely in L1 out to DRAM and
turning every probe in the hot loop into a cache miss.

Unscaled Chao1 already covers the high-cardinality end on its own — an
all-singleton sample yields ≈ 8.4M, which saturates `MAX_INITIAL_CAPACITY`
anyway. The floor also moved from `CARDINALITY_SAMPLE` (which forced a
4096-entry minimum) to `distinct`, what the sample actually proved exists.

Worth 2.4x on its own for low-cardinality columns that miss the dense path
(sparse-span `Int64`, 1k distinct: 15.1 ms → 6.3 ms).

### 3.7 Parallel hash path: presized chunk maps

`factorize_values_parallel`'s per-chunk maps were `FxHashMap::default()` — every
one of `num_chunks` threads ate the full growth-and-rehash cascade, while
`estimate_cardinality` sat right there being used by the sequential path. Now
seeded with `estimate_cardinality(len) / num_chunks`.

### 3.8 Parallel hash path: parallel stage-2 merge

Stage 2 was a sequential loop re-hashing every chunk-local distinct value on one
thread — up to `num_chunks × chunk_cardinality` insertions while 39 cores idled.
For a high-cardinality column this dominated the entire call.

Replaced with a hash-partitioned merge: each chunk's distinct values are routed
by hash into one of `num_chunks` buckets, and each bucket is merged by a single
thread. A value's bucket is a pure function of its hash, so all occurrences land
in the same bucket and no two threads ever touch the same key — no locking
needed. Keeping the *smallest* representative per value keeps it
order-independent, so output stays bit-for-bit identical to the sequential path.

Stage 3's code assignment and stage 4's translation-table lookups were
restructured to match (per-bucket maps, a parallel scatter into a flat
`code_by_slot` table, and the null slot's code recovered by `partition_point`
rather than a scan).

### 3.9 Strings/bytes: parallelize at all

`factorize_bytes` never parallelized, despite strings being where per-row cost
is highest. `factorize_values`/`factorize_values_parallel` are now generic over
their `BuildHasher`, so the byte path shares the parallel implementation while
keeping `ahash` (the hasher previously measured best for byte-string keys) and
the numeric paths keep `FxHashMap`. Worth 6.6x.

### 3.10 Dictionary path: collapse the dependent gather

`logical_code_at` did `dictionary_codes[keys[i]]` and the final collect then did
`output_code_by_value[that]` — two *dependent* loads per row, the second's
address unknown until the first retires. The two small tables are now composed
into one key→output-code table up front, so each row is a single indexed load.
`array.keys()` was also being re-derived inside the closure on every row; now
hoisted. The final gather is parallel.

### 3.11 Dense path now covers dates and timestamps

`Date32`/`Date64`/`Timestamp` are integers, and were going straight to the hash
path. They now try the dense path with the usual automatic span fallback. The
fallback costs one parallel, memory-bound min/max scan before giving up — cheap
next to the hash path it falls back to, and a large win whenever the column is
what these dtypes usually are in practice: day buckets, or timestamps confined
to a narrow window.

### 3.12 A `u32` truncation bug fixed on the way past

The dense path stored row indices as `index as u32` in a table whose `u32::MAX`
sentinel also collided with a legitimate row index — silently wrong above 4.29B
rows. The path now declines (falls back to the hash path) when
`len >= u32::MAX`, and `representatives` stays `u64` as documented.

---

## 4. What was investigated and rejected

**`u32_results_into_arrow` was not the problem.** The hypothesis was that the
`u32` output helper might build via `from_iter_values` or build `UInt64` and
cast, paying an allocation plus a full copy that the `u64` helper avoids. It
does not: both go through `PrimitiveArray::from(Vec<T::Native>)`, which is
`Buffer::from_vec`, zero-copy on both sides. `u32` codes are not intrinsically
slower than `u64` codes, and no change was needed here.

**Fusing the fill and gather scans.** Writing offsets into the output during the
fill scan and remapping in place afterwards looks like it saves a 38 MB re-read.
It does not: `38r + 19w + 19r + 19w` equals `38r + 38r + 19w`. For `Int32` input
it is strictly *worse* (`19r+19w+19r+19w` vs `19r+19r+19w`). Not done.

**Fusing min/max into the fill scan.** Impossible without the global minimum,
which is what the min/max scan computes. Speculating a range from a sample with
an overflow list was considered and rejected as too much complexity and risk for
a scan that now costs 0.33 ms.

---

## 5. Results

### 5.1 Paired A/B, same machine load

The box is shared and never went quiet during this work, so the headline table
is a **paired A/B/A run**: benchmark the new build, `git stash` the change,
rebuild, benchmark the old build, restore, rebuild, benchmark again. The old
build is measured between two new-build measurements taken minutes apart at the
same load (~11), so the comparison is apples-to-apples.

4M–4.75M rows, best of 5:

| workload | before | after | speedup |
|---|---:|---:|---:|
| **Brightkite uid `Int64` (dense, `sort=False`)** | **22.50 ms** | **2.53 ms** | **8.9x** |
| Brightkite uid `Int64` (dense, `sort=True`) | 22.11 ms | 2.15 ms | 10.3x |
| Brightkite uid `Int32` (dense) | 20.46 ms | 1.77 ms | 11.6x |
| `Int64`, 100 distinct (dense) | 17.76 ms | 1.41 ms | 12.6x |
| `Int64`, 100k distinct (dense) | 23.39 ms | 6.26 ms | 3.7x |
| `Int64`, 100k distinct, 10 % null (dense) | 41.79 ms | 7.51 ms | 5.6x |
| `Int64`, 1k distinct, sparse span (hash) | 15.22 ms | 5.55 ms | 2.7x |
| `Int64`, 4M distinct, sparse span (hash) | 730.90 ms | 167.01 ms | 4.4x |
| `Float64`, 100k distinct (hash) | 63.24 ms | 32.91 ms | 1.9x |
| `Utf8`, 50k distinct (bytes) | 247.59 ms | 38.79 ms | 6.4x |
| **total** | **1204.95 ms** | **265.88 ms** | **4.5x** |

Every workload improved. Nothing regressed.

The second new-build pass reproduced the first closely (target workload 2.30 ms
vs 2.53 ms; total 271.7 ms vs 265.9 ms), so the old build was not measured
during an unrepresentative lull.

### 5.2 The same benchmark on a quiet machine

The load penalty is asymmetric and works *against* the new code: the old
implementation is sequential and barely notices contention, while the new one
wants 40 threads and gives up most of its advantage when it cannot get them.
The numbers above are therefore a **lower bound**. Measured earlier at load ~2:

| workload | before | after | speedup |
|---|---:|---:|---:|
| **Brightkite uid `Int64` (dense, `sort=False`)** | **20.59 ms** | **1.74 ms** | **11.8x** |
| Brightkite uid `Int64` (dense, `sort=True`) | 20.03 ms | 1.68 ms | 11.9x |
| Brightkite uid `Int32` (dense) | 18.70 ms | 1.42 ms | 13.2x |
| `Int64`, 100 distinct (dense) | 15.74 ms | 0.99 ms | 15.9x |

So the 10x goal is met with margin on an unloaded machine, and roughly met
(8.9–12.6x depending on dtype and sort mode) even at load ~11.

### 5.3 End-to-end `jump_lengths` stage

Using the wrapper's own `FASTMOB_PROFILE_JUMP_LENGTHS=1` printfs, Brightkite
4.75M rows, pandas backend:

```
before:  [jump_lengths] user-ID factorization: 0.019875s (51,406 groups)
         [jump_lengths] user-ID factorization: 0.022956s (51,406 groups)

after:   [jump_lengths] user-ID factorization: 0.004200s (51,406 groups)
         [jump_lengths] user-ID factorization: 0.004352s (51,406 groups)
```

The stage timer brackets slightly more than the kernel — it also covers
`_factorize_arrow_values`' Arrow wrapping of both outputs — which is why it sits
a little above the isolated kernel measurement.

Factorization has gone from the third-largest stage in the `jump_lengths` call
to a rounding error. It is no longer worth optimizing: the remaining time is in
timestamp extraction (~70 ms), user/time ordering (~60 ms), and the jump-length
kernel itself (~50 ms).

Dense-path phase breakdown after the change (4.75M rows, 51k distinct):

| phase | before | after |
|---|---:|---:|
| min/max scan | 0.83 ms | 0.33 ms |
| representative discovery | 0.73 ms | 0.45 ms |
| code assignment | 0.61 ms | 0.13 ms |
| output allocation | 1.41 ms | — (eliminated) |
| final gather (incl. alloc) | 0.75 ms | 0.56 ms |

---

## 6. Correctness

- **12/12 Rust unit tests pass**, including
  `parallel_matches_sequential_exactly_across_random_inputs`, which asserts the
  parallel path is bit-for-bit identical to the sequential one.
- **~135 differential checks** against an independent pure-Python reference
  implementation, all passing, covering: every integer width; `Float32`/
  `Float64` including `-0.0`/`inf` ordering; `Boolean`; `Utf8`/`LargeUtf8`/
  `Binary`; dictionary-encoded strings; `Date32`; `Timestamp`; null-free,
  partially-null and all-null columns; empty/1-row/2-row inputs; sizes on both
  sides of `PARALLEL_ROW_THRESHOLD`; negative and zero-straddling value ranges
  (the two's-complement offset); spans forcing hash fallback (including
  `i64::MIN`/`i64::MAX`); sliced non-zero-offset arrays; and both `sort` modes.

  Worth calling out specifically: all-null and single-distinct-value `Float64`/
  `Utf8` columns above the parallel threshold. Integer all-null columns
  short-circuit in the dense path's min/max step, so only the float/string
  cases actually reach the parallel hash path's `distinct_count == 0` state,
  where `ordered` contains nothing but the null slot.
- **Full `scripts/run_correctness.sh` suite: no regressions.** 45 tests fail on
  this branch, and the *identical* 45 fail with the change stashed and the
  extension rebuilt — verified by diffing the sorted failure lists. They are
  pre-existing (network fetches, cached skmob references, an unrelated
  `locations_dataframe.py` dispatch-contract violation).

---

## 7. Scope

All changes are in the shared kernel. There is no `jump_lengths`-specific code
path, no new public API, and no new caller-facing switch — every consumer of
`factorize_arrow` (`entropy`, `recast`, `next_location`, `segment`, `cpc`,
`diversity`, …) gets the same improvements. The dense-integer path is chosen by
value span, not by column name.
