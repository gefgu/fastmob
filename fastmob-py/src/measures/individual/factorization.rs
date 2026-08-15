use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::hash::{BuildHasher, Hash, Hasher};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

use arrow_array::types::*;
use arrow_array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, DictionaryArray, GenericStringArray,
    LargeBinaryArray, LargeStringArray, OffsetSizeTrait, PrimitiveArray, StringArray,
    StringViewArray,
};
use arrow_buffer::{ArrowNativeType, NullBuffer};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::utils::{extract_arrow_array, u64_results_into_arrow};

#[derive(Clone, Copy, Debug)]
struct TotalF64(u64);

impl TotalF64 {
    /// Encodes `value` as a `u64` whose plain integer order matches IEEE-754
    /// total order (`-inf < ... < -0.0 == 0.0 < ... < inf < NaN`), via a
    /// monotonic bit-flip transform computed once here. This lets every
    /// comparison downstream (`Ord::cmp`, and in particular the sort in
    /// [`sort_factorized`]) be a single integer compare instead of
    /// reconstructing an `f64` and calling `total_cmp` per comparison.
    fn new(value: f64) -> Self {
        let bits = if value == 0.0 {
            0.0f64.to_bits()
        } else if value.is_nan() {
            f64::NAN.to_bits()
        } else {
            value.to_bits()
        };
        let key = if (bits as i64) < 0 {
            !bits
        } else {
            bits | (1u64 << 63)
        };
        Self(key)
    }

    #[cfg(test)]
    fn value(self) -> f64 {
        let bits = if (self.0 as i64) < 0 {
            self.0 & !(1u64 << 63)
        } else {
            !self.0
        };
        f64::from_bits(bits)
    }
}

impl PartialEq for TotalF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for TotalF64 {}
impl Hash for TotalF64 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}
impl PartialOrd for TotalF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TotalF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

const CARDINALITY_SAMPLE: usize = 4096;
const MAX_INITIAL_CAPACITY: usize = 1 << 22;

/// Row-count threshold above which the parallel factorization path is used
/// instead of the sequential one. A conservative starting point pending
/// empirical tuning against real workloads -- see the module's companion
/// implementation plan for the open-risk note.
const PARALLEL_ROW_THRESHOLD: usize = 100_000;

/// Cardinality (distinct-value count, not row count) threshold above which
/// [`sort_factorized`]'s value sort itself runs in parallel. Sorting
/// operates on `representatives.len()`, which is typically far smaller than
/// row count, so this needs its own, smaller threshold than
/// [`PARALLEL_ROW_THRESHOLD`].
const SORT_PARALLEL_THRESHOLD: usize = 20_000;

/// Ceiling for the dense-integer fast path's lookup-table span
/// ([`factorize_dense_integers`]), grown with input size instead of being
/// hard-capped at a fixed constant, so a large array with a moderately large
/// span (e.g. bucketed timestamps, sparse-but-bounded IDs) still gets the
/// dense path. Bounds worst-case memory for very large spans; exact tuning
/// is pending benchmarking against real workloads.
const DENSE_SPAN_CEILING: usize = 64 * 1024 * 1024;

#[inline]
fn should_parallelize(len: usize) -> bool {
    len >= PARALLEL_ROW_THRESHOLD && rayon::current_num_threads() > 1
}

fn dense_span_cap(len: usize) -> usize {
    len.saturating_mul(4).min(DENSE_SPAN_CEILING)
}

/// Rayon chunk length for the dense-integer path's three row scans.
///
/// Deliberately much smaller than `len / num_threads`: oversubscribing the
/// thread pool by ~8x lets rayon's work stealing smooth over the fact that
/// these scans are memory-bound and the OS may not schedule all workers
/// evenly, while staying large enough that per-chunk overhead is amortized.
fn dense_chunk_len(len: usize) -> usize {
    len.div_ceil(rayon::current_num_threads().max(1) * 8)
        .max(8192)
}

/// Build the `len`-element `codes` output by filling contiguous chunks,
/// **without** the zero-fill a `vec![0u32; len]` + overwrite would pay for.
///
/// This is worth a dedicated helper because that zero-fill was measured as
/// the single largest cost in the whole dense path -- 1.28 ms of a 3.4 ms
/// 4.75M-row factorization, more than the min/max scan, the representative
/// scan and the final gather *combined*. `vec![0u32; len]` lowers to
/// `alloc_zeroed`, and whenever the allocator satisfies that from an already
/// resident chunk rather than a fresh `mmap` it has to genuinely `memset` the
/// buffer -- 19 MB at ~15 GB/s -- immediately before the gather overwrites
/// every byte of it anyway.
///
/// `fill` must initialize *every* element of each slice it is handed; the
/// chunk loops below hand out each of `len`'s elements exactly once.
fn build_row_codes<F>(len: usize, parallel: bool, fill: F) -> Vec<u32>
where
    F: Fn(usize, &mut [MaybeUninit<u32>]) + Send + Sync,
{
    let mut codes: Vec<u32> = Vec::with_capacity(len);
    let chunk_len = dense_chunk_len(len.max(1));
    {
        let spare = &mut codes.spare_capacity_mut()[..len];
        if parallel {
            spare
                .par_chunks_mut(chunk_len)
                .enumerate()
                .for_each(|(chunk_index, out)| fill(chunk_index * chunk_len, out));
        } else {
            spare
                .chunks_mut(chunk_len)
                .enumerate()
                .for_each(|(chunk_index, out)| fill(chunk_index * chunk_len, out));
        }
    }
    // SAFETY: the loops above partition `spare[..len]` into chunks and visit
    // each exactly once, and `fill`'s contract is that it initializes every
    // element of the slice it receives -- so all `len` elements are init.
    unsafe { codes.set_len(len) };
    codes
}

/// Run `body(chunk_start_index, chunk)` over `values` in [`dense_chunk_len`]
/// slices, in parallel or sequentially, so the dense path's scans are written
/// once instead of twice.
#[inline]
fn for_each_dense_chunk<V, F>(values: &[V], parallel: bool, body: F)
where
    V: Sync,
    F: Fn(usize, &[V]) + Send + Sync,
{
    let chunk_len = dense_chunk_len(values.len().max(1));
    if parallel {
        values
            .par_chunks(chunk_len)
            .enumerate()
            .for_each(|(chunk_index, chunk)| body(chunk_index * chunk_len, chunk));
    } else {
        values
            .chunks(chunk_len)
            .enumerate()
            .for_each(|(chunk_index, chunk)| body(chunk_index * chunk_len, chunk));
    }
}

/// Native-register-width offset arithmetic for [`factorize_dense_integers`].
///
/// Replaces the `T::Native: Into<i128>` bound a previous revision used for
/// *every per-row* offset computation. `i128` is not a native width on
/// x86-64 or aarch64, so each `value - min` compiled to a multi-instruction
/// sequence and blocked autovectorization of both the min/max scan and the
/// final gather -- on a large `Int32`/`Int64` column the min/max pass alone
/// should be pure SIMD and memory-bound. Two's complement makes the
/// native-width form exact whenever the true difference fits in a `u64`,
/// which the `span <= dense_span_cap` check below already guarantees.
trait DenseInt: ArrowNativeType + Ord {
    /// `self - min`, where `min <= self` is guaranteed by construction.
    fn offset_from(self, min: Self) -> u64;
}

macro_rules! impl_dense_int {
    ($($native:ty => $wide:ty),* $(,)?) => {$(
        impl DenseInt for $native {
            #[inline(always)]
            fn offset_from(self, min: Self) -> u64 {
                // Signed types widen with sign extension and unsigned types
                // with zero extension, so the wrapping difference's low 64
                // bits are the true difference in both cases.
                (self as $wide).wrapping_sub(min as $wide) as u64
            }
        }
    )*};
}

impl_dense_int!(
    i8 => i64, i16 => i64, i32 => i64, i64 => i64,
    u8 => u64, u16 => u64, u32 => u64, u64 => u64,
);

/// Encode a row index for the dense path's shared offset table.
///
/// The table stores `u32::MAX - row_index` rather than the row index itself,
/// which buys two things at once:
///
/// * `0` becomes the "unset" state, so the table can be allocated with
///   `vec![0u32; span]` -- which Rust lowers to `alloc_zeroed`/`calloc` and
///   therefore gets lazily-faulted zero pages from the OS. The previous
///   `vec![u32::MAX; span]` forced a real, eager `span * 4`-byte memset, i.e.
///   up to **256 MB** written before a single row was read, precisely in the
///   sparse-span case [`DENSE_SPAN_CEILING`] exists to serve.
/// * "smallest row index wins" becomes "largest encoded value wins", so the
///   parallel fill can use a plain `fetch_max` and stay order-independent.
#[inline(always)]
fn encode_representative(index: usize) -> u32 {
    u32::MAX - index as u32
}

#[inline(always)]
fn decode_representative(encoded: u32) -> u64 {
    (u32::MAX - encoded) as u64
}

/// Record `index` as the representative for `slot` if it precedes whatever is
/// already there.
///
/// The unsynchronized `load` guard matters: row indices ascend within a rayon
/// chunk, so after a value's first occurrence in that chunk every later
/// occurrence loses the comparison and skips the read-modify-write entirely.
/// That keeps the number of actual atomic RMWs at roughly the column's
/// *cardinality* rather than its row count, which is what makes this viable
/// for low-cardinality columns where a naive `fetch_max` per row would
/// serialize every thread on the same handful of cache lines.
#[inline(always)]
fn observe_representative(slot: &AtomicU32, index: usize) {
    let encoded = encode_representative(index);
    if slot.load(AtomicOrdering::Relaxed) < encoded {
        slot.fetch_max(encoded, AtomicOrdering::Relaxed);
    }
}

/// Allocate the dense path's `span`-sized offset table as zeroed atomics,
/// going through `vec![0u32; span]` so the allocation is `calloc`-backed
/// (see [`encode_representative`]).
fn zeroed_atomic_slots(span: usize) -> Vec<AtomicU32> {
    let zeros = vec![0u32; span];
    // SAFETY: `AtomicU32` is `#[repr(C)]` over a `u32` and has identical size
    // and alignment, which is exactly the guarantee `AtomicU32::from_mut_slice`
    // is built on; transmuting the owning `Vec` preserves the allocation's
    // layout, so it stays valid to free through the new element type.
    let (pointer, length, capacity) = {
        let mut zeros = std::mem::ManuallyDrop::new(zeros);
        (zeros.as_mut_ptr(), zeros.len(), zeros.capacity())
    };
    unsafe { Vec::from_raw_parts(pointer.cast::<AtomicU32>(), length, capacity) }
}

/// Assign the next factorization code, panicking loudly rather than
/// silently wrapping if a single column's distinct-value count ever exceeds
/// what fits in a `u32`. `u32::MAX` is reserved as an "unset" sentinel by
/// several of this module's lookup-table fast paths, so valid codes stay
/// strictly below it.
#[inline]
fn next_code(count: usize) -> u32 {
    assert!(
        count < u32::MAX as usize,
        "factorization codes must fit in u32 (column cardinality exceeded u32::MAX - 1)"
    );
    count as u32
}

/// Small, deterministically-seeded PRNG used only to pick sample indices for
/// [`estimate_cardinality`]. Not cryptographic; just needs to avoid the
/// aliasing a fixed-stride walk suffers on periodic data.
struct SampleRng(u64);
impl SampleRng {
    fn next_index(&mut self, bound: usize) -> usize {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((self.0 >> 33) % bound as u64) as usize
    }
}

/// Choose a capacity that avoids both a repeated growth cascade for high-cardinality
/// columns and a huge, mostly-empty allocation for categoricals.
///
/// Samples `CARDINALITY_SAMPLE` pseudo-randomly chosen rows (not a fixed
/// stride, which aliases badly on periodic/cyclical data) and extrapolates
/// via a Chao1 estimator (`distinct + f1^2 / (2*f2)`, from the sample's
/// singleton/doubleton counts). This is only ever used as a capacity *hint*:
/// under- or over-estimating never produces incorrect
/// `codes`/`representatives`, only extra rehashing or wasted memory, so a
/// heuristic estimator is acceptable here.
///
/// Chao1 is already a *population* richness estimator -- it predicts the
/// number of distinct values in the whole column from the sample, so it must
/// **not** additionally be multiplied by the sampling fraction. Doing so
/// (as an earlier revision did) double-counts the extrapolation and
/// over-allocates by orders of magnitude: a 10M-row column with 50 distinct
/// values yields `chao1 ~= 50` but `chao1 * (10M / 4096) ~= 122,000`, i.e. a
/// multi-megabyte table for 50 entries, which evicts a map that belonged
/// entirely in L1 out to DRAM and turns every probe in the hot loop into a
/// cache miss. Unscaled Chao1 already covers the high-cardinality end on its
/// own: an all-singleton sample yields `4096 + 4096*4095/2 ~= 8.4M`, which
/// saturates [`MAX_INITIAL_CAPACITY`] anyway.
fn estimate_cardinality<K, F>(len: usize, value_at: &F) -> usize
where
    K: Eq + Hash,
    F: Fn(usize) -> Option<K>,
{
    if len <= CARDINALITY_SAMPLE {
        return len;
    }
    let mut rng = SampleRng(len as u64 ^ 0x9E37_79B9_7F4A_7C15);
    let mut counts: FxHashMap<Option<K>, u32> =
        FxHashMap::with_capacity_and_hasher(CARDINALITY_SAMPLE, Default::default());
    for _ in 0..CARDINALITY_SAMPLE {
        let index = rng.next_index(len);
        *counts.entry(value_at(index)).or_insert(0) += 1;
    }
    let distinct = counts.len();
    let singletons = counts.values().filter(|&&count| count == 1).count() as f64;
    let doubletons = counts.values().filter(|&&count| count == 2).count() as f64;
    let chao1 = if doubletons > 0.0 {
        distinct as f64 + (singletons * singletons) / (2.0 * doubletons)
    } else {
        distinct as f64 + (singletons * (singletons - 1.0)) / 2.0
    };
    // Floor at `distinct` (what the sample actually proved exists), not at
    // `CARDINALITY_SAMPLE` -- the latter forced a 4096-entry minimum even for
    // a column the sample showed to have a handful of distinct values.
    let estimate = chao1.round().max(0.0) as usize;
    estimate.clamp(distinct, MAX_INITIAL_CAPACITY)
}

/// Hash map used by the generic (non-dense) factorization paths, generic over
/// its `BuildHasher` so numeric and byte-string keys can each keep the hasher
/// measured best for them while sharing one implementation.
///
/// `FxHashMap` for short, fixed-size integer/date/float keys; `ahash` for
/// byte strings -- see [`factorize_bytes`] for the measurements behind that
/// split.
#[allow(clippy::disallowed_types)]
type FactorizeMap<K, S> = std::collections::HashMap<K, u32, S>;

fn factorize_values<K, S, F>(len: usize, sort: bool, value_at: F) -> (Vec<u32>, Vec<u64>)
where
    K: Eq + Hash + Ord + Send,
    S: BuildHasher + Default,
    F: Fn(usize) -> Option<K>,
{
    let mut seen: FactorizeMap<K, S> = FactorizeMap::with_capacity_and_hasher(
        estimate_cardinality(len, &value_at),
        Default::default(),
    );
    let mut null_code: Option<u32> = None;
    let mut codes = Vec::with_capacity(len);
    let mut representatives: Vec<u64> = Vec::new();
    for index in 0..len {
        let code = match value_at(index) {
            Some(value) => match seen.entry(value) {
                Entry::Occupied(entry) => *entry.get(),
                Entry::Vacant(entry) => {
                    let code = next_code(representatives.len());
                    entry.insert(code);
                    representatives.push(index as u64);
                    code
                }
            },
            None => *null_code.get_or_insert_with(|| {
                let code = next_code(representatives.len());
                representatives.push(index as u64);
                code
            }),
        };
        codes.push(code);
    }
    if sort {
        sort_factorized(&mut codes, &mut representatives, &value_at);
    }
    (codes, representatives)
}

/// Parallel counterpart to [`factorize_values`]. Unlike the two-pass
/// build-then-remap design this replaces, output is now **bit-for-bit
/// identical** to [`factorize_values`] for the same input, including code
/// `0` always being the first-encountered distinct value -- not just
/// partition-identical (same rows grouped together, arbitrary code labels).
///
/// Four stages:
/// 1. Split `0..len` into `rayon::current_num_threads()` contiguous chunks.
///    Each chunk builds its own local dictionary in one sequential pass and
///    writes LOCAL codes (dense `0..chunk_distinct_count`) directly into its
///    own slice of the output array. Every row's value is hashed exactly
///    once here; the local dictionary is dropped once the chunk's local
///    representatives are known, since a representative row's value can
///    always be recovered later via `value_at` instead of being stored
///    separately.
/// 2. Merge all chunks' distinct values into one global dictionary
///    (value -> representative row index), keeping the *smallest*
///    representative per value -- i.e. the true first-seen row index across
///    the whole array, deterministically, regardless of merge order. This
///    only hashes each chunk-local distinct value once (not every row), and
///    is a plain sequential loop over the bounded (`num_chunks`-sized)
///    per-chunk results rather than a tree reduce, since there's nothing
///    fine-grained left to balance at that point.
/// 3. Assign final codes in first-seen order by sorting `(representative,
///    value)` pairs on `representative` -- this reproduces exactly the
///    order a sequential scan would assign codes in.
/// 4. For each chunk, build a small local-code -> global-code translation
///    table (one hash lookup per chunk-local distinct value) and gather
///    every row's global code through it -- a pure array index per row, no
///    per-row hashing.
fn factorize_values_parallel<K, S, F>(len: usize, sort: bool, value_at: F) -> (Vec<u32>, Vec<u64>)
where
    // `Copy` so stage 2 can route a value into a bucket and still store it as
    // that bucket's key. Every key type this module uses -- the Arrow native
    // integer types, [`TotalF64`], and `&[u8]` -- is already `Copy`.
    K: Eq + Hash + Ord + Copy + Send + Sync,
    S: BuildHasher + Default + Send + Sync,
    F: Fn(usize) -> Option<K> + Sync,
{
    if len == 0 {
        return (Vec::new(), Vec::new());
    }

    let num_chunks = rayon::current_num_threads().max(1);
    let chunk_len = len.div_ceil(num_chunks).max(1);

    // A capacity hint per chunk. The chunk maps used to start empty and eat
    // the full growth-and-rehash cascade on every one of `num_chunks`
    // threads, even though `estimate_cardinality` was right there being used
    // by the sequential path.
    let chunk_capacity = estimate_cardinality(len, &value_at)
        .div_ceil(num_chunks)
        .max(64);

    // `Vec::with_capacity` + `set_len`, not `vec![0u32; len]`: stage 1 writes
    // every element, so pre-zeroing the whole output is a wasted `memset` of
    // `4 * len` bytes -- see [`build_row_codes`] for the measurement.
    let mut codes: Vec<u32> = Vec::with_capacity(len);

    // Stage 1.
    let chunk_locals: Vec<(Option<u32>, Vec<u64>)> = {
        let spare = &mut codes.spare_capacity_mut()[..len];
        spare
            .par_chunks_mut(chunk_len)
            .enumerate()
            .map(|(chunk_index, slots)| {
                let base = chunk_index * chunk_len;
                let mut local_seen: FactorizeMap<K, S> =
                    FactorizeMap::with_capacity_and_hasher(chunk_capacity, Default::default());
                let mut local_null: Option<u32> = None;
                let mut local_representatives: Vec<u64> = Vec::new();
                for (offset, slot) in slots.iter_mut().enumerate() {
                    let index = base + offset;
                    slot.write(match value_at(index) {
                        Some(value) => match local_seen.entry(value) {
                            Entry::Occupied(entry) => *entry.get(),
                            Entry::Vacant(entry) => {
                                let code = next_code(local_representatives.len());
                                entry.insert(code);
                                local_representatives.push(index as u64);
                                code
                            }
                        },
                        None => *local_null.get_or_insert_with(|| {
                            let code = next_code(local_representatives.len());
                            local_representatives.push(index as u64);
                            code
                        }),
                    });
                }
                (local_null, local_representatives)
            })
            .collect()
    };
    // SAFETY: `par_chunks_mut` partitions `spare[..len]` and the loop above
    // writes every element of every chunk exactly once.
    unsafe { codes.set_len(len) };

    // Stage 2: merge the chunk-local dictionaries into one global dictionary
    // (value -> smallest representative row index), **in parallel**.
    //
    // This used to be a sequential loop that re-hashed every chunk-local
    // distinct value on one thread -- up to `num_chunks * chunk_cardinality`
    // hash insertions while every other core idled, which for a
    // high-cardinality column dominated the whole call. Instead, each
    // chunk's distinct values are routed by hash into one of `num_chunks`
    // buckets, and each bucket is then merged by a single thread. Because a
    // value's bucket is a pure function of its hash, all occurrences of a
    // value land in the same bucket and no two threads ever touch the same
    // key -- so this needs no locking and, keeping the *smallest*
    // representative per value, stays order-independent and therefore
    // bit-for-bit identical to the sequential path.
    let bucket_hasher = S::default();
    let bucket_of = |value: &K| (bucket_hasher.hash_one(value) as usize) % num_chunks;

    let scattered: Vec<Vec<Vec<(K, u64)>>> = chunk_locals
        .par_iter()
        .map(|(_, local_representatives)| {
            let mut buckets: Vec<Vec<(K, u64)>> = (0..num_chunks).map(|_| Vec::new()).collect();
            for &representative in local_representatives {
                if let Some(value) = value_at(representative as usize) {
                    buckets[bucket_of(&value)].push((value, representative));
                }
            }
            buckets
        })
        .collect();

    // Per bucket: `value -> slot within this bucket`, plus `slot ->
    // representative`. Slots become global by adding the bucket's offset.
    let buckets: Vec<(FactorizeMap<K, S>, Vec<u64>)> = (0..num_chunks)
        .into_par_iter()
        .map(|bucket_index| {
            let mut map: FactorizeMap<K, S> = FactorizeMap::default();
            let mut bucket_representatives: Vec<u64> = Vec::new();
            for chunk in &scattered {
                for &(value, representative) in &chunk[bucket_index] {
                    match map.entry(value) {
                        Entry::Occupied(entry) => {
                            let slot = *entry.get() as usize;
                            bucket_representatives[slot] =
                                bucket_representatives[slot].min(representative);
                        }
                        Entry::Vacant(entry) => {
                            entry.insert(next_code(bucket_representatives.len()));
                            bucket_representatives.push(representative);
                        }
                    }
                }
            }
            (map, bucket_representatives)
        })
        .collect();

    let mut bucket_offsets: Vec<u32> = Vec::with_capacity(num_chunks + 1);
    let mut distinct_count: usize = 0;
    for (_, bucket_representatives) in &buckets {
        bucket_offsets.push(next_code(distinct_count));
        distinct_count += bucket_representatives.len();
    }
    bucket_offsets.push(next_code(distinct_count));

    let global_null = chunk_locals
        .iter()
        .filter_map(|(local_null, local_representatives)| {
            local_null.map(|code| local_representatives[code as usize])
        })
        .min();

    // Stage 3: assign final codes in first-seen order by sorting on the
    // representative row index -- exactly the order a sequential scan assigns.
    const NULL_SLOT: u32 = u32::MAX;
    let mut ordered: Vec<(u64, u32)> = buckets
        .par_iter()
        .enumerate()
        .flat_map(|(bucket_index, (_, bucket_representatives))| {
            let base = bucket_offsets[bucket_index];
            bucket_representatives
                .par_iter()
                .enumerate()
                .map(move |(slot, &representative)| (representative, base + slot as u32))
        })
        .collect();
    if let Some(representative) = global_null {
        ordered.push((representative, NULL_SLOT));
    }
    if ordered.len() > SORT_PARALLEL_THRESHOLD {
        ordered.par_sort_unstable_by_key(|&(representative, _)| representative);
    } else {
        ordered.sort_unstable_by_key(|&(representative, _)| representative);
    }

    // Representative row indices are distinct, so the null slot's final code
    // is just its rank -- no scan needed to find where it landed.
    let null_code = global_null.map(|representative| {
        next_code(ordered.partition_point(|&(other, _)| other < representative))
    });
    let representatives: Vec<u64> = ordered
        .iter()
        .map(|&(representative, _)| representative)
        .collect();

    // `code_by_slot[global_slot]` is written exactly once, by whichever
    // thread owns that position in `ordered`, so a relaxed atomic store is
    // enough to make the parallel scatter well-defined.
    let code_by_slot = zeroed_atomic_slots(distinct_count);
    ordered
        .par_iter()
        .enumerate()
        .for_each(|(code, &(_, global_slot))| {
            if global_slot != NULL_SLOT {
                code_by_slot[global_slot as usize].store(next_code(code), AtomicOrdering::Relaxed);
            }
        });

    // Stage 4: per chunk, build a local-code -> global-code translation table
    // (one hash lookup per chunk-local distinct value) and gather every row's
    // global code through it -- a pure array index per row, no per-row hashing.
    codes
        .par_chunks_mut(chunk_len)
        .zip(chunk_locals.into_par_iter())
        .for_each(|(slots, (local_null, local_representatives))| {
            let translation: Vec<u32> = local_representatives
                .iter()
                .enumerate()
                .map(|(local_code, &representative)| {
                    if local_null == Some(local_code as u32) {
                        null_code.expect("a null local code must have a global null code")
                    } else {
                        let value = value_at(representative as usize)
                            .expect("non-null local code must have a value");
                        let bucket = bucket_of(&value);
                        let (map, _) = &buckets[bucket];
                        let global_slot = map[&value] + bucket_offsets[bucket];
                        code_by_slot[global_slot as usize].load(AtomicOrdering::Relaxed)
                    }
                })
                .collect();
            for slot in slots.iter_mut() {
                *slot = translation[*slot as usize];
            }
        });

    let mut representatives = representatives;
    if sort {
        sort_factorized(&mut codes, &mut representatives, &value_at);
    }
    (codes, representatives)
}

/// Convert first-seen factorization output to value-sorted codes without a second hash pass.
fn sort_factorized<K, F>(codes: &mut [u32], representatives: &mut Vec<u64>, value_at: &F)
where
    K: Ord + Send,
    F: Fn(usize) -> Option<K>,
{
    // Partition into non-null candidates (sorted with a plain `K::cmp`, no
    // per-comparison `Option` branch) and the at-most-one null slot, which
    // always sorts last -- rather than special-casing `None` on every
    // comparison inside the sort itself.
    let mut non_null: Vec<(usize, K)> = Vec::with_capacity(representatives.len());
    let mut null_old_code: Option<usize> = None;
    for (old_code, &representative) in representatives.iter().enumerate() {
        match value_at(representative as usize) {
            Some(value) => non_null.push((old_code, value)),
            None => null_old_code = Some(old_code),
        }
    }

    if non_null.len() > SORT_PARALLEL_THRESHOLD {
        non_null.par_sort_unstable_by(|(_, left), (_, right)| left.cmp(right));
    } else {
        non_null.sort_unstable_by(|(_, left), (_, right)| left.cmp(right));
    }

    let mut ranks = vec![0u32; representatives.len()];
    let mut new_representatives = Vec::with_capacity(representatives.len());
    for (old_code, _) in &non_null {
        ranks[*old_code] = next_code(new_representatives.len());
        new_representatives.push(representatives[*old_code]);
    }
    if let Some(old_code) = null_old_code {
        ranks[old_code] = next_code(new_representatives.len());
        new_representatives.push(representatives[old_code]);
    }

    if should_parallelize(codes.len()) {
        codes
            .par_iter_mut()
            .for_each(|code| *code = ranks[*code as usize]);
    } else {
        codes
            .iter_mut()
            .for_each(|code| *code = ranks[*code as usize]);
    }
    *representatives = new_representatives;
}

/// Factorize byte-string-keyed data (`&[u8]`). The single real
/// implementation behind both [`factorize_strings`] (a zero-cost
/// `str::as_bytes` adapter over this) and the `Binary`/`LargeBinary`/
/// `BinaryView` Arrow dtypes -- `&str`'s `Ord` is itself defined as its
/// UTF-8 bytes' lexicographic order, so this doesn't change sort behavior
/// for the string dtypes, and it's the only sensible ordering for the
/// non-UTF-8-guaranteed binary dtypes anyway.
/// `ahash`, not `FxHashMap`, on purpose: measured (not assumed) on a 4M-row
/// synthetic string workload at 3 cardinalities (50 / 5,000 / 500,000 distinct
/// values), `ahash` beat `FxHashMap` by ~5-12% on byte-string keys, while
/// `foldhash` was consistently ~5-15% *slower* than `FxHashMap`. The numeric
/// key paths keep [`NumericHasher`], which remains the better choice for
/// short, fixed-size integer/date/float keys.
type BytesHasher = ahash::RandomState;

/// Hasher for the fixed-size integer/date/float keys -- see [`BytesHasher`].
type NumericHasher = rustc_hash::FxBuildHasher;

fn factorize_bytes<'a, F>(len: usize, sort: bool, value_at: F) -> (Vec<u32>, Vec<u64>)
where
    F: Fn(usize) -> Option<&'a [u8]> + Sync,
{
    // Strings are where the per-row hashing cost is *highest*, yet this path
    // used to be the only one that never parallelized at all.
    if should_parallelize(len) {
        factorize_values_parallel::<_, BytesHasher, _>(len, sort, value_at)
    } else {
        factorize_values::<_, BytesHasher, _>(len, sort, value_at)
    }
}

fn factorize_strings<'a, F>(len: usize, sort: bool, value_at: F) -> (Vec<u32>, Vec<u64>)
where
    F: Fn(usize) -> Option<&'a str> + Sync,
{
    factorize_bytes(len, sort, move |index| value_at(index).map(str::as_bytes))
}

/// Sequential/null-count-specialized dispatch shared by every primitive
/// integer/date/timestamp dtype once (if applicable) the dense-integer fast
/// path has been ruled out. Specializing `value_at` on `array.null_count()
/// == 0` removes a per-row `is_valid` branch on the (very common)
/// fully-populated-column case, which also unblocks autovectorization of
/// the gather.
fn factorize_primitive_values<T>(array: &PrimitiveArray<T>, sort: bool) -> (Vec<u32>, Vec<u64>)
where
    T: ArrowPrimitiveType,
    T::Native: Eq + Hash + Ord + Send + Sync,
{
    if array.null_count() == 0 {
        let value_at = move |index: usize| Some(array.value(index));
        if should_parallelize(array.len()) {
            factorize_values_parallel::<_, NumericHasher, _>(array.len(), sort, value_at)
        } else {
            factorize_values::<_, NumericHasher, _>(array.len(), sort, value_at)
        }
    } else {
        let value_at = move |index: usize| array.is_valid(index).then(|| array.value(index));
        if should_parallelize(array.len()) {
            factorize_values_parallel::<_, NumericHasher, _>(array.len(), sort, value_at)
        } else {
            factorize_values::<_, NumericHasher, _>(array.len(), sort, value_at)
        }
    }
}

/// Same null-count specialization as [`factorize_primitive_values`], for the
/// `Float32`/`Float64` dtypes, which need their values wrapped in
/// [`TotalF64`] to be hashable/orderable.
fn factorize_float_values<T>(array: &PrimitiveArray<T>, sort: bool) -> (Vec<u32>, Vec<u64>)
where
    T: ArrowPrimitiveType,
    T::Native: Into<f64>,
{
    if array.null_count() == 0 {
        let value_at = move |index: usize| Some(TotalF64::new(array.value(index).into()));
        if should_parallelize(array.len()) {
            factorize_values_parallel::<_, NumericHasher, _>(array.len(), sort, value_at)
        } else {
            factorize_values::<_, NumericHasher, _>(array.len(), sort, value_at)
        }
    } else {
        let value_at = move |index: usize| {
            array
                .is_valid(index)
                .then(|| TotalF64::new(array.value(index).into()))
        };
        if should_parallelize(array.len()) {
            factorize_values_parallel::<_, NumericHasher, _>(array.len(), sort, value_at)
        } else {
            factorize_values::<_, NumericHasher, _>(array.len(), sort, value_at)
        }
    }
}

/// Smallest and largest non-null value, folded in the array's **native**
/// integer width and (above [`PARALLEL_ROW_THRESHOLD`]) across rayon chunks.
///
/// Returns `None` only when every row is null.
fn dense_min_max<N>(values: &[N], nulls: Option<&NullBuffer>, parallel: bool) -> Option<(N, N)>
where
    N: DenseInt,
{
    if values.is_empty() {
        return None;
    }
    let combine = |(left_min, left_max): (N, N), (right_min, right_max): (N, N)| {
        (left_min.min(right_min), left_max.max(right_max))
    };
    let chunk_len = dense_chunk_len(values.len());

    let Some(nulls) = nulls else {
        // No validity bitmap: a branchless fold the compiler can vectorize.
        let fold = |chunk: &[N]| {
            let first = chunk[0];
            chunk
                .iter()
                .copied()
                .fold((first, first), |(low, high), value| {
                    (low.min(value), high.max(value))
                })
        };
        return if parallel {
            values.par_chunks(chunk_len).map(fold).reduce_with(combine)
        } else {
            Some(fold(values))
        };
    };

    let fold = |base: usize, chunk: &[N]| -> Option<(N, N)> {
        let mut range: Option<(N, N)> = None;
        for (offset, &value) in chunk.iter().enumerate() {
            if nulls.is_valid(base + offset) {
                range = Some(match range {
                    Some((low, high)) => (low.min(value), high.max(value)),
                    None => (value, value),
                });
            }
        }
        range
    };
    if parallel {
        values
            .par_chunks(chunk_len)
            .enumerate()
            .filter_map(|(chunk_index, chunk)| fold(chunk_index * chunk_len, chunk))
            .reduce_with(combine)
    } else {
        fold(0, values)
    }
}

/// Offset value standing in for "the null slot" in the dense path's
/// first-seen ordering. Safe as a sentinel because a real offset is always
/// `< span <= DENSE_SPAN_CEILING` (64Mi), far below `u32::MAX`.
const DENSE_NULL_OFFSET: u32 = u32::MAX;

/// Direct-indexed fast path for integer columns whose value span
/// (`max - min + 1`) is small enough to use a `Vec` instead of a hash map.
///
/// `always_dense` bypasses the span-vs-`dense_span_cap` check entirely --
/// used for `Int8`/`UInt8`/`Int16`/`UInt16`, whose span is statically
/// bounded (≤256 / ≤65536) regardless of array length, so the dense path is
/// always affordable there.
///
/// Three scans over the rows -- min/max, representative discovery, and the
/// final code gather -- each of which runs in parallel above
/// [`PARALLEL_ROW_THRESHOLD`] and is specialized on `null_count() == 0` so
/// the (very common) fully-populated column pays no per-row validity branch.
/// This is by far the most parallelizable code in the module and, until this
/// revision, was the only path that never consulted [`should_parallelize`] at
/// all.
fn factorize_dense_integers<T>(
    array: &PrimitiveArray<T>,
    sort: bool,
    always_dense: bool,
) -> Option<(Vec<u32>, Vec<u64>)>
where
    T: ArrowPrimitiveType,
    T::Native: DenseInt,
{
    let len = array.len();
    if len == 0 {
        return Some((Vec::new(), Vec::new()));
    }
    // Row indices are stored `u32`-encoded in the offset table below. Hand
    // anything wider to the hash path instead of truncating silently, which
    // is what a previous revision did (it stored `index as u32` into a table
    // whose `u32::MAX` sentinel also collided with a legitimate row index).
    if len >= u32::MAX as usize {
        return None;
    }

    let values: &[T::Native] = array.values();
    let nulls = array.nulls();
    let parallel = should_parallelize(len);

    let Some((min, max)) = dense_min_max(values, nulls, parallel) else {
        return Some((vec![0; len], vec![0]));
    };
    let max_offset = max.offset_from(min);
    if max_offset == u64::MAX {
        return None;
    }
    let span = usize::try_from(max_offset + 1).ok()?;
    if !always_dense && span > dense_span_cap(len) {
        return None;
    }

    // A single `span`-sized table, used first for representative row indices
    // and then overwritten in place with the final codes. A previous revision
    // kept two separate `span`-sized tables even though no offset ever needs
    // both at once, doubling peak memory -- which matters, because
    // `DENSE_SPAN_CEILING` permits 64Mi entries, i.e. 256 MB per table.
    let slots = zeroed_atomic_slots(span);
    let null_slot = AtomicU32::new(0);

    match nulls {
        None => for_each_dense_chunk(values, parallel, |base, chunk| {
            for (offset, &value) in chunk.iter().enumerate() {
                observe_representative(&slots[value.offset_from(min) as usize], base + offset);
            }
        }),
        Some(nulls) => for_each_dense_chunk(values, parallel, |base, chunk| {
            for (offset, &value) in chunk.iter().enumerate() {
                let index = base + offset;
                if nulls.is_valid(index) {
                    observe_representative(&slots[value.offset_from(min) as usize], index);
                } else {
                    observe_representative(&null_slot, index);
                }
            }
        }),
    }

    // Every offset that occurs at least once, paired with its representative
    // row index, in ascending offset order -- which, for a direct-indexed
    // table, is also ascending *value* order.
    //
    // One scan, not two. Reading the offsets out and then re-reading each
    // one's representative (as a previous revision did) cost two passes over
    // the slot table for no benefit, and both vectors grew from empty:
    // together those measured ~410us of a 1.8ms call at 51k distinct values,
    // an order of magnitude more than the sort they were feeding.
    let read_slot = |(offset, slot): (usize, &AtomicU32)| match slot.load(AtomicOrdering::Relaxed) {
        0 => None,
        encoded => Some((decode_representative(encoded), offset as u32)),
    };
    let mut candidates: Vec<(u64, u32)> = if parallel && span >= PARALLEL_ROW_THRESHOLD {
        slots.par_iter().enumerate().filter_map(read_slot).collect()
    } else {
        // Reserving the whole span would be counterproductive for a sparse
        // one (`DENSE_SPAN_CEILING` permits 64Mi offsets); cap the guess so a
        // dense, low-cardinality column -- the common case -- still avoids the
        // realloc cascade entirely.
        let mut candidates = Vec::with_capacity(span.min(1 << 20));
        candidates.extend(slots.iter().enumerate().filter_map(read_slot));
        candidates
    };
    let null_representative = match null_slot.load(AtomicOrdering::Relaxed) {
        0 => None,
        encoded => Some(decode_representative(encoded)),
    };

    let mut representatives: Vec<u64> = Vec::with_capacity(candidates.len() + 1);
    let null_code;
    if sort {
        // Ascending offset order is already ascending value order, so codes
        // fall out of a single pass with no sorting at all.
        for &(representative, offset) in &candidates {
            representatives.push(representative);
            let code = next_code(representatives.len() - 1);
            slots[offset as usize].store(code, AtomicOrdering::Relaxed);
        }
        null_code = null_representative.map(|representative| {
            let code = next_code(representatives.len());
            representatives.push(representative);
            code
        });
    } else {
        // First-seen order: order the present offsets (plus the null slot, if
        // any) by representative row index -- the same trick the generic
        // parallel path uses to reproduce sequential-scan order. Every slot is
        // read into `candidates` before any code is written back, so reusing
        // the table for codes cannot clobber a representative still needed.
        if let Some(representative) = null_representative {
            candidates.push((representative, DENSE_NULL_OFFSET));
        }
        if parallel && candidates.len() > SORT_PARALLEL_THRESHOLD {
            candidates.par_sort_unstable_by_key(|&(representative, _)| representative);
        } else {
            candidates.sort_unstable_by_key(|&(representative, _)| representative);
        }

        let mut assigned_null_code = None;
        for (representative, offset) in candidates {
            let code = next_code(representatives.len());
            representatives.push(representative);
            if offset == DENSE_NULL_OFFSET {
                assigned_null_code = Some(code);
            } else {
                slots[offset as usize].store(code, AtomicOrdering::Relaxed);
            }
        }
        null_code = assigned_null_code;
    }

    let null_code = null_code.unwrap_or(0);
    let codes = build_row_codes(len, parallel, |base, out| match nulls {
        None => {
            for (offset, code) in out.iter_mut().enumerate() {
                let value = values[base + offset];
                code.write(slots[value.offset_from(min) as usize].load(AtomicOrdering::Relaxed));
            }
        }
        Some(nulls) => {
            for (offset, code) in out.iter_mut().enumerate() {
                let index = base + offset;
                code.write(if nulls.is_valid(index) {
                    slots[values[index].offset_from(min) as usize].load(AtomicOrdering::Relaxed)
                } else {
                    null_code
                });
            }
        }
    });

    Some((codes, representatives))
}

fn factorize_boolean(array: &BooleanArray, sort: bool) -> (Vec<u32>, Vec<u64>) {
    let state_at = |index: usize| match array.is_valid(index) {
        false => 2usize,
        true if array.value(index) => 1,
        true => 0,
    };
    let mut codes_by_state = [u32::MAX; 3];
    let mut representatives = Vec::new();
    // There are only ever 3 possible states, so once all 3 have been seen
    // the remaining rows can't teach the bookkeeping loop anything new.
    if sort {
        let mut representatives_by_state = [usize::MAX; 3];
        let mut found = 0;
        for index in 0..array.len() {
            let state = state_at(index);
            if representatives_by_state[state] == usize::MAX {
                representatives_by_state[state] = index;
                found += 1;
                if found == 3 {
                    break;
                }
            }
        }
        for state in [0, 1, 2] {
            if representatives_by_state[state] != usize::MAX {
                codes_by_state[state] = next_code(representatives.len());
                representatives.push(representatives_by_state[state] as u64);
            }
        }
    } else {
        let mut found = 0;
        for index in 0..array.len() {
            let state = state_at(index);
            if codes_by_state[state] == u32::MAX {
                codes_by_state[state] = next_code(representatives.len());
                representatives.push(index as u64);
                found += 1;
                if found == 3 {
                    break;
                }
            }
        }
    }
    // The bookkeeping loops above can stop early once all states are found,
    // but every row still needs its own code, so this final gather always
    // touches the whole array -- specialized on `null_count() == 0` so the
    // common fully-populated case skips the `is_valid` check per row.
    let codes = if array.null_count() == 0 {
        (0..array.len())
            .map(|index| codes_by_state[usize::from(array.value(index))])
            .collect()
    } else {
        (0..array.len())
            .map(|index| codes_by_state[state_at(index)])
            .collect()
    };
    (codes, representatives)
}

/// Factorize a string dictionary through its key array. The dictionary
/// values are factorized only once (including duplicate values), then the
/// row scan is made of small integer gathers rather than string hashes.
/// Generic over `O: OffsetSizeTrait` so both `Utf8`- and `LargeUtf8`-valued
/// dictionaries share this implementation.
fn factorize_string_dictionary<K, O>(array: &DictionaryArray<K>, sort: bool) -> (Vec<u32>, Vec<u64>)
where
    K: ArrowDictionaryKeyType,
    K::Native: ArrowNativeType,
    O: OffsetSizeTrait,
{
    let values = array
        .values()
        .as_any()
        .downcast_ref::<GenericStringArray<O>>()
        .expect("dictionary dispatch and values must agree");
    let (dictionary_codes, dictionary_representatives) =
        factorize_strings(values.len(), sort, |index| {
            values.is_valid(index).then(|| values.value(index))
        });
    let dictionary_null_code = (0..values.len())
        .find(|&index| !values.is_valid(index))
        .map(|index| dictionary_codes[index] as usize);
    let virtual_null_code = dictionary_representatives.len();
    let null_code = dictionary_null_code.unwrap_or(virtual_null_code);
    let logical_code_count =
        dictionary_representatives.len() + usize::from(dictionary_null_code.is_none());

    // Hoisted: `array.keys()` re-derives the key array from the dictionary's
    // layout, and a previous revision called it *inside* the closure below,
    // i.e. once per row.
    let keys = array.keys();

    // Recompute a row's dictionary-relative logical code from its key on
    // demand rather than materializing an `array.len()`-sized intermediate
    // buffer: `dictionary_codes` is small (bounded by dictionary
    // cardinality) and index-only, so recomputation is cheaper than the
    // extra allocation and memory round-trip a stored buffer would cost.
    let logical_code_at = |index: usize| -> usize {
        if array.is_valid(index) {
            dictionary_codes[keys.value(index).as_usize()] as usize
        } else {
            null_code
        }
    };

    let mut output_code_by_value = vec![u32::MAX; logical_code_count];
    let mut representatives = Vec::new();
    if sort {
        // Only needed for the sort branch -- building this unconditionally
        // (as a prior version of this function did) was a full O(len) pass
        // wasted whenever `sort == false`.
        let mut representatives_by_value = vec![usize::MAX; logical_code_count];
        for index in 0..array.len() {
            let code = logical_code_at(index);
            if representatives_by_value[code] == usize::MAX {
                representatives_by_value[code] = index;
            }
        }
        for code in 0..dictionary_representatives.len() {
            if code == null_code {
                continue;
            }
            if representatives_by_value[code] != usize::MAX {
                output_code_by_value[code] = next_code(representatives.len());
                representatives.push(representatives_by_value[code] as u64);
            }
        }
        if representatives_by_value[null_code] != usize::MAX {
            output_code_by_value[null_code] = next_code(representatives.len());
            representatives.push(representatives_by_value[null_code] as u64);
        }
    } else {
        for index in 0..array.len() {
            let code = logical_code_at(index);
            if output_code_by_value[code] == u32::MAX {
                output_code_by_value[code] = next_code(representatives.len());
                representatives.push(index as u64);
            }
        }
    }
    // Collapse the row scan's two *dependent* loads --
    // `output_code_by_value[dictionary_codes[key]]` -- into one, by composing
    // the two small tables into a single key -> output-code table up front.
    // The composed table is dictionary-sized, so it usually stays cache-hot,
    // and each row now costs a single indexed load instead of a load whose
    // address is not known until the previous load retires.
    let output_code_by_key: Vec<u32> = dictionary_codes
        .iter()
        .map(|&code| output_code_by_value[code as usize])
        .collect();
    let null_output_code = output_code_by_value[null_code];

    let parallel = should_parallelize(array.len());
    let codes = build_row_codes(array.len(), parallel, |base, out| {
        if array.null_count() == 0 {
            for (offset, code) in out.iter_mut().enumerate() {
                code.write(output_code_by_key[keys.value(base + offset).as_usize()]);
            }
        } else {
            for (offset, code) in out.iter_mut().enumerate() {
                let index = base + offset;
                code.write(if array.is_valid(index) {
                    output_code_by_key[keys.value(index).as_usize()]
                } else {
                    null_output_code
                });
            }
        }
    });
    (codes, representatives)
}

macro_rules! factorize_integer {
    ($array:expr, $type:ty, $sort:expr, $always_dense:expr) => {{
        let array = $array
            .as_any()
            .downcast_ref::<PrimitiveArray<$type>>()
            .expect("Arrow data type and primitive array must agree");
        factorize_dense_integers(array, $sort, $always_dense)
            .unwrap_or_else(|| factorize_primitive_values(array, $sort))
    }};
}

macro_rules! factorize_string_dictionary {
    ($array:expr, $key_type:ty, $offset:ty, $sort:expr) => {{
        let array = $array
            .as_any()
            .downcast_ref::<DictionaryArray<$key_type>>()
            .expect("Arrow data type and dictionary array must agree");
        factorize_string_dictionary::<$key_type, $offset>(array, $sort)
    }};
}

/// Factorize a supported Arrow array without crossing the Python boundary.
///
/// This is shared by Arrow-native kernels that need categorical codes but
/// must keep dataframe adapters out of their hot path. `codes` is `u32`
/// (bounded by column cardinality, which is never realistically
/// `> u32::MAX`); `representatives` stays `u64` (row indices, not bounded by
/// cardinality). `codes` remain `u32` at the [`factorize_arrow`] Python
/// boundary; `representatives` remain `u64` because they are row indices.
///
/// The generic hashable-key path (integers with a too-large-to-densify
/// span, floats, dates, timestamps) automatically dispatches to the
/// parallel implementation once `array.len()` clears
/// [`PARALLEL_ROW_THRESHOLD`] -- see [`factorize_values_parallel`]'s docs
/// for why that's now safe to do unconditionally (its output is bit-for-bit
/// identical to the sequential path). There is no caller-facing parallel
/// switch; the dense-integer fast path, booleans, strings/bytes, and
/// dictionary-encoded strings have their own dedicated (and, where
/// applicable, independently threshold-gated) implementations.
///
/// Supported dtypes: all integer widths (`Int8`..`UInt64`), `Float32`/
/// `Float64`, `Boolean`, `Date32`/`Date64`, `Timestamp` (any unit),
/// `Utf8`/`LargeUtf8`/`Utf8View`, `Binary`/`LargeBinary`/`BinaryView`, and
/// `Utf8`- or `LargeUtf8`-valued `Dictionary` arrays keyed by any integer
/// type.
pub(crate) fn factorize_array(
    array: &dyn Array,
    sort: bool,
) -> Result<(Vec<u32>, Vec<u64>), String> {
    use arrow_schema::DataType;

    let result = match array.data_type() {
        DataType::Int8 => factorize_integer!(array, Int8Type, sort, true),
        DataType::Int16 => factorize_integer!(array, Int16Type, sort, true),
        DataType::Int32 => factorize_integer!(array, Int32Type, sort, false),
        DataType::Int64 => factorize_integer!(array, Int64Type, sort, false),
        DataType::UInt8 => factorize_integer!(array, UInt8Type, sort, true),
        DataType::UInt16 => factorize_integer!(array, UInt16Type, sort, true),
        DataType::UInt32 => factorize_integer!(array, UInt32Type, sort, false),
        DataType::UInt64 => factorize_integer!(array, UInt64Type, sort, false),
        // Dates and timestamps are integers, so they get the dense fast path
        // too, with the usual automatic fallback when the span is too wide to
        // densify. That fallback now costs one (parallel, memory-bound)
        // min/max scan before it gives up -- cheap next to the hash path it
        // falls back to, and a large win whenever the column is what these
        // dtypes usually are in practice: day buckets, or timestamps confined
        // to a narrow window.
        DataType::Date32 => factorize_integer!(array, Date32Type, sort, false),
        DataType::Date64 => factorize_integer!(array, Date64Type, sort, false),
        DataType::Timestamp(unit, _) => match unit {
            arrow_schema::TimeUnit::Second => {
                factorize_integer!(array, TimestampSecondType, sort, false)
            }
            arrow_schema::TimeUnit::Millisecond => {
                factorize_integer!(array, TimestampMillisecondType, sort, false)
            }
            arrow_schema::TimeUnit::Microsecond => {
                factorize_integer!(array, TimestampMicrosecondType, sort, false)
            }
            arrow_schema::TimeUnit::Nanosecond => {
                factorize_integer!(array, TimestampNanosecondType, sort, false)
            }
        },
        DataType::Float32 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<Float32Type>>()
                .unwrap();
            factorize_float_values(array, sort)
        }
        DataType::Float64 => {
            let array = array
                .as_any()
                .downcast_ref::<PrimitiveArray<Float64Type>>()
                .unwrap();
            factorize_float_values(array, sort)
        }
        DataType::Boolean => {
            let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
            factorize_boolean(array, sort)
        }
        DataType::Utf8 => {
            let array = array.as_any().downcast_ref::<StringArray>().unwrap();
            factorize_strings(array.len(), sort, |index| {
                array.is_valid(index).then(|| array.value(index))
            })
        }
        DataType::LargeUtf8 => {
            let array = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
            factorize_strings(array.len(), sort, |index| {
                array.is_valid(index).then(|| array.value(index))
            })
        }
        DataType::Utf8View => {
            let array = array.as_any().downcast_ref::<StringViewArray>().unwrap();
            factorize_strings(array.len(), sort, |index| {
                array.is_valid(index).then(|| array.value(index))
            })
        }
        DataType::Binary => {
            let array = array.as_any().downcast_ref::<BinaryArray>().unwrap();
            factorize_bytes(array.len(), sort, |index| {
                array.is_valid(index).then(|| array.value(index))
            })
        }
        DataType::LargeBinary => {
            let array = array.as_any().downcast_ref::<LargeBinaryArray>().unwrap();
            factorize_bytes(array.len(), sort, |index| {
                array.is_valid(index).then(|| array.value(index))
            })
        }
        DataType::BinaryView => {
            let array = array.as_any().downcast_ref::<BinaryViewArray>().unwrap();
            factorize_bytes(array.len(), sort, |index| {
                array.is_valid(index).then(|| array.value(index))
            })
        }
        DataType::Dictionary(key_type, value_type)
            if matches!(value_type.as_ref(), DataType::Utf8) =>
        {
            match key_type.as_ref() {
                DataType::Int8 => factorize_string_dictionary!(array, Int8Type, i32, sort),
                DataType::Int16 => factorize_string_dictionary!(array, Int16Type, i32, sort),
                DataType::Int32 => factorize_string_dictionary!(array, Int32Type, i32, sort),
                DataType::Int64 => factorize_string_dictionary!(array, Int64Type, i32, sort),
                DataType::UInt8 => factorize_string_dictionary!(array, UInt8Type, i32, sort),
                DataType::UInt16 => factorize_string_dictionary!(array, UInt16Type, i32, sort),
                DataType::UInt32 => factorize_string_dictionary!(array, UInt32Type, i32, sort),
                DataType::UInt64 => factorize_string_dictionary!(array, UInt64Type, i32, sort),
                key_type => {
                    return Err(format!(
                        "unsupported dictionary key dtype for factorization: {key_type}"
                    ));
                }
            }
        }
        DataType::Dictionary(key_type, value_type)
            if matches!(value_type.as_ref(), DataType::LargeUtf8) =>
        {
            match key_type.as_ref() {
                DataType::Int8 => factorize_string_dictionary!(array, Int8Type, i64, sort),
                DataType::Int16 => factorize_string_dictionary!(array, Int16Type, i64, sort),
                DataType::Int32 => factorize_string_dictionary!(array, Int32Type, i64, sort),
                DataType::Int64 => factorize_string_dictionary!(array, Int64Type, i64, sort),
                DataType::UInt8 => factorize_string_dictionary!(array, UInt8Type, i64, sort),
                DataType::UInt16 => factorize_string_dictionary!(array, UInt16Type, i64, sort),
                DataType::UInt32 => factorize_string_dictionary!(array, UInt32Type, i64, sort),
                DataType::UInt64 => factorize_string_dictionary!(array, UInt64Type, i64, sort),
                key_type => {
                    return Err(format!(
                        "unsupported dictionary key dtype for factorization: {key_type}"
                    ));
                }
            }
        }
        data_type => {
            return Err(format!(
                "unsupported Arrow dtype for factorization: {data_type}"
            ));
        }
    };
    Ok(result)
}

#[pyfunction]
#[pyo3(signature = (values, sort = false))]
pub fn factorize_arrow(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    sort: bool,
) -> PyResult<(PyArray, PyArray)> {
    let values = extract_arrow_array(values, "values")?;
    let (array, _field) = values.into_inner();
    let result = py
        .detach(|| factorize_array(array.as_ref(), sort))
        .map_err(PyValueError::new_err)?;
    Ok((
        crate::utils::u32_results_into_arrow(result.0),
        u64_results_into_arrow(result.1),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow_array::Int32Array;
    use std::sync::Arc;

    /// Deterministic xorshift-ish PRNG (mirrors `next_location.rs`'s test helper).
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.0 >> 33
        }
        fn range(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    #[test]
    fn total_f64_matches_ieee_total_order() {
        let neg_inf = TotalF64::new(f64::NEG_INFINITY);
        let neg_one = TotalF64::new(-1.0);
        let neg_zero = TotalF64::new(-0.0);
        let pos_zero = TotalF64::new(0.0);
        let pos_one = TotalF64::new(1.0);
        let pos_inf = TotalF64::new(f64::INFINITY);
        let nan = TotalF64::new(f64::NAN);

        assert!(neg_inf < neg_one);
        assert!(neg_one < neg_zero);
        assert_eq!(neg_zero, pos_zero);
        assert!(pos_zero < pos_one);
        assert!(pos_one < pos_inf);
        assert!(pos_inf < nan);

        assert_eq!(TotalF64::new(3.5).value(), 3.5);
        assert_eq!(TotalF64::new(-3.5).value(), -3.5);
        assert!(TotalF64::new(f64::NAN).value().is_nan());
    }

    #[test]
    fn should_parallelize_respects_row_threshold() {
        assert!(!should_parallelize(0));
        assert!(!should_parallelize(PARALLEL_ROW_THRESHOLD - 1));
        // Whether the >= threshold case actually parallelizes also depends
        // on `rayon::current_num_threads() > 1`, which is environment
        // dependent (e.g. a single-core CI runner) -- only assert the part
        // of the contract that's not environment dependent.
        if rayon::current_num_threads() > 1 {
            assert!(should_parallelize(PARALLEL_ROW_THRESHOLD));
        }
    }

    #[test]
    fn parallel_matches_sequential_exactly_across_random_inputs() {
        let mut rng = Lcg(0xFACD_u64);
        for trial in 0..300 {
            let len = 1 + (rng.range(500) as usize);
            let vocab = 1 + (rng.range(20) as usize); // low-to-moderate cardinality
            let null_rate_pct = rng.range(4); // 0 => no nulls, else ~1-in-N nulls
            let values: Vec<Option<u64>> = (0..len)
                .map(|_| {
                    if null_rate_pct > 0 && rng.range(null_rate_pct * 5 + 1) == 0 {
                        None
                    } else {
                        Some(1_000_000 + rng.range(vocab as u64))
                    }
                })
                .collect();
            let value_at = |index: usize| values[index];

            let (seq_codes, seq_reps) =
                factorize_values::<_, NumericHasher, _>(len, false, value_at);
            let (par_codes, par_reps) =
                factorize_values_parallel::<_, NumericHasher, _>(len, false, value_at);

            assert_eq!(
                seq_codes, par_codes,
                "trial {trial}: codes mismatch (values={values:?})"
            );
            assert_eq!(
                seq_reps, par_reps,
                "trial {trial}: representatives mismatch (values={values:?})"
            );
        }
    }

    #[test]
    fn parallel_matches_sequential_with_sort_enabled() {
        let mut rng = Lcg(0xB0BA_u64);
        for trial in 0..100 {
            let len = 1 + (rng.range(300) as usize);
            let vocab = 1 + (rng.range(15) as usize);
            let values: Vec<Option<u64>> =
                (0..len).map(|_| Some(rng.range(vocab as u64))).collect();
            let value_at = |index: usize| values[index];

            let (seq_codes, seq_reps) =
                factorize_values::<_, NumericHasher, _>(len, true, value_at);
            let (par_codes, par_reps) =
                factorize_values_parallel::<_, NumericHasher, _>(len, true, value_at);

            assert_eq!(seq_codes, par_codes, "trial {trial}: sorted codes mismatch");
            assert_eq!(
                seq_reps, par_reps,
                "trial {trial}: sorted representatives mismatch"
            );
        }
    }

    #[test]
    fn parallel_handles_empty_and_single_element_input() {
        let empty: Vec<Option<u64>> = Vec::new();
        let (codes, reps) =
            factorize_values_parallel::<_, NumericHasher, _>(0, false, |i| empty[i]);
        assert!(codes.is_empty());
        assert!(reps.is_empty());

        let single = [Some(42u64)];
        let (codes, reps) =
            factorize_values_parallel::<_, NumericHasher, _>(1, false, |i| single[i]);
        assert_eq!(codes, vec![0]);
        assert_eq!(reps, vec![0]);
    }

    #[test]
    fn factorize_values_null_free_matches_nullable_equivalent() {
        let values: Vec<u64> = vec![7, 3, 7, 1, 3, 9];
        let nullable: Vec<Option<u64>> = values.iter().map(|&v| Some(v)).collect();
        for sort in [false, true] {
            let (no_null_codes, no_null_reps) =
                factorize_values::<_, NumericHasher, _>(values.len(), sort, |i| Some(values[i]));
            let (nullable_codes, nullable_reps) =
                factorize_values::<_, NumericHasher, _>(nullable.len(), sort, |i| nullable[i]);
            assert_eq!(no_null_codes, nullable_codes);
            assert_eq!(no_null_reps, nullable_reps);
        }
    }

    #[test]
    fn factorize_bytes_preserves_first_seen_order_and_groups_nulls() {
        let values: Vec<Option<&[u8]>> = vec![Some(b"b"), None, Some(b"a"), Some(b"b"), None];
        let (codes, reps) = factorize_bytes(values.len(), false, |i| values[i]);
        assert_eq!(codes, vec![0, 1, 2, 0, 1]);
        assert_eq!(reps, vec![0, 1, 2]);
    }

    #[test]
    fn factorize_bytes_sorted_puts_null_last() {
        let values: Vec<Option<&[u8]>> = vec![Some(b"b"), None, Some(b"a")];
        let (codes, _) = factorize_bytes(values.len(), true, |i| values[i]);
        // sorted order: "a" (0), "b" (1), null (2)
        assert_eq!(codes, vec![1, 2, 0]);
    }

    #[test]
    fn factorize_boolean_null_free_matches_nullable_equivalent() {
        let values = [true, false, true, true, false];
        let nullable: Vec<Option<bool>> = values.iter().map(|&v| Some(v)).collect();
        let no_null_array = BooleanArray::from(values.to_vec());
        let nullable_array = BooleanArray::from(nullable);
        for sort in [false, true] {
            let (no_null_codes, no_null_reps) = factorize_boolean(&no_null_array, sort);
            let (nullable_codes, nullable_reps) = factorize_boolean(&nullable_array, sort);
            assert_eq!(no_null_codes, nullable_codes);
            assert_eq!(no_null_reps, nullable_reps);
        }
    }

    #[test]
    fn factorize_dense_integers_null_free_matches_nullable_equivalent() {
        let values: Vec<i32> = vec![5, 2, 5, 9, 2];
        let no_null_array = PrimitiveArray::<Int32Type>::from(values.clone());
        let nullable_array =
            PrimitiveArray::<Int32Type>::from(values.into_iter().map(Some).collect::<Vec<_>>());
        for sort in [false, true] {
            let no_null = factorize_dense_integers(&no_null_array, sort, false).unwrap();
            let nullable = factorize_dense_integers(&nullable_array, sort, false).unwrap();
            assert_eq!(no_null, nullable);
        }
    }

    #[test]
    fn factorize_dense_integers_always_dense_skips_span_check() {
        // Span (256) would exceed a tiny array's `dense_span_cap`, but
        // `always_dense` (used for Int8/UInt8/Int16/UInt16) bypasses that.
        let values: Vec<i8> = vec![i8::MIN, 0, i8::MAX, 0];
        let array = PrimitiveArray::<Int8Type>::from(values);
        assert!(factorize_dense_integers(&array, false, false).is_none());
        let (codes, reps) = factorize_dense_integers(&array, false, true).unwrap();
        assert_eq!(codes, vec![0, 1, 2, 1]);
        assert_eq!(reps, vec![0, 1, 2]);
    }

    #[test]
    fn factorize_string_dictionary_utf8_and_large_utf8_agree() {
        let keys = Int32Array::from(vec![Some(1i32), Some(0), None, Some(1)]);
        let utf8_values = StringArray::from(vec![Some("a"), Some("b")]);
        let utf8_dict =
            DictionaryArray::<Int32Type>::try_new(keys.clone(), Arc::new(utf8_values)).unwrap();
        let large_utf8_values = LargeStringArray::from(vec![Some("a"), Some("b")]);
        let large_utf8_dict =
            DictionaryArray::<Int32Type>::try_new(keys, Arc::new(large_utf8_values)).unwrap();

        for sort in [false, true] {
            let utf8_result = factorize_string_dictionary::<Int32Type, i32>(&utf8_dict, sort);
            let large_utf8_result =
                factorize_string_dictionary::<Int32Type, i64>(&large_utf8_dict, sort);
            assert_eq!(utf8_result, large_utf8_result);
        }
    }
}
