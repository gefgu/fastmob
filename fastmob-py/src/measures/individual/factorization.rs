use std::cmp::Ordering;
use std::collections::hash_map::Entry;
use std::hash::{Hash, Hasher};

use arrow_array::types::*;
use arrow_array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, DictionaryArray, GenericStringArray,
    LargeBinaryArray, LargeStringArray, OffsetSizeTrait, PrimitiveArray, StringArray,
    StringViewArray,
};
use arrow_buffer::ArrowNativeType;
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
/// singleton/doubleton counts) scaled by the sampling fraction. This is only
/// ever used as a capacity *hint*: under- or over-estimating never produces
/// incorrect `codes`/`representatives`, only extra rehashing or wasted
/// memory, so a heuristic estimator is acceptable here.
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
    let scale = len as f64 / CARDINALITY_SAMPLE as f64;
    let estimate = (chao1 * scale).round().max(0.0) as usize;
    estimate.clamp(CARDINALITY_SAMPLE, MAX_INITIAL_CAPACITY)
}

fn factorize_values<K, F>(len: usize, sort: bool, value_at: F) -> (Vec<u32>, Vec<u64>)
where
    K: Eq + Hash + Ord + Send,
    F: Fn(usize) -> Option<K>,
{
    let mut seen: FxHashMap<K, u32> = FxHashMap::with_capacity_and_hasher(
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
fn factorize_values_parallel<K, F>(len: usize, sort: bool, value_at: F) -> (Vec<u32>, Vec<u64>)
where
    K: Eq + Hash + Ord + Send + Sync,
    F: Fn(usize) -> Option<K> + Sync,
{
    if len == 0 {
        return (Vec::new(), Vec::new());
    }

    let num_chunks = rayon::current_num_threads().max(1);
    let chunk_len = len.div_ceil(num_chunks).max(1);

    let mut codes = vec![0u32; len];

    // Stage 1.
    let chunk_locals: Vec<(Option<u32>, Vec<u64>)> = codes
        .par_chunks_mut(chunk_len)
        .enumerate()
        .map(|(chunk_index, slots)| {
            let base = chunk_index * chunk_len;
            let mut local_seen: FxHashMap<K, u32> = FxHashMap::default();
            let mut local_null: Option<u32> = None;
            let mut local_representatives: Vec<u64> = Vec::new();
            for (offset, slot) in slots.iter_mut().enumerate() {
                let index = base + offset;
                *slot = match value_at(index) {
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
                };
            }
            (local_null, local_representatives)
        })
        .collect();

    // Stage 2.
    let mut global_map: FxHashMap<K, u64> = FxHashMap::default();
    let mut global_null: Option<u64> = None;
    for (local_null, local_representatives) in &chunk_locals {
        for &representative in local_representatives {
            if let Some(value) = value_at(representative as usize) {
                global_map
                    .entry(value)
                    .and_modify(|r| *r = (*r).min(representative))
                    .or_insert(representative);
            }
        }
        if let Some(local_null_code) = local_null {
            let representative = local_representatives[*local_null_code as usize];
            global_null = Some(global_null.map_or(representative, |r| r.min(representative)));
        }
    }

    // Stage 3.
    let mut ordered: Vec<(u64, Option<K>)> = global_map
        .into_iter()
        .map(|(value, representative)| (representative, Some(value)))
        .collect();
    if let Some(representative) = global_null {
        ordered.push((representative, None));
    }
    ordered.sort_unstable_by_key(|&(representative, _)| representative);

    let mut representatives: Vec<u64> = Vec::with_capacity(ordered.len());
    let mut value_to_code: FxHashMap<K, u32> =
        FxHashMap::with_capacity_and_hasher(ordered.len(), Default::default());
    let mut null_code: Option<u32> = None;
    for (code_index, (representative, value)) in ordered.into_iter().enumerate() {
        let code = next_code(code_index);
        representatives.push(representative);
        match value {
            Some(value) => {
                value_to_code.insert(value, code);
            }
            None => null_code = Some(code),
        }
    }

    // Stage 4.
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
                        value_to_code[&value]
                    }
                })
                .collect();
            for slot in slots.iter_mut() {
                *slot = translation[*slot as usize];
            }
        });

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
fn factorize_bytes<'a, F>(len: usize, sort: bool, value_at: F) -> (Vec<u32>, Vec<u64>)
where
    F: Fn(usize) -> Option<&'a [u8]>,
{
    // `ahash`, not `FxHashMap`, on purpose: measured (not assumed) on a 4M-row
    // synthetic string workload at 3 cardinalities (50 / 5,000 / 500,000
    // distinct values), `ahash` beat `FxHashMap` by ~5-12% here, while
    // `foldhash` was consistently ~5-15% *slower* than `FxHashMap`. Numeric
    // key paths (`factorize_values`, `factorize_values_parallel`) keep
    // `FxHashMap`, which remains the better choice for short, fixed-size
    // integer/date/float keys.
    #[allow(clippy::disallowed_types)]
    let mut seen: std::collections::HashMap<&[u8], u32, ahash::RandomState> =
        std::collections::HashMap::with_capacity_and_hasher(
            estimate_cardinality(len, &value_at),
            ahash::RandomState::default(),
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

fn factorize_strings<'a, F>(len: usize, sort: bool, value_at: F) -> (Vec<u32>, Vec<u64>)
where
    F: Fn(usize) -> Option<&'a str>,
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
            factorize_values_parallel(array.len(), sort, value_at)
        } else {
            factorize_values(array.len(), sort, value_at)
        }
    } else {
        let value_at = move |index: usize| array.is_valid(index).then(|| array.value(index));
        if should_parallelize(array.len()) {
            factorize_values_parallel(array.len(), sort, value_at)
        } else {
            factorize_values(array.len(), sort, value_at)
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
            factorize_values_parallel(array.len(), sort, value_at)
        } else {
            factorize_values(array.len(), sort, value_at)
        }
    } else {
        let value_at = move |index: usize| {
            array
                .is_valid(index)
                .then(|| TotalF64::new(array.value(index).into()))
        };
        if should_parallelize(array.len()) {
            factorize_values_parallel(array.len(), sort, value_at)
        } else {
            factorize_values(array.len(), sort, value_at)
        }
    }
}

/// Direct-indexed fast path for integer columns whose value span
/// (`max - min + 1`) is small enough to use a `Vec` instead of a hash map.
///
/// `always_dense` bypasses the span-vs-`dense_span_cap` check entirely --
/// used for `Int8`/`UInt8`/`Int16`/`UInt16`, whose span is statically
/// bounded (≤256 / ≤65536) regardless of array length, so the dense path is
/// always affordable there.
fn factorize_dense_integers<T>(
    array: &PrimitiveArray<T>,
    sort: bool,
    always_dense: bool,
) -> Option<(Vec<u32>, Vec<u64>)>
where
    T: ArrowPrimitiveType,
    T::Native: Into<i128>,
{
    if array.is_empty() {
        return Some((Vec::new(), Vec::new()));
    }
    let range = if array.null_count() == 0 {
        let mut values = array.values().iter().map(|&value| value.into());
        let first = values.next().expect("non-empty array has a first value");
        Some(values.fold((first, first), |(min, max), value| {
            (min.min(value), max.max(value))
        }))
    } else {
        let mut values = (0..array.len())
            .filter(|&index| array.is_valid(index))
            .map(|index| array.value(index).into());
        values.next().map(|first| {
            values.fold((first, first), |(min, max), value| {
                (min.min(value), max.max(value))
            })
        })
    };
    let Some((min, max)) = range else {
        return Some((vec![0; array.len()], vec![0]));
    };
    let span = usize::try_from(max - min + 1).ok()?;
    if !always_dense && span > dense_span_cap(array.len()) {
        return None;
    }

    // Representative row index per distinct offset -- a single shared prep
    // step used by both the sort and no-sort branches below (previously two
    // separately hand-written, near-duplicated loops).
    let mut representative_by_value = vec![u32::MAX; span];
    let mut null_representative: Option<u32> = None;
    for index in 0..array.len() {
        if array.is_valid(index) {
            let offset = (array.value(index).into() - min) as usize;
            if representative_by_value[offset] == u32::MAX {
                representative_by_value[offset] = index as u32;
            }
        } else {
            null_representative.get_or_insert(index as u32);
        }
    }

    let mut codes_by_value = vec![u32::MAX; span];
    let mut representatives: Vec<u64> = Vec::new();
    let null_code;
    if sort {
        // Offset order already equals value order, so assigning codes in
        // ascending offset order directly gives the value-sorted result.
        for (offset, &representative) in representative_by_value.iter().enumerate() {
            if representative != u32::MAX {
                codes_by_value[offset] = next_code(representatives.len());
                representatives.push(representative as u64);
            }
        }
        null_code = null_representative.map(|representative| {
            let code = next_code(representatives.len());
            representatives.push(representative as u64);
            code
        });
    } else {
        // First-seen order: sort candidate offsets (plus the null slot, if
        // any) by representative row index -- the same trick the generic
        // parallel path uses to reproduce sequential-scan order.
        let mut candidates: Vec<(u32, Option<usize>)> = representative_by_value
            .iter()
            .enumerate()
            .filter(|&(_, &representative)| representative != u32::MAX)
            .map(|(offset, &representative)| (representative, Some(offset)))
            .collect();
        if let Some(representative) = null_representative {
            candidates.push((representative, None));
        }
        candidates.sort_unstable_by_key(|&(representative, _)| representative);

        let mut assigned_null_code = None;
        for (representative, offset) in candidates {
            let code = next_code(representatives.len());
            representatives.push(representative as u64);
            match offset {
                Some(offset) => codes_by_value[offset] = code,
                None => assigned_null_code = Some(code),
            }
        }
        null_code = assigned_null_code;
    }

    let codes = (0..array.len())
        .map(|index| {
            if array.is_valid(index) {
                codes_by_value[(array.value(index).into() - min) as usize]
            } else {
                null_code.expect("a null row has a null code")
            }
        })
        .collect();
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

    // Recompute a row's dictionary-relative logical code from its key on
    // demand rather than materializing an `array.len()`-sized intermediate
    // buffer: `dictionary_codes` is small (bounded by dictionary
    // cardinality) and index-only, so recomputation is cheaper than the
    // extra allocation and memory round-trip a stored buffer would cost.
    let logical_code_at = |index: usize| -> usize {
        if array.is_valid(index) {
            dictionary_codes[array.keys().value(index).as_usize()] as usize
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
    let codes = (0..array.len())
        .map(|index| output_code_by_value[logical_code_at(index)])
        .collect();
    (codes, representatives)
}

macro_rules! factorize_primitive {
    ($array:expr, $type:ty, $sort:expr) => {{
        let array = $array
            .as_any()
            .downcast_ref::<PrimitiveArray<$type>>()
            .expect("Arrow data type and primitive array must agree");
        factorize_primitive_values(array, $sort)
    }};
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
        DataType::Date32 => factorize_primitive!(array, Date32Type, sort),
        DataType::Date64 => factorize_primitive!(array, Date64Type, sort),
        DataType::Timestamp(unit, _) => match unit {
            arrow_schema::TimeUnit::Second => {
                factorize_primitive!(array, TimestampSecondType, sort)
            }
            arrow_schema::TimeUnit::Millisecond => {
                factorize_primitive!(array, TimestampMillisecondType, sort)
            }
            arrow_schema::TimeUnit::Microsecond => {
                factorize_primitive!(array, TimestampMicrosecondType, sort)
            }
            arrow_schema::TimeUnit::Nanosecond => {
                factorize_primitive!(array, TimestampNanosecondType, sort)
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

            let (seq_codes, seq_reps) = factorize_values(len, false, value_at);
            let (par_codes, par_reps) = factorize_values_parallel(len, false, value_at);

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

            let (seq_codes, seq_reps) = factorize_values(len, true, value_at);
            let (par_codes, par_reps) = factorize_values_parallel(len, true, value_at);

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
        let (codes, reps) = factorize_values_parallel(0, false, |i| empty[i]);
        assert!(codes.is_empty());
        assert!(reps.is_empty());

        let single = [Some(42u64)];
        let (codes, reps) = factorize_values_parallel(1, false, |i| single[i]);
        assert_eq!(codes, vec![0]);
        assert_eq!(reps, vec![0]);
    }

    #[test]
    fn factorize_values_null_free_matches_nullable_equivalent() {
        let values: Vec<u64> = vec![7, 3, 7, 1, 3, 9];
        let nullable: Vec<Option<u64>> = values.iter().map(|&v| Some(v)).collect();
        for sort in [false, true] {
            let (no_null_codes, no_null_reps) =
                factorize_values(values.len(), sort, |i| Some(values[i]));
            let (nullable_codes, nullable_reps) =
                factorize_values(nullable.len(), sort, |i| nullable[i]);
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
