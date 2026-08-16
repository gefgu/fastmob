use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;

pub type IndexRanges = Vec<(u64, u64)>;
pub type OrderedIndexRanges = (Vec<u64>, IndexRanges);

/// Value the PyO3 binding substitutes for a null timestamp before handing the
/// buffer to these kernels. It is the smallest `i64`, so it sorts before every
/// real timestamp -- but it is excluded from the range/divisor analysis below,
/// since a single null would otherwise force the packed-key span to the full
/// `i64` width.
const NULL_TIMESTAMP: i64 = i64::MIN;

/// Group length at or above which a single group's sort is itself parallelized.
///
/// Without this, a column where one group holds most of the rows (a single
/// heavy user, a default/unknown uid bucket) degenerates to one thread doing
/// nearly all the work while the rest idle -- the parallel iterator below
/// splits by *group count*, which says nothing about where the rows are.
const PARALLEL_GROUP_LEN: usize = 1 << 16;

/// Group length at or below which insertion sort beats pdqsort's setup cost.
/// Grouped trajectory data is overwhelmingly made of short per-user runs, so
/// this is the common case, not the corner case.
const INSERTION_SORT_LEN: usize = 24;

/// Group count above which ordering the per-group slices longest-first is
/// skipped. The ordering helps rayon schedule a few uneven groups, but it is
/// itself a sort over the group list -- at 500k groups it cost more than the
/// scheduling it bought.
const MAX_GROUPS_TO_ORDER_BY_LENGTH: usize = 1 << 13;

/// Rough cap on the per-chunk histogram tables used by the parallel scatter.
/// They are `chunks * num_groups * 4` bytes, so a column with very many groups
/// gets fewer chunks rather than an unbounded allocation.
const SCATTER_HISTOGRAM_BUDGET: usize = 64 << 20;

/// Minimum rows per scatter chunk; below this the histogram bookkeeping costs
/// more than the parallelism returns.
const MIN_SCATTER_CHUNK: usize = 1 << 14;

/// Exact divisors worth testing, coarsest first -- the unit conversions between
/// Arrow's second/milli/micro/nanosecond timestamp types. See [`TimeKeyPlan`].
const CANDIDATE_DIVISORS: [i64; 4] = [1_000_000_000, 1_000_000, 1_000, 1];

pub fn split_ordered_index_ranges((indices, ranges): OrderedIndexRanges) -> (Vec<u64>, Vec<u64>) {
    (indices, ranges.into_iter().map(|(_, end)| end).collect())
}

/// Start/end offsets of each maximal run of equal values.
///
/// Generic over the element type so a caller can find the run structure of a
/// uid column directly, without dense-coding it first: `pyarrow`'s
/// `run_end_encode` answers the same question but measured ~15ms per 4M rows
/// against well under 1ms here.
pub fn run_boundaries<T: PartialEq + Sync>(values: &[T]) -> (Vec<u64>, Vec<u64>) {
    let n = values.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let boundaries: Vec<u64> = values
        .par_windows(2)
        .enumerate()
        .filter_map(|(idx, window)| (window[0] != window[1]).then_some((idx + 1) as u64))
        .collect();

    let mut starts = Vec::with_capacity(boundaries.len() + 1);
    starts.push(0);
    starts.extend_from_slice(&boundaries);

    let mut ends = boundaries;
    ends.push(n as u64);
    (starts, ends)
}

pub fn presorted_ranges_for_u32_codes(codes: &[u32]) -> (Vec<u64>, Vec<u64>) {
    run_boundaries(codes)
}

use rustc_hash::FxHashSet;

/// Row count below which validation stays on one thread: the rayon dispatch
/// costs more than it saves until roughly this size.
const VALIDATE_SEQUENTIAL_BELOW: usize = 1 << 15;

/// Smallest validation chunk. Below this the per-chunk summary bookkeeping
/// starts to show against the few rows it covers.
const VALIDATE_MIN_CHUNK: usize = 1 << 13;

/// Chunks handed to rayon per thread. Oversubscribing this way keeps the
/// schedule's tail short without making the chunks so small that the reduce
/// tree dominates; between 8 and 64 the difference measured as noise, so this
/// is the middle of a flat region rather than a sharp optimum.
const VALIDATE_CHUNKS_PER_THREAD: usize = 16;

/// Validates that:
///   1. every distinct code occupies exactly one contiguous run, and
///   2. if `check_timestamps`, timestamps are non-decreasing *within* each run.
///
/// Timestamps across run boundaries are unconstrained.
///
/// Both properties are checked without ever walking the runs sequentially,
/// which is what makes this parallel:
///
/// * (2) is a purely local predicate over adjacent rows -- `codes[i] !=
///   codes[i - 1] || timestamps[i] >= timestamps[i - 1]` -- so it needs no run
///   structure at all and vectorizes to an OR-reduction of compares.
/// * (1) holds exactly when the sequence of *run leaders* (the code at each run
///   start) has no duplicate. Run starts are themselves a local predicate, so
///   every chunk can find its own leaders independently.
///
/// The leaders are then summarized rather than materialized: a strictly
/// monotone leader sequence is duplicate-free by definition, and grouped
/// trajectory data almost always is (factorizing uids in first-appearance order
/// makes the leaders `0, 1, 2, …` by construction; a uid-sorted frame makes
/// them ascending; a descending frame, descending). That collapses the common
/// case to one allocation-free pass over the data. Only genuinely arbitrary
/// group order falls through to [`leaders_unique`].
pub fn validate_grouped_u32_codes(
    codes: &[u32],
    timestamps: Option<&[i64]>,
    check_timestamps: bool,
) -> bool {
    // Hoist the Option/length check out of the hot path entirely.
    let timestamps: Option<&[i64]> = if check_timestamps {
        match timestamps {
            Some(t) if t.len() == codes.len() => Some(t),
            _ => return false,
        }
    } else {
        None
    };

    let n = codes.len();
    if n < 2 {
        return true;
    }
    if n < VALIDATE_SEQUENTIAL_BELOW {
        return validate_sequential(codes, timestamps);
    }

    let threads = rayon::current_num_threads().max(1);
    let chunk = (n / (threads * VALIDATE_CHUNKS_PER_THREAD).max(1)).max(VALIDATE_MIN_CHUNK);

    // `try_reduce` short-circuits the whole iterator on the first chunk that
    // sees a backwards timestamp, so invalid input still costs a fraction of a
    // pass rather than a full one.
    let summary = (0..n.div_ceil(chunk))
        .into_par_iter()
        .map(|index| {
            let (from, to) = validate_chunk_bounds(n, chunk, index);
            if let Some(timestamps) = timestamps
                && !chunk_timestamps_ok(codes, timestamps, from, to)
            {
                return None;
            }
            Some(chunk_leader_summary(codes, from, to))
        })
        .try_reduce(|| LeaderSummary::EMPTY, |a, b| Some(a.merge(b)));

    match summary {
        None => false,
        Some(summary) => summary.monotone() || leaders_unique(codes, chunk, &summary),
    }
}

/// Half-open row range covered by chunk `index`. Row 0 has no predecessor, so
/// every chunk's scan starts at 1 at the earliest.
#[inline]
fn validate_chunk_bounds(n: usize, chunk: usize, index: usize) -> (usize, usize) {
    ((index * chunk).max(1), ((index + 1) * chunk).min(n))
}

/// The single-group form of [`validate_grouped_u32_codes`]: with no uid column
/// every row is one individual, so contiguity is vacuous and only the timestamp
/// order is left to check.
pub fn validate_non_decreasing_timestamps(timestamps: &[i64]) -> bool {
    let n = timestamps.len();
    if n < 2 {
        return true;
    }
    if n < VALIDATE_SEQUENTIAL_BELOW {
        return non_decreasing(timestamps, 1, n);
    }
    let threads = rayon::current_num_threads().max(1);
    let chunk = (n / (threads * VALIDATE_CHUNKS_PER_THREAD).max(1)).max(VALIDATE_MIN_CHUNK);
    (0..n.div_ceil(chunk)).into_par_iter().all(|index| {
        let (from, to) = validate_chunk_bounds(n, chunk, index);
        non_decreasing(timestamps, from, to)
    })
}

/// Are timestamps non-decreasing inside every run delimited by `ends`?
///
/// The same property [`validate_grouped_u32_codes`] checks, but for a caller
/// that already holds the run boundaries and would rather not pay to dense-code
/// the uid column just to re-derive them. Boundaries between runs are
/// unconstrained, as ever.
pub fn validate_non_decreasing_within_ends(timestamps: &[i64], ends: &[u64]) -> bool {
    let n = timestamps.len();
    if ends.last().is_some_and(|&last| last as usize > n) {
        return false;
    }
    ends.par_iter().enumerate().all(|(position, &end)| {
        let start = if position == 0 {
            0
        } else {
            ends[position - 1] as usize
        };
        let end = end as usize;
        start >= end || non_decreasing(timestamps, start + 1, end)
    })
}

/// Branchless, so it vectorizes to an OR-reduction of compares.
#[inline]
fn non_decreasing(timestamps: &[i64], from: usize, to: usize) -> bool {
    if from >= to {
        return true;
    }
    let (previous, next) = (&timestamps[from - 1..to - 1], &timestamps[from..to]);
    let mut bad = 0u8;
    for (&a, &b) in previous.iter().zip(next) {
        bad |= (b < a) as u8;
    }
    bad == 0
}

/// Are all adjacent pairs in `from..to` that stay inside one run in
/// non-decreasing time order?
///
/// Deliberately branchless and kept separate from the leader scan below: fusing
/// the two puts the leader push -- a real branch -- inside the loop and stops
/// the comparison from vectorizing, which measured slower even though it halves
/// the number of passes over data that is already in L2 by then.
#[inline]
fn chunk_timestamps_ok(codes: &[u32], timestamps: &[i64], from: usize, to: usize) -> bool {
    if from >= to {
        return true;
    }
    let (code_prev, code_next) = (&codes[from - 1..to - 1], &codes[from..to]);
    let (time_prev, time_next) = (&timestamps[from - 1..to - 1], &timestamps[from..to]);
    let mut bad = 0u8;
    for (((&a, &b), &x), &y) in code_prev
        .iter()
        .zip(code_next)
        .zip(time_prev)
        .zip(time_next)
    {
        bad |= ((a == b) & (y < x)) as u8;
    }
    bad == 0
}

/// What one chunk needs to report about its run leaders: enough to decide
/// monotonicity globally, plus the size and range needed to pick a duplicate
/// test if it turns out not to be monotone.
#[derive(Clone, Copy)]
struct LeaderSummary {
    first: u32,
    last: u32,
    max: u32,
    count: usize,
    any: bool,
    increasing: bool,
    decreasing: bool,
}

impl LeaderSummary {
    const EMPTY: Self = Self {
        first: 0,
        last: 0,
        max: 0,
        count: 0,
        any: false,
        increasing: true,
        decreasing: true,
    };

    #[inline]
    fn push(&mut self, code: u32) {
        if self.any {
            self.increasing &= self.last < code;
            self.decreasing &= self.last > code;
            self.max = self.max.max(code);
        } else {
            self.first = code;
            self.max = code;
            self.any = true;
        }
        self.last = code;
        self.count += 1;
    }

    /// Concatenates `other`'s leader sequence after `self`'s, so the join is
    /// checked as well as each side.
    #[inline]
    fn merge(mut self, other: Self) -> Self {
        if !other.any {
            return self;
        }
        if !self.any {
            return other;
        }
        self.increasing &= other.increasing && self.last < other.first;
        self.decreasing &= other.decreasing && self.last > other.first;
        self.last = other.last;
        self.max = self.max.max(other.max);
        self.count += other.count;
        self
    }

    /// Strictly monotone in either direction implies all leaders are distinct.
    #[inline]
    fn monotone(&self) -> bool {
        self.increasing || self.decreasing
    }
}

#[inline]
fn chunk_leader_summary(codes: &[u32], from: usize, to: usize) -> LeaderSummary {
    let mut summary = LeaderSummary::EMPTY;
    // Row 0 starts the first run, and only the chunk that owns it may say so.
    if from == 1 {
        summary.push(codes[0]);
    }
    for index in from..to {
        let code = codes[index];
        if code != codes[index - 1] {
            summary.push(code);
        }
    }
    summary
}

/// Is a dense bitset over `0..=max` a reasonable way to test `count` leaders
/// for duplicates? It costs one bit per *possible* code, so it is only sane
/// while the code space stays within a small multiple of the leaders in it.
#[inline]
fn bitset_is_worthwhile(max: u32, count: usize) -> bool {
    (max as usize) / 8 <= count * 16 + (1 << 16)
}

/// Duplicate test for a leader sequence that is not monotone.
///
/// When the code space is dense the leaders never need to be materialized:
/// a second parallel pass sets one bit per leader in a shared bitset and fails
/// on the first bit that was already set. Every leader is visited exactly once
/// across all chunks, so `Relaxed` is enough -- the atomics are here to make
/// concurrent writes to the same word well defined, not to order anything.
fn leaders_unique(codes: &[u32], chunk: usize, summary: &LeaderSummary) -> bool {
    if !bitset_is_worthwhile(summary.max, summary.count) {
        return leaders_unique_collected(codes, chunk, summary.count);
    }

    let n = codes.len();
    let bits: Vec<AtomicU64> = (0..(summary.max as usize) / 64 + 1)
        .map(|_| AtomicU64::new(0))
        .collect();
    let claim = |code: u32| -> bool {
        let (word, bit) = ((code as usize) >> 6, 1u64 << (code & 63));
        bits[word].fetch_or(bit, Ordering::Relaxed) & bit == 0
    };

    if !claim(codes[0]) {
        return false;
    }
    (0..n.div_ceil(chunk)).into_par_iter().all(|index| {
        let (from, to) = validate_chunk_bounds(n, chunk, index);
        for index in from..to {
            if codes[index] != codes[index - 1] && !claim(codes[index]) {
                return false;
            }
        }
        true
    })
}

/// Fallback for a code space too sparse to bitset: gather the leaders, then
/// test them directly.
fn leaders_unique_collected(codes: &[u32], chunk: usize, count: usize) -> bool {
    let n = codes.len();
    let parts: Vec<Vec<u32>> = (0..n.div_ceil(chunk))
        .into_par_iter()
        .map(|index| {
            let (from, to) = validate_chunk_bounds(n, chunk, index);
            let mut leaders = Vec::new();
            for index in from..to {
                if codes[index] != codes[index - 1] {
                    leaders.push(codes[index]);
                }
            }
            leaders
        })
        .collect();

    let mut leaders = Vec::with_capacity(count);
    leaders.push(codes[0]);
    for part in &parts {
        leaders.extend_from_slice(part);
    }
    if leaders.len() > 1 << 16 {
        leaders.par_sort_unstable();
        return leaders.par_windows(2).all(|pair| pair[0] != pair[1]);
    }
    let mut seen = FxHashSet::with_capacity_and_hasher(leaders.len() * 2, Default::default());
    leaders.iter().all(|&code| seen.insert(code))
}

/// Single-threaded path for inputs small enough that the parallel machinery is
/// pure overhead. Fused, unlike the parallel path, because at this size the
/// data is cache-resident and the extra pass costs more than the branch does.
fn validate_sequential(codes: &[u32], timestamps: Option<&[i64]>) -> bool {
    let n = codes.len();
    let mut summary = LeaderSummary::EMPTY;
    summary.push(codes[0]);

    match timestamps {
        None => {
            for index in 1..n {
                let code = codes[index];
                if code != codes[index - 1] {
                    summary.push(code);
                }
            }
        }
        Some(timestamps) => {
            let mut bad = 0u8;
            for index in 1..n {
                let code = codes[index];
                let same_run = code == codes[index - 1];
                bad |= (same_run & (timestamps[index] < timestamps[index - 1])) as u8;
                if !same_run {
                    summary.push(code);
                }
            }
            if bad != 0 {
                return false;
            }
        }
    }

    if summary.monotone() {
        return true;
    }

    let mut leaders = Vec::with_capacity(summary.count);
    leaders.push(codes[0]);
    for index in 1..n {
        if codes[index] != codes[index - 1] {
            leaders.push(codes[index]);
        }
    }
    if bitset_is_worthwhile(summary.max, summary.count) {
        let mut bits = vec![0u64; (summary.max as usize) / 64 + 1];
        return leaders.iter().all(|&code| {
            let (word, bit) = ((code as usize) >> 6, 1u64 << (code & 63));
            let fresh = bits[word] & bit == 0;
            bits[word] |= bit;
            fresh
        });
    }
    let mut seen = FxHashSet::with_capacity_and_hasher(leaders.len() * 2, Default::default());
    leaders.iter().all(|&code| seen.insert(code))
}

// ---------------------------------------------------------------------------
// Timestamp key planning
// ---------------------------------------------------------------------------

/// Smallest/largest non-null timestamp, and whether any null is present.
#[derive(Clone, Copy, Debug)]
struct TimeRange {
    min: i64,
    max: i64,
    has_null: bool,
}

impl TimeRange {
    fn empty() -> Self {
        Self {
            min: i64::MAX,
            max: i64::MIN,
            has_null: false,
        }
    }

    fn merge(self, other: Self) -> Self {
        Self {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
            has_null: self.has_null || other.has_null,
        }
    }

    fn is_empty(&self) -> bool {
        self.min > self.max
    }
}

/// Branchless min/max so the scan stays vectorizable.
///
/// Deliberately *not* fused with the divisibility test below: integer division
/// by a constant does not vectorize, so folding it in here would slow the
/// common case (a span needing no scaling at all) to serve the uncommon one.
fn scan_time_range(timestamps: &[i64]) -> TimeRange {
    timestamps
        .par_chunks(1 << 14)
        .map(|chunk| {
            let mut range = TimeRange::empty();
            for &value in chunk {
                let is_null = value == NULL_TIMESTAMP;
                range.has_null |= is_null;
                // A null is `i64::MIN`, so it can never win `max`; for `min` it
                // is replaced by `i64::MAX` to keep it out of the range.
                let for_min = if is_null { i64::MAX } else { value };
                range.min = range.min.min(for_min);
                range.max = range.max.max(value);
            }
            range
        })
        .reduce(TimeRange::empty, TimeRange::merge)
}

/// Is every non-null timestamp an exact multiple of `divisor`?
fn all_divisible_by(timestamps: &[i64], divisor: i64) -> bool {
    timestamps.par_chunks(1 << 14).all(|chunk| {
        chunk
            .iter()
            .all(|&value| value == NULL_TIMESTAMP || value % divisor == 0)
    })
}

/// How to turn a timestamp plus its row index into a single sortable `u64`.
///
/// The point is to remove the indirection from the comparator. Sorting row
/// indices with a `timestamps[index]` lookup makes every one of the ~`n log n`
/// comparisons a random read into a buffer far larger than cache; packing key
/// and payload into one word makes the sort read purely sequential memory, and
/// the timestamp buffer is touched exactly once, sequentially, while the keys
/// are built.
///
/// Layout is `(scaled_offset << idx_bits) | row_index`, so the low bits break
/// ties by row index -- reproducing the previous comparator's
/// `.then(left.cmp(&right))` with no explicit tiebreak.
///
/// `idx_bits` is sized to the actual row count rather than fixed at 32: at 4.7M
/// rows only 23 bits are needed for the index, leaving 41 for the timestamp
/// instead of 32.
///
/// `divisor` is the coarsest unit conversion that divides every timestamp
/// exactly. Arrow reports the *declared* unit, which is routinely finer than
/// the data's real resolution -- pandas hands over `timestamp[us]` (or `[ns]`)
/// even when every value is a whole second. Dividing by an exact common divisor
/// is an order isomorphism, so it is free in correctness terms, and it is what
/// lets the key fit for such columns: Brightkite spans ~2.6 years, which is
/// 2^46 microseconds (too wide) but only 2^26 seconds (comfortable).
#[derive(Clone, Copy, Debug)]
struct TimeKeyPlan {
    base: i64,
    idx_bits: u32,
    /// `1` when a null is present, reserving scaled offset `0` for null rows so
    /// they still sort first; `0` otherwise.
    null_offset: u64,
}

impl TimeKeyPlan {
    #[inline(always)]
    fn pack<const DIVISOR: u64>(&self, timestamp: i64, row: usize) -> u64 {
        let offset = if timestamp == NULL_TIMESTAMP {
            0
        } else {
            // Exact: `base` is the minimum non-null timestamp and the span was
            // verified to fit. `wrapping_sub` keeps this at native width rather
            // than widening to i128 per row.
            timestamp.wrapping_sub(self.base) as u64 / DIVISOR + self.null_offset
        };
        (offset << self.idx_bits) | row as u64
    }
}

/// Bits needed to hold every row index in `0..n`.
fn index_bits(n: usize) -> u32 {
    u64::BITS - (n as u64).saturating_sub(1).leading_zeros()
}

/// Build a packing plan, or `None` when no `u64` layout can represent this
/// column. Callers then fall back to sorting row indices through the timestamp
/// buffer, which is what this module did unconditionally before.
fn plan_time_keys(timestamps: &[i64], n: usize) -> Option<(TimeKeyPlan, u64)> {
    let idx_bits = index_bits(n);
    if idx_bits >= u64::BITS {
        return None;
    }
    // A single row needs no index bits at all, leaving the offset all 64 -- a
    // bound no `u64` can hold, and a shift that is undefined rather than merely
    // large. Saturating is exact here: nothing can exceed a 64-bit limit.
    let offset_limit = 1u64.checked_shl(u64::BITS - idx_bits).unwrap_or(u64::MAX);

    let range = scan_time_range(timestamps);
    if range.is_empty() {
        // Every row is null: offsets all collapse to 0, only the index matters.
        return Some((
            TimeKeyPlan {
                base: 0,
                idx_bits,
                null_offset: 0,
            },
            1,
        ));
    }
    let null_offset = u64::from(range.has_null);
    let span = (range.max as i128 - range.min as i128) as u64;

    for &divisor in &CANDIDATE_DIVISORS {
        if (span / divisor as u64) + null_offset >= offset_limit {
            continue;
        }
        // Testing only the coarsest divisor that fits is sufficient: a coarser
        // one dividing the data implies this one does too.
        if divisor == 1 || all_divisible_by(timestamps, divisor) {
            return Some((
                TimeKeyPlan {
                    base: range.min,
                    idx_bits,
                    null_offset,
                },
                divisor as u64,
            ));
        }
    }
    None
}

/// Dispatch on the divisor so the division inside the packing loop is by a
/// compile-time constant (a multiply-high), not a runtime `idiv`.
macro_rules! with_divisor {
    ($divisor:expr, |$name:ident| $body:expr) => {
        match $divisor {
            1_000_000_000 => {
                const $name: u64 = 1_000_000_000;
                $body
            }
            1_000_000 => {
                const $name: u64 = 1_000_000;
                $body
            }
            1_000 => {
                const $name: u64 = 1_000;
                $body
            }
            _ => {
                const $name: u64 = 1;
                $body
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Grouping and the parallel scatter
// ---------------------------------------------------------------------------

/// Reinterpret a zeroed `Vec<u64>` as atomics so the scatter below can write
/// disjoint, data-dependent slots from many threads without raw pointers.
/// Every slot is written exactly once, so `Relaxed` stores suffice and compile
/// to plain stores.
fn zeroed_atomic_u64(len: usize) -> Vec<AtomicU64> {
    let zeros = vec![0u64; len];
    let (pointer, length, capacity) = {
        let mut zeros = std::mem::ManuallyDrop::new(zeros);
        (zeros.as_mut_ptr(), zeros.len(), zeros.capacity())
    };
    // SAFETY: `AtomicU64` has the same size and alignment as `u64` -- the
    // guarantee `AtomicU64::from_mut_slice` rests on -- so the allocation stays
    // valid to use and to free through the new element type.
    unsafe { Vec::from_raw_parts(pointer.cast::<AtomicU64>(), length, capacity) }
}

fn atomics_into_u64(values: Vec<AtomicU64>) -> Vec<u64> {
    let (pointer, length, capacity) = {
        let mut values = std::mem::ManuallyDrop::new(values);
        (values.as_mut_ptr(), values.len(), values.capacity())
    };
    // SAFETY: inverse of `zeroed_atomic_u64` -- same size, alignment and
    // allocation, and every thread that wrote to it has been joined.
    unsafe { Vec::from_raw_parts(pointer.cast::<u64>(), length, capacity) }
}

fn scatter_chunk_len(n: usize, num_groups: usize) -> usize {
    let threads = rayon::current_num_threads().max(1);
    let by_budget = (SCATTER_HISTOGRAM_BUDGET / (num_groups.max(1) * size_of::<u32>())).max(1);
    // The histogram bookkeeping is `chunks * num_groups` work regardless of how
    // few rows each group holds, so a column with very many tiny groups must
    // use fewer chunks or the prefix-sum passes cost more than the parallel
    // scatter saves. Capping at `n / num_groups` keeps that bookkeeping at or
    // below one pass over the rows.
    let by_bookkeeping = (n / num_groups.max(1)).max(1);
    let chunks = threads.min(by_budget).min(by_bookkeeping).max(1);
    n.div_ceil(chunks).max(MIN_SCATTER_CHUNK)
}

/// Group `0..n` by `codes`, placing `value_at(row)` in each row's slot so that
/// every group's values end up contiguous and in ascending row order.
///
/// The scatter used to be one sequential loop, and it measured as the dominant
/// cost of the whole kernel (29.8ms of a 45ms Brightkite call) -- unsurprising,
/// since it is `n` random writes across a buffer far larger than cache with
/// nothing to overlap them against.
///
/// Parallelizing it needs each thread to own disjoint destination slots, which
/// is what the per-chunk histogram buys: counting each chunk's rows per group
/// first gives every (chunk, group) pair its own reserved sub-range, so no two
/// threads ever target the same slot and the writes need no synchronization.
/// The histogram also replaces the sequential counting pass the previous
/// version already needed to build `ranges`, so this adds no pass over `codes`.
fn scatter_by_group<F>(
    codes: &[u32],
    num_groups: usize,
    value_at: F,
) -> Result<(Vec<u64>, IndexRanges), String>
where
    F: Fn(usize) -> u64 + Sync,
{
    let n = codes.len();

    // Validate once, up front, so the hot loops below can index unchecked-ish
    // and so an out-of-range code cannot panic inside a rayon worker.
    if let Some(&highest) = codes.par_iter().max()
        && highest as usize >= num_groups
    {
        return Err("uid code must be less than num_groups".to_string());
    }

    let chunk_len = scatter_chunk_len(n, num_groups);
    let mut histograms: Vec<Vec<u32>> = codes
        .par_chunks(chunk_len)
        .map(|chunk| {
            let mut counts = vec![0u32; num_groups];
            for &code in chunk {
                counts[code as usize] += 1;
            }
            counts
        })
        .collect();

    // Turn per-chunk counts into per-chunk write cursors: for each group, an
    // exclusive prefix sum across chunks. Looping chunk-outer/group-inner keeps
    // both `counts` and `running` sequential in memory.
    let mut running = vec![0u32; num_groups];
    for counts in histograms.iter_mut() {
        for (group, count) in counts.iter_mut().enumerate() {
            let offset = running[group];
            running[group] = offset + *count;
            *count = offset;
        }
    }

    let mut ranges = Vec::with_capacity(num_groups);
    let mut starts = vec![0u32; num_groups];
    let mut current = 0u32;
    for (group, &total) in running.iter().enumerate() {
        starts[group] = current;
        if total != 0 {
            ranges.push((current as u64, (current + total) as u64));
            current += total;
        }
    }
    for counts in histograms.iter_mut() {
        for (group, cursor) in counts.iter_mut().enumerate() {
            *cursor += starts[group];
        }
    }

    let slots = zeroed_atomic_u64(n);
    codes
        .par_chunks(chunk_len)
        .zip(histograms.par_iter_mut())
        .enumerate()
        .for_each(|(chunk_index, (chunk, cursors))| {
            let base = chunk_index * chunk_len;
            for (offset, &code) in chunk.iter().enumerate() {
                let row = base + offset;
                let cursor = &mut cursors[code as usize];
                slots[*cursor as usize].store(value_at(row), Ordering::Relaxed);
                *cursor += 1;
            }
        });

    Ok((atomics_into_u64(slots), ranges))
}

// ---------------------------------------------------------------------------
// Per-group sorting
// ---------------------------------------------------------------------------

/// Split `buf` into one mutable slice per range. `ranges` is contiguous and
/// ascending by construction, so repeated `split_at_mut` covers it exactly --
/// no raw pointers needed.
fn split_groups<'a>(buf: &'a mut [u64], ranges: &[(u64, u64)]) -> Vec<&'a mut [u64]> {
    let mut groups = Vec::with_capacity(ranges.len());
    let mut rest = buf;
    let mut consumed = 0u64;
    for &(start, end) in ranges {
        let tail = std::mem::take(&mut rest);
        let (_, tail) = tail.split_at_mut((start - consumed) as usize);
        let (group, tail) = tail.split_at_mut((end - start) as usize);
        groups.push(group);
        rest = tail;
        consumed = end;
    }
    groups
}

fn sort_group_packed(group: &mut [u64], _solo: bool) {
    // Trajectory data very often arrives already in time order, in which case
    // every group is sorted and this collapses to a linear scan.
    if group.len() <= 1 || group.is_sorted() {
        return;
    }
    if group.len() <= INSERTION_SORT_LEN {
        for position in 1..group.len() {
            let value = group[position];
            let mut hole = position;
            while hole > 0 && group[hole - 1] > value {
                group[hole] = group[hole - 1];
                hole -= 1;
            }
            group[hole] = value;
        }
    } else if group.len() >= PARALLEL_GROUP_LEN {
        group.par_sort_unstable();
    } else {
        group.sort_unstable();
    }
}

/// Fallback ordering for columns whose timestamps cannot be packed: sort row
/// indices through the timestamp buffer, as this module always did.
///
/// `solo` means this group is the entire workload, which decides how to sort a
/// large one. Measured on a 4M-row column: when it is the only group, rayon's
/// stable `par_sort_by` beats `par_sort_unstable_by` (52ms vs 69ms) because a
/// merge sort's predictable access pattern tolerates the comparator's random
/// reads better than pdqsort's random partitioning. But when several large
/// groups sort concurrently, the stable sort's per-sort `O(len)` scratch buffer
/// turns into real allocation and memory pressure, and it loses badly instead
/// (60ms vs 28ms on eight 500k-row groups). Hence the split rather than one
/// blanket choice.
fn sort_group_indirect(group: &mut [u64], timestamps: &[i64], solo: bool) {
    let compare = |&left: &u64, &right: &u64| {
        timestamps[left as usize]
            .cmp(&timestamps[right as usize])
            .then(left.cmp(&right))
    };
    if group.len() <= 1 {
        return;
    }
    if group.len() < PARALLEL_GROUP_LEN {
        group.sort_unstable_by(compare);
    } else if solo {
        group.par_sort_by(compare);
    } else {
        group.par_sort_unstable_by(compare);
    }
}

fn sort_all_groups<F>(buf: &mut [u64], ranges: &[(u64, u64)], sort_one: F)
where
    F: Fn(&mut [u64], bool) + Sync + Send,
{
    let mut groups = split_groups(buf, ranges);
    // Whether a single group is the whole workload -- see `sort_group_indirect`.
    let solo = groups.len() == 1;
    if groups.len() <= MAX_GROUPS_TO_ORDER_BY_LENGTH {
        // Longest first, so rayon starts the expensive groups before the cheap
        // ones and the schedule's tail is made of small tasks. Only worth doing
        // while the group list is short enough that ordering it is negligible.
        groups.sort_unstable_by_key(|group| std::cmp::Reverse(group.len()));
    }
    groups
        .par_iter_mut()
        .for_each(|group| sort_one(group, solo));
}

/// Strip the sort key, leaving the row indices the caller asked for. In place,
/// so the packed path never allocates a second `n`-sized buffer.
fn unpack_rows(keys: &mut [u64], idx_bits: u32) {
    if idx_bits >= u64::BITS {
        return;
    }
    let mask = (1u64 << idx_bits) - 1;
    keys.par_iter_mut().for_each(|key| *key &= mask);
}

// ---------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------

pub fn time_ordered_indices_single_user(timestamps: &[i64]) -> OrderedIndexRanges {
    let n = timestamps.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let ranges = vec![(0u64, n as u64)];

    if let Some((plan, divisor)) = plan_time_keys(timestamps, n) {
        let mut keys: Vec<u64> = with_divisor!(divisor, |DIVISOR| timestamps
            .par_iter()
            .enumerate()
            .map(|(row, &timestamp)| plan.pack::<DIVISOR>(timestamp, row))
            .collect());
        sort_all_groups(&mut keys, &ranges, sort_group_packed);
        unpack_rows(&mut keys, plan.idx_bits);
        return (keys, ranges);
    }

    let mut indices: Vec<u64> = (0..n as u64).collect();
    sort_all_groups(&mut indices, &ranges, |group, solo| {
        sort_group_indirect(group, timestamps, solo)
    });
    (indices, ranges)
}

pub fn time_ordered_indices_for_u32_codes(
    codes: &[u32],
    timestamps: &[i64],
    num_groups: usize,
) -> Result<OrderedIndexRanges, String> {
    let n = codes.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if timestamps.len() != n {
        return Err(format!(
            "uid codes and timestamps must have the same length (got {n} and {})",
            timestamps.len()
        ));
    }
    if num_groups == 0 {
        return Err(
            "num_groups must be greater than zero when uid codes are not empty".to_string(),
        );
    }

    match plan_time_keys(timestamps, n) {
        Some((plan, divisor)) => {
            let (mut keys, ranges) = with_divisor!(divisor, |DIVISOR| scatter_by_group(
                codes,
                num_groups,
                |row| plan.pack::<DIVISOR>(timestamps[row], row)
            ))?;
            sort_all_groups(&mut keys, &ranges, sort_group_packed);
            unpack_rows(&mut keys, plan.idx_bits);
            Ok((keys, ranges))
        }
        None => {
            let (mut indices, ranges) = scatter_by_group(codes, num_groups, |row| row as u64)?;
            sort_all_groups(&mut indices, &ranges, |group, solo| {
                sort_group_indirect(group, timestamps, solo)
            });
            Ok((indices, ranges))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Straightforward restatement of the contract, used as the oracle below.
    fn reference(codes: &[u32], timestamps: Option<&[i64]>, check_timestamps: bool) -> bool {
        let timestamps = if check_timestamps {
            match timestamps {
                Some(values) if values.len() == codes.len() => Some(values),
                _ => return false,
            }
        } else {
            None
        };
        let mut seen = Vec::new();
        let mut start = 0usize;
        while start < codes.len() {
            let leader = codes[start];
            let mut end = start + 1;
            while end < codes.len() && codes[end] == leader {
                end += 1;
            }
            if let Some(timestamps) = timestamps
                && timestamps[start..end].windows(2).any(|w| w[1] < w[0])
            {
                return false;
            }
            if seen.contains(&leader) {
                return false;
            }
            seen.push(leader);
            start = end;
        }
        true
    }

    fn assert_matches_reference(codes: &[u32], timestamps: &[i64]) {
        for check in [false, true] {
            assert_eq!(
                validate_grouped_u32_codes(codes, Some(timestamps), check),
                reference(codes, Some(timestamps), check),
                "codes={:?} check_timestamps={check}",
                &codes[..codes.len().min(16)]
            );
        }
    }

    #[test]
    fn accepts_empty_and_single_row_input() {
        assert!(validate_grouped_u32_codes(&[], Some(&[]), true));
        assert!(validate_grouped_u32_codes(&[7], Some(&[5]), true));
    }

    #[test]
    fn rejects_missing_or_mismatched_timestamps_only_when_checking() {
        assert!(!validate_grouped_u32_codes(&[1, 1], None, true));
        assert!(!validate_grouped_u32_codes(&[1, 1], Some(&[0]), true));
        assert!(validate_grouped_u32_codes(&[1, 1], None, false));
    }

    #[test]
    fn accepts_contiguous_groups_with_rising_timestamps() {
        assert_matches_reference(&[2, 2, 9, 9, 4], &[1, 2, 0, 5, 3]);
    }

    #[test]
    fn rejects_a_group_that_reappears() {
        assert!(!validate_grouped_u32_codes(
            &[1, 2, 1],
            Some(&[0, 0, 0]),
            false
        ));
        // Non-monotone leaders that are still unique stay valid.
        assert!(validate_grouped_u32_codes(
            &[5, 1, 3],
            Some(&[0, 0, 0]),
            false
        ));
    }

    #[test]
    fn rejects_backwards_timestamps_only_inside_a_group() {
        assert!(!validate_grouped_u32_codes(&[1, 1], Some(&[5, 4]), true));
        // The drop happens across a group boundary, which is unconstrained.
        assert!(validate_grouped_u32_codes(&[1, 2], Some(&[5, 4]), true));
    }

    #[test]
    fn validates_timestamps_against_supplied_run_ends() {
        // Two runs: [0,2) and [2,4). The drop at index 2 is a run boundary.
        assert!(validate_non_decreasing_within_ends(&[1, 2, 0, 5], &[2, 4]));
        // The same drop inside a single run is a violation.
        assert!(!validate_non_decreasing_within_ends(&[1, 2, 0, 5], &[4]));
        assert!(validate_non_decreasing_within_ends(&[], &[]));
        // Ends running past the timestamps are rejected rather than panicking.
        assert!(!validate_non_decreasing_within_ends(&[1, 2], &[5]));
    }

    #[test]
    fn validates_single_group_timestamps() {
        assert!(validate_non_decreasing_timestamps(&[]));
        assert!(validate_non_decreasing_timestamps(&[3]));
        assert!(validate_non_decreasing_timestamps(&[1, 1, 2, 9]));
        assert!(!validate_non_decreasing_timestamps(&[1, 0]));
        // Long enough to cross into the parallel path and its chunk seams.
        let mut rising: Vec<i64> = (0..(VALIDATE_SEQUENTIAL_BELOW as i64 * 4)).collect();
        assert!(validate_non_decreasing_timestamps(&rising));
        for at in [1usize, VALIDATE_MIN_CHUNK, rising.len() - 1] {
            let saved = rising[at];
            rising[at] = -1;
            assert!(
                !validate_non_decreasing_timestamps(&rising),
                "break at {at}"
            );
            rising[at] = saved;
        }
    }

    #[test]
    fn treats_null_timestamps_as_the_smallest_value() {
        assert!(validate_grouped_u32_codes(
            &[1, 1, 1],
            Some(&[NULL_TIMESTAMP, 3, 4]),
            true
        ));
        assert!(!validate_grouped_u32_codes(
            &[1, 1, 1],
            Some(&[3, 4, NULL_TIMESTAMP]),
            true
        ));
    }

    /// Every shape below is larger than `VALIDATE_SEQUENTIAL_BELOW`, so these
    /// exercise the parallel path, its chunk boundaries, and both fallbacks.
    #[test]
    fn parallel_path_matches_reference_on_large_inputs() {
        let n = 250_000usize;

        let ascending: Vec<u32> = (0..n).map(|row| (row / 97) as u32).collect();
        let descending: Vec<u32> = ascending.iter().map(|&code| u32::MAX - code).collect();
        // Group order that is neither ascending nor descending: forces the
        // non-monotone duplicate test.
        let shuffled: Vec<u32> = ascending
            .iter()
            .map(|&code| (code.wrapping_mul(2_654_435_761)) >> 8)
            .collect();
        let single = vec![11u32; n];
        let sparse: Vec<u32> = ascending
            .iter()
            .map(|&code| code.wrapping_mul(2_654_435_761))
            .collect();
        let mut repeated = ascending.clone();
        repeated[n - 1] = repeated[0];

        let rising: Vec<i64> = (0..n as i64).collect();
        for codes in [
            &ascending,
            &descending,
            &shuffled,
            &single,
            &sparse,
            &repeated,
        ] {
            assert_matches_reference(codes, &rising);
        }

        // A single backwards step inside a group, at several offsets, including
        // one straddling a chunk boundary.
        for at in [1usize, VALIDATE_MIN_CHUNK, n / 2, n - 1] {
            let mut broken = rising.clone();
            broken[at] = -1;
            assert_matches_reference(&ascending, &broken);
            assert_matches_reference(&single, &broken);
        }
    }
}
