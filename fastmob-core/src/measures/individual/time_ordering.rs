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

pub fn presorted_ranges_for_u32_codes(codes: &[u32]) -> (Vec<u64>, Vec<u64>) {
    let n = codes.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let boundaries: Vec<u64> = codes
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
    let offset_limit = 1u64 << (u64::BITS - idx_bits);

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
