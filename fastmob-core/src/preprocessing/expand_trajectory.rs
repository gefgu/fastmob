use rayon::prelude::*;
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

/// (user_range_idx, timestamp_ms, source_row_idx) -- one triple per emitted
/// 5-minute slice. `source_row_idx` is the original input row that produced
/// the slice, so the Python side can `pc.take` any per-row column (location,
/// purpose, ...) it needs without this kernel knowing about them at all.
pub type ExpandTrajectoryBatchResult = (Vec<usize>, Vec<i64>, Vec<usize>);

const FIVE_MIN_MS: i64 = 5 * 60 * 1000;

/// `fastmob.utils._common._extract_timestamps`'s null-datetime-row sentinel
/// (`i64::MIN`), matched exactly -- see that function's docstring for why a
/// sentinel is used instead of a nullable Arrow array here.
const NULL_TIMESTAMP_SENTINEL_MS: i64 = i64::MIN;

fn ceil_5min_ms(t: i64) -> i64 {
    let rem = t.rem_euclid(FIVE_MIN_MS);
    if rem == 0 { t } else { t + (FIVE_MIN_MS - rem) }
}

fn floor_5min_ms(t: i64) -> i64 {
    t - t.rem_euclid(FIVE_MIN_MS)
}

struct ExpandedRow {
    timestamp_ms: i64,
    source_row_idx: usize,
}

/// Expands one user's (start_ms, end_ms) intervals into 5-minute-aligned
/// slices between `ceil_5min(start)` and `floor_5min(end)` inclusive.
/// `user_indices` must already be ordered by (start_timestamp, original row
/// index) -- exactly what `time_ordered_user_indices` produces -- so that
/// "keep the first input occurrence" dedup (a `HashSet` on timestamp alone,
/// since this runs per-user there's no cross-user key collision to worry
/// about) matches the original row order deterministically, including ties.
fn expand_5min_for_user(
    start_ms: &[i64],
    end_ms: &[i64],
    user_indices: &[usize],
) -> Vec<ExpandedRow> {
    let mut rows: Vec<ExpandedRow> = Vec::new();
    let mut seen: HashSet<i64> = HashSet::default();
    for &idx in user_indices {
        let start = start_ms[idx];
        let end = end_ms[idx];
        if start == NULL_TIMESTAMP_SENTINEL_MS || end == NULL_TIMESTAMP_SENTINEL_MS {
            continue;
        }
        let mut t = ceil_5min_ms(start);
        let last = floor_5min_ms(end);
        while t <= last {
            if seen.insert(t) {
                rows.push(ExpandedRow {
                    timestamp_ms: t,
                    source_row_idx: idx,
                });
            }
            t += FIVE_MIN_MS;
        }
    }
    rows
}

pub fn expand_5min_trajectory_batch_indexed_impl(
    start_ms: &[i64],
    end_ms: &[i64],
    sorted_indices: &[usize],
    ends: &[usize],
) -> ExpandTrajectoryBatchResult {
    let per_user_rows: Vec<Vec<ExpandedRow>> = (0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            expand_5min_for_user(start_ms, end_ms, &sorted_indices[start..end])
        })
        .collect();

    let n: usize = per_user_rows.iter().map(|rows| rows.len()).sum();
    let mut user_range_idx = Vec::with_capacity(n);
    let mut timestamps_ms = Vec::with_capacity(n);
    let mut source_row_idx = Vec::with_capacity(n);
    for (range_idx, rows) in per_user_rows.into_iter().enumerate() {
        for row in rows {
            user_range_idx.push(range_idx);
            timestamps_ms.push(row.timestamp_ms);
            source_row_idx.push(row.source_row_idx);
        }
    }
    (user_range_idx, timestamps_ms, source_row_idx)
}

// --------------------------------------------------------------------------
// impute_gaps=True variant: fills missing 5-minute slices between a user's
// first and last observed slice using per-user "anchor" locations inferred
// from three fixed hour-of-day windows. Mirrors
// `fastmob.measures.individual.mobility_profiling._impute_5min_gaps` /
// `_gap_imputation_anchors` / `_anchor_for_hour` exactly (including Python
// `max(dict, key=dict.get)`'s tie-break-by-first-insertion-order semantics,
// replicated below via `AnchorTally`'s `first_seen` field), operating
// per-user in the same rayon pass as the plain expansion above rather than
// as a separate Python post-pass over the flattened ~24M-row output.
/// (user_range_idx, run_start_timestamp_ms, location_code, run_length) --
/// one quadruple per *maximal run of consecutive same-location slices*
/// (observed and/or imputed), not one row per slice. Safe to collapse this
/// way because the only downstream consumer
/// (`intermittance_and_degree_of_return`'s block-length aggregation in
/// `fastmob/measures/individual/mobility_profiling.py`) only cares about
/// row-adjacency and per-block row counts, never real elapsed time within a
/// block -- a skipped (un-anchored) gap slice produces no row in the
/// uncompressed form either, so it never breaks adjacency there, and
/// therefore never needs to break a compressed run here. `run_length` sums
/// to what the uncompressed slice count would have been; the caller turns
/// `.count()` into `.sum(run_length)` to get identical block lengths.
/// Collapses a `home`-anchored 3-hour nightly stretch (36 slices) into one
/// row instead of 36 -- the dominant source of `impute_gaps=True`'s
/// otherwise-unbounded memory growth for users with a long, sparse
/// check-in history (see `measures_notes.md` in fastmob_benchmarks for the
/// concrete before/after).
pub type ExpandTrajectoryWithImputationBatchResult = (Vec<usize>, Vec<i64>, Vec<u32>, Vec<u32>);

const HOUR_MS: i64 = 60 * 60 * 1000;

/// UTC-equivalent hour-of-day (0-23) for a millisecond timestamp, matching
/// `fastmob.utils._common._extract_timestamps`'s ms-since-epoch convention
/// (naive datetimes are treated as UTC-equivalent), which is exactly what a
/// naive pandas `Timestamp.hour` reads for the same wall-clock value.
fn hour_of_day(timestamp_ms: i64) -> u32 {
    timestamp_ms.div_euclid(HOUR_MS).rem_euclid(24) as u32
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum AnchorWindow {
    Home,
    WorkA,
    WorkB,
}

fn anchor_window_for_hour(hour: u32) -> Option<AnchorWindow> {
    match hour {
        2..=5 => Some(AnchorWindow::Home),
        10 => Some(AnchorWindow::WorkA),
        14..=16 => Some(AnchorWindow::WorkB),
        _ => None,
    }
}

/// Tracks (count, first-insertion-position) per location so ties resolve
/// the same way Python's `max(dict, key=dict.get)` does: the location whose
/// *first* occurrence (in the window's row-iteration order) came earliest
/// wins among equally-frequent candidates.
#[derive(Default)]
struct AnchorTally {
    counts: HashMap<u32, (u64, usize)>,
    next_position: usize,
}

impl AnchorTally {
    fn record(&mut self, location_code: u32) {
        let next_position = self.next_position;
        let entry = self
            .counts
            .entry(location_code)
            .or_insert((0, next_position));
        entry.0 += 1;
        self.next_position += 1;
    }

    fn winner(&self) -> Option<u32> {
        self.counts
            .iter()
            .max_by(|(_, (count_a, pos_a)), (_, (count_b, pos_b))| {
                count_a.cmp(count_b).then(pos_b.cmp(pos_a))
            })
            .map(|(&location_code, _)| location_code)
    }
}

/// Expands one user's intervals (as [`expand_5min_for_user`] does), fills
/// gaps in `[first_observed, last_observed]` using hour-window anchors, and
/// run-length-encodes the result: `(run_start_ms, location_code,
/// run_length)`, one entry per maximal run of consecutive same-location
/// slices. See [`ExpandTrajectoryWithImputationBatchResult`]'s docs for why
/// collapsing this way is safe (a skipped/un-anchored gap slice never
/// breaks a run, matching the uncompressed form's row-adjacency semantics).
fn expand_5min_with_imputation_for_user(
    start_ms: &[i64],
    end_ms: &[i64],
    location_codes: &[u32],
    user_indices: &[usize],
) -> Vec<(i64, u32, u32)> {
    let observed_rows = expand_5min_for_user(start_ms, end_ms, user_indices);
    if observed_rows.is_empty() {
        return Vec::new();
    }

    let mut observed: Vec<(i64, u32)> = observed_rows
        .into_iter()
        .map(|row| (row.timestamp_ms, location_codes[row.source_row_idx]))
        .collect();
    observed.sort_unstable_by_key(|&(timestamp_ms, _)| timestamp_ms);

    let mut tallies: HashMap<AnchorWindow, AnchorTally> = HashMap::default();
    for &(timestamp_ms, location_code) in &observed {
        if let Some(window) = anchor_window_for_hour(hour_of_day(timestamp_ms)) {
            tallies.entry(window).or_default().record(location_code);
        }
    }
    let anchor_for = |window: AnchorWindow| -> Option<u32> {
        tallies.get(&window).and_then(AnchorTally::winner)
    };
    let home_anchor = anchor_for(AnchorWindow::Home);
    let work_a_anchor = anchor_for(AnchorWindow::WorkA);
    let work_b_anchor = anchor_for(AnchorWindow::WorkB);

    let observed_by_timestamp: HashMap<i64, u32> = observed.iter().copied().collect();
    let first_timestamp = observed.first().unwrap().0;
    let last_timestamp = observed.last().unwrap().0;

    // Tracks, per user, every location seen so far in this walk -- a
    // location's very first-ever slice is this user's "exploration" of it;
    // every later slice of the same location (even the very next one) is a
    // "return". `intermittance_and_degree_of_return`'s block detection
    // needs that transition to land between two separate blocks even when
    // the location doesn't change (e.g. `home` observed once, then imputed
    // solidly afterward: the first `home` slice is its own 1-slice
    // exploration block, every subsequent `home` slice is one return
    // block) -- so a run may only absorb a slice when *both* the location
    // and this explore/return state match the run's, not location alone.
    let mut seen: HashSet<u32> = HashSet::default();
    let mut result: Vec<(i64, u32, u32)> = Vec::new();
    let mut current_run: Option<(i64, u32, u32, bool)> = None; // (run_start_ms, location_code, run_length, is_known)

    let mut t = first_timestamp;
    while t <= last_timestamp {
        let location = observed_by_timestamp.get(&t).copied().or_else(|| {
            match anchor_window_for_hour(hour_of_day(t)) {
                Some(AnchorWindow::Home) => home_anchor,
                Some(AnchorWindow::WorkA) => work_a_anchor,
                Some(AnchorWindow::WorkB) => work_b_anchor,
                None => None,
            }
        });
        if let Some(location) = location {
            let is_known = !seen.insert(location); // insert() returns false if already present
            match &mut current_run {
                Some((_, run_location, run_length, run_is_known))
                    if *run_location == location && *run_is_known == is_known =>
                {
                    *run_length += 1;
                }
                _ => {
                    if let Some(run) = current_run.take() {
                        result.push((run.0, run.1, run.2));
                    }
                    current_run = Some((t, location, 1, is_known));
                }
            }
        }
        // A skipped slice (no observation, no anchor) leaves `current_run`
        // untouched -- it doesn't start a new run or close the current one,
        // matching the uncompressed form (no row at all for that slice).
        t += FIVE_MIN_MS;
    }
    if let Some(run) = current_run.take() {
        result.push((run.0, run.1, run.2));
    }
    result
}

pub fn expand_5min_trajectory_with_imputation_batch_indexed_impl(
    start_ms: &[i64],
    end_ms: &[i64],
    location_codes: &[u32],
    sorted_indices: &[usize],
    ends: &[usize],
) -> ExpandTrajectoryWithImputationBatchResult {
    let per_user_runs: Vec<Vec<(i64, u32, u32)>> = (0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            expand_5min_with_imputation_for_user(
                start_ms,
                end_ms,
                location_codes,
                &sorted_indices[start..end],
            )
        })
        .collect();

    let n: usize = per_user_runs.iter().map(|runs| runs.len()).sum();
    let mut user_range_idx = Vec::with_capacity(n);
    let mut timestamps_ms = Vec::with_capacity(n);
    let mut location_out = Vec::with_capacity(n);
    let mut run_length_out = Vec::with_capacity(n);
    for (range_idx, runs) in per_user_runs.into_iter().enumerate() {
        for (run_start_ms, location_code, run_length) in runs {
            user_range_idx.push(range_idx);
            timestamps_ms.push(run_start_ms);
            location_out.push(location_code);
            run_length_out.push(run_length);
        }
    }
    (user_range_idx, timestamps_ms, location_out, run_length_out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aligned_stay_yields_inclusive_5min_slices() {
        // 08:00 to 08:10 (ms since epoch, arbitrary base) -> 08:00, 08:05, 08:10.
        let base = 1_600_000_000_000i64; // arbitrary 5-min-aligned-ish base, corrected below
        let base = base - base.rem_euclid(FIVE_MIN_MS);
        let start = vec![base];
        let end = vec![base + 10 * 60 * 1000];
        let sorted_indices = vec![0usize];
        let ends = vec![1usize];
        let (user_idx, ts, src) =
            expand_5min_trajectory_batch_indexed_impl(&start, &end, &sorted_indices, &ends);
        assert_eq!(user_idx, vec![0, 0, 0]);
        assert_eq!(ts, vec![base, base + 5 * 60 * 1000, base + 10 * 60 * 1000]);
        assert_eq!(src, vec![0, 0, 0]);
    }

    #[test]
    fn non_aligned_stay_yields_inner_slices_only() {
        let base = 1_600_000_000_000i64;
        let base = base - base.rem_euclid(FIVE_MIN_MS);
        let start = vec![base + 2 * 60 * 1000]; // +02:00
        let end = vec![base + 13 * 60 * 1000]; // +13:00
        let sorted_indices = vec![0usize];
        let ends = vec![1usize];
        let (_user_idx, ts, _src) =
            expand_5min_trajectory_batch_indexed_impl(&start, &end, &sorted_indices, &ends);
        assert_eq!(ts, vec![base + 5 * 60 * 1000, base + 10 * 60 * 1000]);
    }

    #[test]
    fn duplicate_timestamp_across_rows_keeps_first_occurrence() {
        let base = 1_600_000_000_000i64;
        let base = base - base.rem_euclid(FIVE_MIN_MS);
        // Row 0: base..base (single slice at `base`, source_row_idx 0).
        // Row 1: base..base (same slice, should be skipped -- row 0 already claimed it).
        let start = vec![base, base];
        let end = vec![base, base];
        let sorted_indices = vec![0usize, 1usize];
        let ends = vec![2usize];
        let (_user_idx, ts, src) =
            expand_5min_trajectory_batch_indexed_impl(&start, &end, &sorted_indices, &ends);
        assert_eq!(ts, vec![base]);
        assert_eq!(src, vec![0]);
    }

    #[test]
    fn null_sentinel_rows_are_skipped() {
        let start = vec![NULL_TIMESTAMP_SENTINEL_MS, 0];
        let end = vec![NULL_TIMESTAMP_SENTINEL_MS, 0];
        let sorted_indices = vec![0usize, 1usize];
        let ends = vec![2usize];
        let (_user_idx, ts, src) =
            expand_5min_trajectory_batch_indexed_impl(&start, &end, &sorted_indices, &ends);
        assert_eq!(ts, vec![0]);
        assert_eq!(src, vec![1]);
    }

    // location codes: home=0, shop=1 (matches
    // `test_impute_gaps_fills_missing_nighttime_slice_with_home_anchor` /
    // `test_impute_gaps_leaves_missing_slice_absent_without_anchor` in
    // tests/correctness/individual/test_intermittance.py).
    const HOME: u32 = 0;
    const SHOP: u32 = 1;

    fn ms(hour: i64, minute: i64) -> i64 {
        hour * HOUR_MS + minute * 60_000
    }

    #[test]
    fn imputation_fills_nighttime_gap_with_home_anchor() {
        // Row 0: home, 02:00-02:05 -> slices 02:00, 02:05.
        // Row 1: shop, 02:15-02:15 -> slice 02:15.
        // Gap at 02:10 (hour 2, Home window); observed home count (2) beats
        // shop count (1) in that window, so 02:10 is imputed as home.
        // Compressed into 3 runs, not 2: the very first `home` slice
        // (02:00) is this user's first-ever occurrence of `home` --
        // "exploration" -- and must stay its own run even though 02:05 and
        // the imputed 02:10 (both "return", already-seen `home`) compress
        // together into a second run. Matches
        // `test_impute_gaps_fills_missing_nighttime_slice_with_home_anchor`'s
        // expected mean_exploration=1.0/mean_return=2.0 in
        // tests/correctness/individual/test_intermittance.py.
        let start = vec![ms(2, 0), ms(2, 15)];
        let end = vec![ms(2, 5), ms(2, 15)];
        let location_codes = vec![HOME, SHOP];
        let sorted_indices = vec![0usize, 1usize];
        let ends = vec![2usize];
        let (_user_idx, ts, loc, run_len) =
            expand_5min_trajectory_with_imputation_batch_indexed_impl(
                &start,
                &end,
                &location_codes,
                &sorted_indices,
                &ends,
            );
        assert_eq!(ts, vec![ms(2, 0), ms(2, 5), ms(2, 15)]);
        assert_eq!(loc, vec![HOME, HOME, SHOP]);
        assert_eq!(run_len, vec![1, 2, 1]);
    }

    #[test]
    fn imputation_leaves_gap_absent_outside_anchor_windows() {
        // Same shape, but at 08:00/08:15 -- hour 8 is in none of the three
        // anchor windows, so the 08:10 gap stays unfilled. 08:00 (first-ever
        // `home`, exploration) and 08:05 (return) still can't merge -- same
        // explore/return split as above, just with no compression benefit
        // here since 08:05 has no further same-state neighbor to absorb.
        let start = vec![ms(8, 0), ms(8, 15)];
        let end = vec![ms(8, 5), ms(8, 15)];
        let location_codes = vec![HOME, SHOP];
        let sorted_indices = vec![0usize, 1usize];
        let ends = vec![2usize];
        let (_user_idx, ts, loc, run_len) =
            expand_5min_trajectory_with_imputation_batch_indexed_impl(
                &start,
                &end,
                &location_codes,
                &sorted_indices,
                &ends,
            );
        assert_eq!(ts, vec![ms(8, 0), ms(8, 5), ms(8, 15)]);
        assert_eq!(loc, vec![HOME, HOME, SHOP]);
        assert_eq!(run_len, vec![1, 1, 1]);
    }

    #[test]
    fn long_run_across_a_gap_compresses_to_a_single_return_record() {
        // Two home observations three days apart; every home-window slice
        // (hours 2-5) in between gets imputed as home, uninterrupted by any
        // other location. The very first slice (day 0, 02:00) is `home`'s
        // first-ever occurrence (exploration, run length 1); every
        // subsequent home slice across the whole 3-day span is "return" and
        // compresses into ONE run -- this is the actual memory-blowup fix:
        // hundreds of slices collapsing to a single row instead of one row
        // per 5-minute slice.
        let three_days_ms = 3 * 24 * HOUR_MS;
        let start = vec![ms(2, 0), three_days_ms + ms(2, 0)];
        let end = vec![ms(2, 0), three_days_ms + ms(2, 0)];
        let location_codes = vec![HOME, HOME];
        let sorted_indices = vec![0usize, 1usize];
        let ends = vec![2usize];
        let (_user_idx, ts, loc, run_len) =
            expand_5min_trajectory_with_imputation_batch_indexed_impl(
                &start,
                &end,
                &location_codes,
                &sorted_indices,
                &ends,
            );
        assert_eq!(ts, vec![ms(2, 0), ms(2, 5)]);
        assert_eq!(loc, vec![HOME, HOME]);
        assert_eq!(run_len[0], 1, "first slice is home's first-ever occurrence");
        assert!(
            run_len[1] > 100,
            "expected a long compressed return run, got {}",
            run_len[1]
        );
    }
}
