use std::time::Instant;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::utils::{ranges_from_ends, validate_indexed_ends};

type DailyMotifsResult = Result<(Vec<usize>, Vec<i32>, Vec<i64>), String>;

const MICROS_PER_HOUR: i64 = 3_600_000_000;
const MICROS_PER_DAY: i64 = 86_400_000_000;

#[derive(Debug)]
struct DailyMotifResult {
    user_idx: usize,
    date_id: i32,
    motif_id: i64,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
struct NodeCode {
    location: u32,
    purpose: u16,
}

#[derive(Default)]
struct Scratch {
    cleaned: Vec<NodeCode>,
    motif_cache: FxHashMap<(u8, u64), i64>,
}

/// Where a row's purpose code comes from: either a flat array (one value
/// per visit row -- used by the standalone `daily_motifs` API, where the
/// caller already supplies a purpose column), or a small
/// `(user_idx, location_code) -> purpose_code` lookup built once from a
/// separate Locations-grain table (used by the Staypoints/Locations
/// hierarchy integration), read only, shared across `rayon` threads with no
/// locking needed.
enum PurposeSource<'a> {
    Flat(&'a [u16]),
    Lookup {
        table: &'a FxHashMap<(u32, u32), u16>,
        unmatched_code: u16,
    },
}

struct MotifColumns<'a> {
    location_codes: &'a [u32],
    purpose_source: PurposeSource<'a>,
    start_timestamps_us: &'a [i64],
    end_timestamps_us: &'a [i64],
    durations: Option<&'a [f64]>,
    home_purpose_code: u16,
}

impl MotifColumns<'_> {
    #[inline]
    fn purpose_at(&self, user_idx: usize, row: usize) -> u16 {
        match &self.purpose_source {
            PurposeSource::Flat(codes) => codes[row],
            PurposeSource::Lookup {
                table,
                unmatched_code,
            } => table
                .get(&(user_idx as u32, self.location_codes[row]))
                .copied()
                .unwrap_or(*unmatched_code),
        }
    }

    #[inline]
    fn node(&self, user_idx: usize, row: usize) -> NodeCode {
        NodeCode {
            location: self.location_codes[row],
            purpose: self.purpose_at(user_idx, row),
        }
    }

    #[inline]
    fn start_hour(&self, row: usize) -> u8 {
        hour_from_timestamp(self.start_timestamps_us[row])
    }

    #[inline]
    fn end_hour(&self, row: usize) -> u8 {
        hour_from_timestamp(self.end_timestamps_us[row])
    }

    #[inline]
    fn date_id(&self, row: usize) -> i32 {
        self.start_timestamps_us[row].div_euclid(MICROS_PER_DAY) as i32
    }

    #[inline]
    fn duration(&self, row: usize) -> f64 {
        self.durations.map_or(0.0, |values| values[row])
    }

    #[inline]
    fn is_home(&self, user_idx: usize, row: usize) -> bool {
        self.purpose_at(user_idx, row) == self.home_purpose_code
    }
}

#[inline]
fn hour_from_timestamp(timestamp_us: i64) -> u8 {
    (timestamp_us.rem_euclid(MICROS_PER_DAY) / MICROS_PER_HOUR) as u8
}

#[inline]
fn ordered_row(indices: Option<&[usize]>, position: usize) -> usize {
    indices.map_or(position, |values| values[position])
}

fn next_permutation(v: &mut [usize]) -> bool {
    let n = v.len();
    if n < 2 {
        return false;
    }
    let mut i = n - 1;
    loop {
        if i == 0 {
            return false;
        }
        i -= 1;
        if v[i] < v[i + 1] {
            break;
        }
    }
    let mut j = n - 1;
    while v[j] <= v[i] {
        j -= 1;
    }
    v.swap(i, j);
    v[i + 1..].reverse();
    true
}

#[inline]
fn cell_bit(n: usize, i: usize, j: usize) -> u64 {
    1u64 << (n * n - 1 - (i * n + j))
}

#[inline]
fn is_home_code(code: NodeCode, home_purpose_code: u16) -> bool {
    code.purpose == home_purpose_code
}

fn edges_to_mask(n: usize, edges: &[(u32, u32)]) -> u64 {
    let mut adj = 0u64;
    for &(u, v) in edges {
        adj |= cell_bit(n, u as usize, v as usize);
    }
    adj
}

fn canonical_adjacency_mask(n: usize, adj: u64) -> i64 {
    if n == 0 {
        return 0;
    }
    let mut perm = [0usize; 6];
    for (k, slot) in perm.iter_mut().enumerate().take(n) {
        *slot = k;
    }
    let mut best = 0u64;
    loop {
        let mut val = 0u64;
        let mut rest = adj;
        while rest != 0 {
            let bit = rest.trailing_zeros() as usize;
            rest &= rest - 1;
            let cell = n * n - 1 - bit;
            val |= cell_bit(n, perm[cell / n], perm[cell % n]);
        }
        best = best.max(val);
        if !next_permutation(&mut perm[1..n]) {
            break;
        }
    }
    ((n as i64) << 36) | best as i64
}

pub fn canonical_adjacency_form_internal(
    n_nodes: usize,
    edges: &[(u32, u32)],
) -> Result<i64, String> {
    if n_nodes > 6 {
        return Err("This optimized function supports at most 6 nodes".to_string());
    }
    Ok(canonical_adjacency_mask(
        n_nodes,
        edges_to_mask(n_nodes, edges),
    ))
}

pub fn canonical_adjacency_form(n_nodes: u32, edges: Vec<(u32, u32)>) -> Result<i64, String> {
    canonical_adjacency_form_internal(n_nodes as usize, &edges)
}

fn compute_primary_home(
    user_idx: usize,
    columns: &MotifColumns<'_>,
    indices: Option<&[usize]>,
    start: usize,
    end: usize,
) -> Option<NodeCode> {
    let mut night_duration: FxHashMap<NodeCode, f64> = FxHashMap::default();
    for position in start..end {
        let row = ordered_row(indices, position);
        if columns.is_home(user_idx, row)
            && (columns.start_hour(row) >= 22 || columns.start_hour(row) < 6)
        {
            *night_duration
                .entry(columns.node(user_idx, row))
                .or_insert(0.0) += columns.duration(row);
        }
    }
    if !night_duration.is_empty() {
        return night_duration
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(code, _)| code);
    }

    let mut home_counts: FxHashMap<NodeCode, usize> = FxHashMap::default();
    for position in start..end {
        let row = ordered_row(indices, position);
        if columns.is_home(user_idx, row) {
            *home_counts.entry(columns.node(user_idx, row)).or_insert(0) += 1;
        }
    }
    home_counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(code, _)| code)
}

#[inline]
fn previous_primary_home_bleeds_into_today(
    user_idx: usize,
    columns: &MotifColumns<'_>,
    row: usize,
    primary_home: NodeCode,
) -> bool {
    columns.node(user_idx, row) == primary_home
        && (columns.end_hour(row) < 3 || columns.end_hour(row) == 23)
}

#[allow(clippy::too_many_arguments)]
fn compute_day(
    user_idx: usize,
    columns: &MotifColumns<'_>,
    indices: Option<&[usize]>,
    day_start: usize,
    day_end: usize,
    primary_home: NodeCode,
    last_night_node: Option<NodeCode>,
    next_day_first_node: Option<NodeCode>,
    scratch: &mut Scratch,
) -> DailyMotifResult {
    let first_row = ordered_row(indices, day_start);
    let date_id = columns.date_id(first_row);
    scratch.cleaned.clear();
    scratch
        .cleaned
        .push(last_night_node.unwrap_or(primary_home));
    for position in day_start..day_end {
        let code = columns.node(user_idx, ordered_row(indices, position));
        if scratch.cleaned.last().copied() != Some(code) {
            scratch.cleaned.push(code);
        }
    }

    let last_code = *scratch.cleaned.last().expect("sequence is never empty");
    if !is_home_code(last_code, columns.home_purpose_code) {
        let closure = next_day_first_node.unwrap_or(primary_home);
        if scratch.cleaned.last().copied() != Some(closure) {
            scratch.cleaned.push(closure);
        }
    }

    let mut nodes = [NodeCode::default(); 6];
    let mut n_nodes = 1usize;
    nodes[0] = primary_home;
    for &code in &scratch.cleaned {
        if !nodes[..n_nodes].contains(&code) {
            if n_nodes == 6 {
                return DailyMotifResult {
                    user_idx,
                    date_id,
                    motif_id: -1,
                };
            }
            nodes[n_nodes] = code;
            n_nodes += 1;
        }
    }

    let mut adj = 0u64;
    for window in scratch.cleaned.windows(2) {
        let i = nodes[..n_nodes]
            .iter()
            .position(|&node| node == window[0])
            .expect("cleaned node must be registered");
        let j = nodes[..n_nodes]
            .iter()
            .position(|&node| node == window[1])
            .expect("cleaned node must be registered");
        adj |= cell_bit(n_nodes, i, j);
    }

    let key = (n_nodes as u8, adj);
    let motif_id = *scratch
        .motif_cache
        .entry(key)
        .or_insert_with(|| canonical_adjacency_mask(n_nodes, adj));
    DailyMotifResult {
        user_idx,
        date_id,
        motif_id,
    }
}

fn process_user(
    user_idx: usize,
    columns: &MotifColumns<'_>,
    indices: Option<&[usize]>,
    start: usize,
    end: usize,
    scratch: &mut Scratch,
) -> Vec<DailyMotifResult> {
    let Some(primary_home) = compute_primary_home(user_idx, columns, indices, start, end) else {
        return Vec::new();
    };

    let mut results = Vec::new();
    let mut day_start = start;
    while day_start < end {
        let date_id = columns.date_id(ordered_row(indices, day_start));
        let mut day_end = day_start + 1;
        while day_end < end && columns.date_id(ordered_row(indices, day_end)) == date_id {
            day_end += 1;
        }

        let last_night_node = if day_start > start {
            let row = ordered_row(indices, day_start - 1);
            previous_primary_home_bleeds_into_today(user_idx, columns, row, primary_home)
                .then(|| columns.node(user_idx, row))
        } else {
            None
        };
        let next_day_first_node = if day_end < end {
            let row = ordered_row(indices, day_end);
            (columns.start_hour(row) < 3 && columns.is_home(user_idx, row))
                .then(|| columns.node(user_idx, row))
        } else {
            None
        };
        results.push(compute_day(
            user_idx,
            columns,
            indices,
            day_start,
            day_end,
            primary_home,
            last_night_node,
            next_day_first_node,
            scratch,
        ));
        day_start = day_end;
    }
    results
}

fn validate_columns(columns: &MotifColumns<'_>) -> Result<(), String> {
    let n = columns.location_codes.len();
    let purpose_len_ok = match &columns.purpose_source {
        PurposeSource::Flat(codes) => codes.len() == n,
        // A lookup table's size has no length relationship to `n` -- nothing to check.
        PurposeSource::Lookup { .. } => true,
    };
    if !purpose_len_ok
        || columns.start_timestamps_us.len() != n
        || columns.end_timestamps_us.len() != n
    {
        return Err("All input column vectors must have the same length".to_string());
    }
    if columns.durations.is_some_and(|values| values.len() != n) {
        return Err("All input column vectors must have the same length".to_string());
    }
    Ok(())
}

fn compute_daily_motifs_impl(
    columns: MotifColumns<'_>,
    indices: Option<&[usize]>,
    ends: &[usize],
) -> DailyMotifsResult {
    validate_columns(&columns)?;
    let n = columns.location_codes.len();
    if let Some(indices) = indices {
        validate_indexed_ends(n, indices, ends)?;
    } else {
        let ranges = ranges_from_ends(ends)?;
        if ranges.last().is_some_and(|&(_, end)| end > n) {
            return Err("range end must be within input array bounds".to_string());
        }
    }
    let ranges = ranges_from_ends(ends)?;
    let started = Instant::now();
    let grouped: Vec<Vec<DailyMotifResult>> = ranges
        .par_iter()
        .enumerate()
        .map_init(Scratch::default, |scratch, (user_idx, &(start, end))| {
            process_user(user_idx, &columns, indices, start, end, scratch)
        })
        .collect();

    let total_len = grouped.iter().map(Vec::len).sum();
    let mut out_users = Vec::with_capacity(total_len);
    let mut out_dates = Vec::with_capacity(total_len);
    let mut out_motifs = Vec::with_capacity(total_len);
    for group in grouped {
        for result in group {
            out_users.push(result.user_idx);
            out_dates.push(result.date_id);
            out_motifs.push(result.motif_id);
        }
    }
    if std::env::var_os("FASTMOB_PROFILE_MOTIFS").is_some() {
        eprintln!(
            "fastmob motif rust: users={} outputs={} process_and_output={:.6}s",
            ends.len(),
            total_len,
            started.elapsed().as_secs_f64()
        );
    }
    Ok((out_users, out_dates, out_motifs))
}

#[allow(clippy::too_many_arguments)]
pub fn compute_daily_motifs_indexed(
    location_codes: &[u32],
    purpose_codes: &[u16],
    start_timestamps_us: &[i64],
    end_timestamps_us: &[i64],
    durations: Option<&[f64]>,
    indices: &[usize],
    ends: &[usize],
    home_purpose_code: u16,
) -> DailyMotifsResult {
    compute_daily_motifs_impl(
        MotifColumns {
            location_codes,
            purpose_source: PurposeSource::Flat(purpose_codes),
            start_timestamps_us,
            end_timestamps_us,
            durations,
            home_purpose_code,
        },
        Some(indices),
        ends,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn compute_daily_motifs_presorted(
    location_codes: &[u32],
    purpose_codes: &[u16],
    start_timestamps_us: &[i64],
    end_timestamps_us: &[i64],
    durations: Option<&[f64]>,
    ends: &[usize],
    home_purpose_code: u16,
) -> DailyMotifsResult {
    compute_daily_motifs_impl(
        MotifColumns {
            location_codes,
            purpose_source: PurposeSource::Flat(purpose_codes),
            start_timestamps_us,
            end_timestamps_us,
            durations,
            home_purpose_code,
        },
        None,
        ends,
    )
}

/// Rust-level-join sibling of [`compute_daily_motifs_indexed`]: instead of a
/// flat, pre-joined `purpose_codes` array, takes a small
/// `(user_idx, location_code) -> purpose_code` lookup built from a separate
/// Locations-grain table, resolving each row's purpose lazily during the
/// per-user scan. A lookup miss (e.g. a staypoint whose location fell below
/// a clustering threshold) resolves to `unmatched_purpose_code`, matching
/// the same never-error null-purpose convention as the flat path.
#[allow(clippy::too_many_arguments)]
pub fn compute_daily_motifs_indexed_joined(
    location_codes: &[u32],
    start_timestamps_us: &[i64],
    end_timestamps_us: &[i64],
    durations: Option<&[f64]>,
    indices: &[usize],
    ends: &[usize],
    home_purpose_code: u16,
    lookup: &FxHashMap<(u32, u32), u16>,
    unmatched_purpose_code: u16,
) -> DailyMotifsResult {
    compute_daily_motifs_impl(
        MotifColumns {
            location_codes,
            purpose_source: PurposeSource::Lookup {
                table: lookup,
                unmatched_code: unmatched_purpose_code,
            },
            start_timestamps_us,
            end_timestamps_us,
            durations,
            home_purpose_code,
        },
        Some(indices),
        ends,
    )
}

/// Rust-level-join sibling of [`compute_daily_motifs_presorted`]. See
/// [`compute_daily_motifs_indexed_joined`].
#[allow(clippy::too_many_arguments)]
pub fn compute_daily_motifs_presorted_joined(
    location_codes: &[u32],
    start_timestamps_us: &[i64],
    end_timestamps_us: &[i64],
    durations: Option<&[f64]>,
    ends: &[usize],
    home_purpose_code: u16,
    lookup: &FxHashMap<(u32, u32), u16>,
    unmatched_purpose_code: u16,
) -> DailyMotifsResult {
    compute_daily_motifs_impl(
        MotifColumns {
            location_codes,
            purpose_source: PurposeSource::Lookup {
                table: lookup,
                unmatched_code: unmatched_purpose_code,
            },
            start_timestamps_us,
            end_timestamps_us,
            durations,
            home_purpose_code,
        },
        None,
        ends,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_columns<'a>(
        location_codes: &'a [u32],
        purpose_codes: &'a [u16],
        start: &'a [i64],
        end: &'a [i64],
    ) -> MotifColumns<'a> {
        MotifColumns {
            location_codes,
            purpose_source: PurposeSource::Flat(purpose_codes),
            start_timestamps_us: start,
            end_timestamps_us: end,
            durations: None,
            home_purpose_code: 0,
        }
    }

    #[test]
    fn canonicalization_keeps_node_zero_anchored() {
        let left = canonical_adjacency_form_internal(3, &[(0, 1), (1, 2)]).unwrap();
        let right = canonical_adjacency_form_internal(3, &[(1, 0), (0, 2)]).unwrap();
        assert_ne!(left, right);
    }

    #[test]
    fn previous_primary_home_bleeds_only_for_small_hours_or_23() {
        let location_codes = [7];
        let purpose_codes = [0];
        let start = [0];
        for hour in [0, 2, 23] {
            let end = [i64::from(hour) * MICROS_PER_HOUR];
            assert!(previous_primary_home_bleeds_into_today(
                0,
                &test_columns(&location_codes, &purpose_codes, &start, &end),
                0,
                NodeCode {
                    location: 7,
                    purpose: 0
                }
            ));
        }
        for hour in [3, 4] {
            let end = [i64::from(hour) * MICROS_PER_HOUR];
            assert!(!previous_primary_home_bleeds_into_today(
                0,
                &test_columns(&location_codes, &purpose_codes, &start, &end),
                0,
                NodeCode {
                    location: 7,
                    purpose: 0
                }
            ));
        }
    }

    #[test]
    fn flat_and_lookup_purpose_sources_produce_identical_results() {
        // 2 users, 4 rows each: home overnight, work daytime, repeated for 2 days.
        let location_codes: Vec<u32> = vec![
            10, 11, 10, 11, // user 0: home, work, home, work
            20, 21, 20, 21, // user 1: home, work, home, work
        ];
        let purpose_codes: Vec<u16> = vec![0, 1, 0, 1, 0, 1, 0, 1]; // 0 = HOME, 1 = WORK
        let day0 = 0i64;
        let day1 = MICROS_PER_DAY;
        let start_timestamps_us: Vec<i64> = vec![
            day0,
            day0 + 9 * MICROS_PER_HOUR,
            day0 + 18 * MICROS_PER_HOUR,
            day1 + 9 * MICROS_PER_HOUR,
            day0,
            day0 + 9 * MICROS_PER_HOUR,
            day0 + 18 * MICROS_PER_HOUR,
            day1 + 9 * MICROS_PER_HOUR,
        ];
        let end_timestamps_us: Vec<i64> = start_timestamps_us
            .iter()
            .map(|&t| t + 4 * MICROS_PER_HOUR)
            .collect();
        let indices: Vec<usize> = (0..8).collect();
        let ends: Vec<usize> = vec![4, 8];
        let home_purpose_code = 0u16;

        let flat_result = compute_daily_motifs_indexed(
            &location_codes,
            &purpose_codes,
            &start_timestamps_us,
            &end_timestamps_us,
            None,
            &indices,
            &ends,
            home_purpose_code,
        )
        .unwrap();

        let mut lookup: FxHashMap<(u32, u32), u16> = FxHashMap::default();
        lookup.insert((0, 10), 0);
        lookup.insert((0, 11), 1);
        lookup.insert((1, 20), 0);
        lookup.insert((1, 21), 1);
        let unmatched_purpose_code = 2u16;

        let lookup_result = compute_daily_motifs_indexed_joined(
            &location_codes,
            &start_timestamps_us,
            &end_timestamps_us,
            None,
            &indices,
            &ends,
            home_purpose_code,
            &lookup,
            unmatched_purpose_code,
        )
        .unwrap();

        assert_eq!(flat_result, lookup_result);
    }
}
