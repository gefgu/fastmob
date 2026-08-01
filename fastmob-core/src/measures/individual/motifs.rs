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
    location: u64,
    purpose: u64,
}

#[derive(Default)]
struct Scratch {
    cleaned: Vec<NodeCode>,
    motif_cache: FxHashMap<(u8, u64), i64>,
}

struct MotifColumns<'a> {
    location_codes: &'a [u64],
    purpose_codes: &'a [u64],
    start_timestamps_us: &'a [i64],
    end_timestamps_us: &'a [i64],
    durations: Option<&'a [f64]>,
    home_purpose_code: u64,
}

impl MotifColumns<'_> {
    #[inline]
    fn node(&self, row: usize) -> NodeCode {
        NodeCode {
            location: self.location_codes[row],
            purpose: self.purpose_codes[row],
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
    fn is_home(&self, row: usize) -> bool {
        self.purpose_codes[row] == self.home_purpose_code
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
fn is_home_code(code: NodeCode, home_purpose_code: u64) -> bool {
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
    columns: &MotifColumns<'_>,
    indices: Option<&[usize]>,
    start: usize,
    end: usize,
) -> Option<NodeCode> {
    let mut night_duration: FxHashMap<NodeCode, f64> = FxHashMap::default();
    for position in start..end {
        let row = ordered_row(indices, position);
        if columns.is_home(row) && (columns.start_hour(row) >= 22 || columns.start_hour(row) < 6) {
            *night_duration.entry(columns.node(row)).or_insert(0.0) += columns.duration(row);
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
        if columns.is_home(row) {
            *home_counts.entry(columns.node(row)).or_insert(0) += 1;
        }
    }
    home_counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(code, _)| code)
}

#[inline]
fn previous_primary_home_bleeds_into_today(
    columns: &MotifColumns<'_>,
    row: usize,
    primary_home: NodeCode,
) -> bool {
    columns.node(row) == primary_home && (columns.end_hour(row) < 3 || columns.end_hour(row) == 23)
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
        let code = columns.node(ordered_row(indices, position));
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
    let Some(primary_home) = compute_primary_home(columns, indices, start, end) else {
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
            previous_primary_home_bleeds_into_today(columns, row, primary_home)
                .then(|| columns.node(row))
        } else {
            None
        };
        let next_day_first_node = if day_end < end {
            let row = ordered_row(indices, day_end);
            (columns.start_hour(row) < 3 && columns.is_home(row)).then(|| columns.node(row))
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
    if columns.purpose_codes.len() != n
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
    location_codes: &[u64],
    purpose_codes: &[u64],
    start_timestamps_us: &[i64],
    end_timestamps_us: &[i64],
    durations: Option<&[f64]>,
    indices: &[usize],
    ends: &[usize],
    home_purpose_code: u64,
) -> DailyMotifsResult {
    compute_daily_motifs_impl(
        MotifColumns {
            location_codes,
            purpose_codes,
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
    location_codes: &[u64],
    purpose_codes: &[u64],
    start_timestamps_us: &[i64],
    end_timestamps_us: &[i64],
    durations: Option<&[f64]>,
    ends: &[usize],
    home_purpose_code: u64,
) -> DailyMotifsResult {
    compute_daily_motifs_impl(
        MotifColumns {
            location_codes,
            purpose_codes,
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
        location_codes: &'a [u64],
        purpose_codes: &'a [u64],
        start: &'a [i64],
        end: &'a [i64],
    ) -> MotifColumns<'a> {
        MotifColumns {
            location_codes,
            purpose_codes,
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
                &test_columns(&location_codes, &purpose_codes, &start, &end),
                0,
                NodeCode {
                    location: 7,
                    purpose: 0
                }
            ));
        }
    }
}
