use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::models::shared::{cdf_choice, derive_agent_seed};

const SECONDS_PER_DAY: i64 = 86_400;
const NS_PER_SECOND: i64 = 1_000_000_000;
/// Default temporal granularity: 24 hourly slots per day (the historical behaviour).
pub const DEFAULT_SLOTS_PER_DAY: usize = 24;

// State encoding: state_idx = slot * 2 + typicality
// typicality: 0 = non-typical (other), 1 = typical (home)
// There are `slots_per_day` slots of the day and 2 typicality values, so the
// Markov chain has `n_states = slots_per_day * 2` states and an
// `n_states * n_states` transition matrix.
#[inline(always)]
fn state_idx(slot: usize, typicality: usize) -> usize {
    slot * 2 + typicality
}

#[inline(always)]
fn n_states_for(slots_per_day: usize) -> usize {
    slots_per_day * 2
}

#[inline(always)]
fn slot_seconds_for(slots_per_day: usize) -> i64 {
    SECONDS_PER_DAY / slots_per_day as i64
}

/// Accumulate raw transition counts from one user's pre-processed slot value sequence.
///
/// `values`: per-slot location rank sequence (rank 1 = home).
/// `shift`: starting slot (0..slots_per_day) of the first entry.
/// `counts`: mutable flat [n_states*n_states] accumulator (caller zeroes before first user).
/// `slots_per_day`: number of time slots per day (24 = hourly, 96 = 15-min, 288 = 5-min).
///
/// Implements the same tau-lookahead logic as Python `_update_markov_chain`.
pub fn markov_diary_update_chain_impl(
    values: &[usize],
    shift: usize,
    counts: &mut [f64],
    slots_per_day: usize,
) -> Result<(), String> {
    let n_states = n_states_for(slots_per_day);
    let n_matrix = n_states * n_states;
    if counts.len() < n_matrix {
        return Err(format!(
            "counts must have length {} but got {}",
            n_matrix,
            counts.len()
        ));
    }
    let home: usize = 1;
    let typical: usize = 1;
    let non_typical: usize = 0;
    let n = values.len();
    let mut slot: usize = 0;

    while slot + 1 < n {
        let h = (slot + shift) % slots_per_day;
        let next_h = (h + 1) % slots_per_day;
        let loc_h = values[slot];
        let next_loc_h = values[slot + 1];

        if loc_h == home {
            if next_loc_h == home {
                // home → home
                counts[state_idx(h, typical) * n_states + state_idx(next_h, typical)] += 1.0;
                slot += 1;
            } else if slot + 2 < n {
                // home → other: scan tau
                let mut tau = 1usize;
                let mut j = slot + 2;
                while j < n {
                    if values[j] == next_loc_h {
                        tau += 1;
                        j += 1;
                    } else {
                        break;
                    }
                }
                let next_state_h = (h + tau) % slots_per_day;
                counts[state_idx(h, typical) * n_states + state_idx(next_state_h, non_typical)] +=
                    1.0;
                slot = j.saturating_sub(2);
                slot += 1;
            } else {
                break;
            }
        } else {
            // currently at non-home
            if next_loc_h == home {
                // other → home
                counts[state_idx(h, non_typical) * n_states + state_idx(next_h, typical)] += 1.0;
                slot += 1;
            } else if slot + 2 < n {
                // other → same other: scan tau
                let mut tau = 1usize;
                let mut j = slot + 2;
                while j < n {
                    if values[j] == next_loc_h {
                        tau += 1;
                        j += 1;
                    } else {
                        break;
                    }
                }
                let next_state_h = (h + tau) % slots_per_day;
                counts[state_idx(h, non_typical) * n_states
                    + state_idx(next_state_h, non_typical)] += 1.0;
                slot = j.saturating_sub(2);
                slot += 1;
            } else {
                break;
            }
        }
    }
    Ok(())
}

/// Normalise counts in-place: each row sums to 1 (all-zero rows stay zero).
pub fn markov_diary_normalize_impl(counts: &mut [f64], n_states: usize) {
    for row in counts.chunks_mut(n_states) {
        let total: f64 = row.iter().sum();
        if total > 0.0 {
            for v in row.iter_mut() {
                *v /= total;
            }
        }
    }
}

/// Convert normalized probability matrix to CDF matrix (each row is cumulative).
pub fn markov_diary_build_cdf_impl(probs: &[f64], n_states: usize) -> Vec<f64> {
    let mut cdf = probs.to_vec();
    for row in cdf.chunks_mut(n_states) {
        let mut cumsum = 0.0;
        for v in row.iter_mut() {
            cumsum += *v;
            *v = cumsum;
        }
    }
    cdf
}

fn validate_fit_inputs(
    uids: &[i64],
    timestamps_ns: &[i64],
    loc_codes: &[i64],
) -> Result<(), String> {
    if uids.len() != timestamps_ns.len() || uids.len() != loc_codes.len() {
        return Err("uids, timestamps_ns, and loc_codes must have the same length".to_string());
    }
    if uids.iter().any(|&uid| uid < 0) {
        return Err("uids must contain non-negative factorized codes".to_string());
    }
    if loc_codes.iter().any(|&loc| loc < 0) {
        return Err("loc_codes must contain non-negative factorized codes".to_string());
    }
    Ok(())
}

fn floor_slot(ns: i64, slot_ns: i64) -> i64 {
    ns.div_euclid(slot_ns)
}

fn location_ranks(
    rows: &[usize],
    loc_codes: &[i64],
) -> (
    FxHashMap<i64, usize>,
    FxHashMap<i64, usize>,
    FxHashMap<i64, usize>,
) {
    let mut freq: FxHashMap<i64, usize> = FxHashMap::default();
    let mut first_seen: FxHashMap<i64, usize> = FxHashMap::default();
    for (pos, &row) in rows.iter().enumerate() {
        let loc = loc_codes[row];
        *freq.entry(loc).or_insert(0) += 1;
        first_seen.entry(loc).or_insert(pos);
    }

    let mut locs: Vec<i64> = freq.keys().copied().collect();
    locs.sort_by(|a, b| {
        freq[b]
            .cmp(&freq[a])
            .then_with(|| first_seen[a].cmp(&first_seen[b]))
            .then_with(|| a.cmp(b))
    });

    let ranks = locs
        .into_iter()
        .enumerate()
        .map(|(idx, loc)| (loc, idx + 1))
        .collect();
    (ranks, freq, first_seen)
}

fn best_locations_by_slot(
    rows: &[usize],
    timestamps_ns: &[i64],
    loc_codes: &[i64],
    freq: &FxHashMap<i64, usize>,
    first_seen: &FxHashMap<i64, usize>,
    slot_ns: i64,
) -> FxHashMap<i64, i64> {
    let mut counts_by_slot: FxHashMap<i64, FxHashMap<i64, usize>> = FxHashMap::default();
    for &row in rows {
        let slot = floor_slot(timestamps_ns[row], slot_ns);
        let loc = loc_codes[row];
        let loc_counts = counts_by_slot.entry(slot).or_default();
        *loc_counts.entry(loc).or_insert(0) += 1;
    }

    let mut best_by_slot: FxHashMap<i64, i64> = FxHashMap::default();
    for (slot, loc_counts) in counts_by_slot {
        let mut best_loc: Option<i64> = None;
        let mut best_count = 0usize;
        let mut best_freq = 0usize;
        let mut best_first_seen = usize::MAX;
        for (loc, count) in loc_counts {
            let global_freq = freq[&loc];
            let first = first_seen[&loc];
            let better = count > best_count
                || (count == best_count
                    && (global_freq > best_freq
                        || (global_freq == best_freq
                            && (first < best_first_seen
                                || (first == best_first_seen
                                    && best_loc.is_none_or(|current| loc < current))))));
            if better {
                best_loc = Some(loc);
                best_count = count;
                best_freq = global_freq;
                best_first_seen = first;
            }
        }
        if let Some(loc) = best_loc {
            best_by_slot.insert(slot, loc);
        }
    }
    best_by_slot
}

fn create_time_series_for_rows(
    timestamps_ns: &[i64],
    loc_codes: &[i64],
    rows: &[usize],
    slots_per_day: usize,
    slot_ns: i64,
) -> Result<Option<(Vec<usize>, usize)>, String> {
    if rows.is_empty() {
        return Ok(None);
    }

    let min_slot = rows
        .iter()
        .map(|&row| floor_slot(timestamps_ns[row], slot_ns))
        .min()
        .ok_or_else(|| "cannot create a time series for an empty user".to_string())?;
    let max_slot = rows
        .iter()
        .map(|&row| floor_slot(timestamps_ns[row], slot_ns))
        .max()
        .ok_or_else(|| "cannot create a time series for an empty user".to_string())?;
    let len_i64 = max_slot - min_slot + 1;
    let len = usize::try_from(len_i64).map_err(|_| "time series is too long".to_string())?;
    let shift = min_slot.rem_euclid(slots_per_day as i64) as usize;

    let (ranks, freq, first_seen) = location_ranks(rows, loc_codes);
    let best_by_slot =
        best_locations_by_slot(rows, timestamps_ns, loc_codes, &freq, &first_seen, slot_ns);
    let mut slot_locs: Vec<Option<i64>> = vec![None; len];
    for (slot, loc) in best_by_slot {
        let idx = usize::try_from(slot - min_slot)
            .map_err(|_| "slot index is out of bounds".to_string())?;
        slot_locs[idx] = Some(loc);
    }

    let mut last = None;
    for loc in &mut slot_locs {
        if loc.is_some() {
            last = *loc;
        } else {
            *loc = last;
        }
    }
    if slot_locs.first().is_some_and(Option::is_none)
        && let Some(first_loc) = slot_locs.iter().copied().flatten().next()
    {
        for loc in slot_locs.iter_mut().take_while(|loc| loc.is_none()) {
            *loc = Some(first_loc);
        }
    }

    let values = slot_locs
        .into_iter()
        .map(|loc| loc.and_then(|code| ranks.get(&code).copied()).unwrap_or(0))
        .collect();
    Ok(Some((values, shift)))
}

pub fn markov_diary_fit_from_arrays_impl(
    uids: &[i64],
    timestamps_ns: &[i64],
    loc_codes: &[i64],
    n_individuals: usize,
    slots_per_day: usize,
) -> Result<Vec<f64>, String> {
    validate_fit_inputs(uids, timestamps_ns, loc_codes)?;
    let n_states = n_states_for(slots_per_day);
    let n_matrix = n_states * n_states;
    let slot_ns = slot_seconds_for(slots_per_day) * NS_PER_SECOND;
    if n_individuals == 0 || uids.is_empty() {
        return Ok(markov_diary_build_cdf_impl(&vec![0.0; n_matrix], n_states));
    }

    let selected_users = uids
        .iter()
        .copied()
        .filter(|&uid| uid >= 0)
        .max()
        .map(|max_uid| n_individuals.min(max_uid as usize + 1))
        .unwrap_or(0);
    if selected_users == 0 {
        return Ok(markov_diary_build_cdf_impl(&vec![0.0; n_matrix], n_states));
    }

    let mut rows_by_user: Vec<Vec<usize>> = (0..selected_users).map(|_| Vec::new()).collect();
    for (row, &uid) in uids.iter().enumerate() {
        let user = uid as usize;
        if user < selected_users {
            rows_by_user[user].push(row);
        }
    }

    let counts = rows_by_user
        .par_iter()
        .map(|rows| -> Result<Vec<f64>, String> {
            let mut local_counts = vec![0.0_f64; n_matrix];
            if let Some((values, shift)) =
                create_time_series_for_rows(timestamps_ns, loc_codes, rows, slots_per_day, slot_ns)?
            {
                markov_diary_update_chain_impl(&values, shift, &mut local_counts, slots_per_day)?;
            }
            Ok(local_counts)
        })
        .try_reduce(
            || vec![0.0_f64; n_matrix],
            |mut acc, local| {
                for (target, value) in acc.iter_mut().zip(local) {
                    *target += value;
                }
                Ok(acc)
            },
        )?;

    let mut probs = counts;
    markov_diary_normalize_impl(&mut probs, n_states);
    Ok(markov_diary_build_cdf_impl(&probs, n_states))
}

/// Generate one diary from the CDF matrix.
/// Returns parallel arrays: (unix_timestamps_seconds, abstract_locations).
///
/// Mirrors `_generate_list` logic:
/// - Start state: (slot=0, typicality=home=1)
/// - Abstract location: 0 = home, 1,2,3,… = other locations in encounter order
/// - The returned diary is "short" (consecutive identical locations are deduplicated)
///
/// `diary_length` is the number of time slots to generate (not hours).
/// `slots_per_day` / `slot_seconds` define the temporal granularity.
pub fn markov_diary_generate_impl(
    cdf_matrix: &[f64],
    diary_length: usize,
    start_ts: i64,
    seed: u64,
    slots_per_day: usize,
    slot_seconds: i64,
) -> (Vec<i64>, Vec<i32>) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let n_states = n_states_for(slots_per_day);

    // --- generate state sequence V ---
    let mut v: Vec<(usize, usize)> = Vec::with_capacity(diary_length / 4 + 4);
    let prev_state = (0usize, 1usize); // slot=0, home=typical
    v.push(prev_state);
    let mut prev_state = prev_state;
    let mut i: usize = 0;

    while i < diary_length {
        let h = i % slots_per_day;
        let row_start = state_idx(prev_state.0, prev_state.1) * n_states;
        let row = &cdf_matrix[row_start..row_start + n_states];
        let row_total = row[n_states - 1];

        let next_state_idx = if row_total <= 0.0 || !row_total.is_finite() {
            // fallback: advance to next slot, same typicality
            state_idx((prev_state.0 + 1) % slots_per_day, prev_state.1)
        } else {
            cdf_choice(&mut rng, row)
        };

        let next_h = next_state_idx / 2;
        let next_r = next_state_idx % 2;
        let next_state = (next_h, next_r);
        v.push(next_state);

        let j = next_state.0;
        let step = if j > h { j - h } else { slots_per_day - h + j };
        i += step;
        prev_state = next_state;
    }

    // --- build full per-slot diary ---
    // diary entries: (timestamp_s, abstract_location)
    // abstract_location 0 = home, 1+ = other visit orders
    let mut current_ts = start_ts;
    let mut diary: Vec<(i64, i32)> = Vec::with_capacity(diary_length + 4);
    diary.push((current_ts, 0i32));
    let mut other_count: i32 = 1;

    for win in v.windows(2) {
        let prev = win[0];
        let next = win[1];
        let (h_prev, _) = prev;
        let (h_next, s_next) = next;
        if s_next == 1 {
            // home
            current_ts += slot_seconds;
            diary.push((current_ts, 0));
            other_count = 1;
        } else {
            let steps = if h_next > h_prev {
                h_next - h_prev
            } else {
                slots_per_day - h_prev + h_next
            };
            for _ in 0..steps {
                current_ts += slot_seconds;
                diary.push((current_ts, other_count));
            }
            other_count += 1;
        }
    }

    // --- build short diary (deduplicate consecutive same abstract_location) ---
    let mut timestamps: Vec<i64> = Vec::with_capacity(diary_length);
    let mut abstract_locs: Vec<i32> = Vec::with_capacity(diary_length);
    let mut prev_loc: i32 = -1;
    for (ts, loc) in diary.into_iter().take(diary_length) {
        if loc != prev_loc {
            timestamps.push(ts);
            abstract_locs.push(loc);
            prev_loc = loc;
        }
    }

    (timestamps, abstract_locs)
}

/// Build a home-only CDF matrix: agents always remain at home (abstract_location=0).
///
/// Encodes a Markov chain that transitions slot-by-slot, always staying in the
/// home/typical state (typicality=1). Used as a fallback when no diary has been fitted.
fn markov_diary_home_only_cdf_impl(slots_per_day: usize) -> Vec<f64> {
    let n_states = n_states_for(slots_per_day);
    let mut probs = vec![0.0_f64; n_states * n_states];
    for h in 0..slots_per_day {
        let next_h = (h + 1) % slots_per_day;
        // State h*2+1 = home at slot h  →  state next_h*2+1 = home at slot next_h
        probs[state_idx(h, 1) * n_states + state_idx(next_h, 1)] = 1.0;
    }
    markov_diary_build_cdf_impl(&probs, n_states)
}

/// Batch-generate diaries for `n_agents` in parallel (independent per agent).
/// Returns flat arrays + per-agent slice boundaries.
///
/// When `cdf_matrix` is `None`, a home-only fallback CDF is used so that every
/// diary slot carries `abstract_location = 0` (home).
pub fn markov_diary_batch_generate_impl(
    cdf_matrix: Option<&[f64]>,
    diary_length: usize,
    start_ts: i64,
    n_agents: usize,
    master_seed: u64,
    slots_per_day: usize,
    slot_seconds: i64,
) -> (Vec<i64>, Vec<i32>, Vec<usize>, Vec<usize>) {
    let owned_cdf: Vec<f64>;
    let cdf: &[f64] = match cdf_matrix {
        Some(m) => m,
        None => {
            owned_cdf = markov_diary_home_only_cdf_impl(slots_per_day);
            &owned_cdf
        }
    };

    let results: Vec<(Vec<i64>, Vec<i32>)> = (0..n_agents)
        .into_par_iter()
        .map(|agent| {
            let seed = derive_agent_seed(master_seed, agent, 2);
            markov_diary_generate_impl(
                cdf,
                diary_length,
                start_ts,
                seed,
                slots_per_day,
                slot_seconds,
            )
        })
        .collect();

    let mut flat_ts: Vec<i64> = Vec::new();
    let mut flat_locs: Vec<i32> = Vec::new();
    let mut starts: Vec<usize> = Vec::with_capacity(n_agents);
    let mut ends: Vec<usize> = Vec::with_capacity(n_agents);

    for (ts, locs) in results {
        starts.push(flat_ts.len());
        flat_ts.extend_from_slice(&ts);
        flat_locs.extend_from_slice(&locs);
        ends.push(flat_ts.len());
    }

    (flat_ts, flat_locs, starts, ends)
}

#[cfg(test)]
mod tests {
    use super::*;

    const HOUR_NS: i64 = 3_600_000_000_000;
    const SLOTS_PER_DAY: usize = DEFAULT_SLOTS_PER_DAY;

    fn ns(hours: i64) -> i64 {
        hours * HOUR_NS
    }

    fn slot_ns() -> i64 {
        slot_seconds_for(SLOTS_PER_DAY) * NS_PER_SECOND
    }

    #[test]
    fn create_time_series_ranks_locations_by_frequency_then_first_seen() {
        let timestamps = vec![ns(0), ns(1), ns(2), ns(3)];
        let locs = vec![7, 9, 9, 7];
        let rows = vec![0, 1, 2, 3];

        let (values, shift) =
            create_time_series_for_rows(&timestamps, &locs, &rows, SLOTS_PER_DAY, slot_ns())
                .unwrap()
                .unwrap();

        assert_eq!(shift, 0);
        assert_eq!(values, vec![1, 2, 2, 1]);
    }

    #[test]
    fn create_time_series_forward_fills_missing_slot_bins() {
        let timestamps = vec![ns(0), ns(2)];
        let locs = vec![3, 4];
        let rows = vec![0, 1];

        let (values, shift) =
            create_time_series_for_rows(&timestamps, &locs, &rows, SLOTS_PER_DAY, slot_ns())
                .unwrap()
                .unwrap();

        assert_eq!(shift, 0);
        assert_eq!(values, vec![1, 1, 2]);
    }

    #[test]
    fn create_time_series_chooses_bin_location_by_count_then_global_frequency() {
        let timestamps = vec![ns(0), ns(0), ns(1)];
        let locs = vec![2, 1, 1];
        let rows = vec![0, 1, 2];

        let (values, shift) =
            create_time_series_for_rows(&timestamps, &locs, &rows, SLOTS_PER_DAY, slot_ns())
                .unwrap()
                .unwrap();

        assert_eq!(shift, 0);
        assert_eq!(values, vec![1, 1]);
    }

    #[test]
    fn create_time_series_uses_first_seen_as_final_bin_tiebreaker() {
        let timestamps = vec![ns(0), ns(0)];
        let locs = vec![20, 10];
        let rows = vec![0, 1];

        let (values, shift) =
            create_time_series_for_rows(&timestamps, &locs, &rows, SLOTS_PER_DAY, slot_ns())
                .unwrap()
                .unwrap();

        assert_eq!(shift, 0);
        assert_eq!(values, vec![1]);
    }

    #[test]
    fn fit_from_arrays_limits_users_by_first_seen_factorized_order() {
        let uids = vec![0, 0, 1, 1];
        let timestamps = vec![ns(0), ns(1), ns(0), ns(1)];
        let locs = vec![1, 1, 1, 2];

        let cdf =
            markov_diary_fit_from_arrays_impl(&uids, &timestamps, &locs, 1, SLOTS_PER_DAY).unwrap();
        let n_states = n_states_for(SLOTS_PER_DAY);
        let row_start = state_idx(0, 1) * n_states;
        let home_to_hour_one_home = row_start + state_idx(1, 1);
        let home_to_hour_one_away = row_start + state_idx(1, 0);

        assert_eq!(cdf[home_to_hour_one_away], 0.0);
        assert_eq!(cdf[home_to_hour_one_home], 1.0);
    }

    #[test]
    fn fifteen_minute_granularity_emits_quarter_hour_timestamps() {
        let slots_per_day = 96usize; // 15-minute slots
        let slot_seconds = slot_seconds_for(slots_per_day);
        assert_eq!(slot_seconds, 900);
        let (ts, locs, starts, ends) = markov_diary_batch_generate_impl(
            None,
            slots_per_day,
            0,
            1,
            7,
            slots_per_day,
            slot_seconds,
        );
        // Home-only fallback collapses to a single home record.
        assert_eq!(starts, vec![0]);
        assert_eq!(ends, vec![1]);
        assert_eq!(ts, vec![0]);
        assert_eq!(locs, vec![0]);
    }
}
