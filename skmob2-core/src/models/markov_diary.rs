use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use crate::models::shared::{cdf_choice, derive_agent_seed};

const N_STATES: usize = 48; // 24 hours × 2 typicality values
const N_MATRIX: usize = N_STATES * N_STATES;

// State encoding: state_idx = hour * 2 + typicality
// typicality: 0 = non-typical (other), 1 = typical (home)
#[inline(always)]
fn state_idx(hour: usize, typicality: usize) -> usize {
    hour * 2 + typicality
}

/// Accumulate raw transition counts from one user's pre-processed hourly value sequence.
///
/// `values`: hourly location rank sequence (rank 1 = home).
/// `shift`: starting hour (0-23) of the first slot.
/// `counts`: mutable flat [48*48] accumulator (caller zeroes before first user).
///
/// Implements the same tau-lookahead logic as Python `_update_markov_chain`.
pub fn markov_diary_update_chain_impl(
    values: &[usize],
    shift: usize,
    counts: &mut [f64],
) -> Result<(), String> {
    if counts.len() < N_MATRIX {
        return Err(format!(
            "counts must have length {} but got {}",
            N_MATRIX,
            counts.len()
        ));
    }
    let home: usize = 1;
    let typical: usize = 1;
    let non_typical: usize = 0;
    let n = values.len();
    let mut slot: usize = 0;

    while slot + 1 < n {
        let h = (slot + shift) % 24;
        let next_h = (h + 1) % 24;
        let loc_h = values[slot];
        let next_loc_h = values[slot + 1];

        if loc_h == home {
            if next_loc_h == home {
                // home → home
                counts[state_idx(h, typical) * N_STATES + state_idx(next_h, typical)] += 1.0;
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
                let next_state_h = (h + tau) % 24;
                counts[state_idx(h, typical) * N_STATES
                    + state_idx(next_state_h, non_typical)] += 1.0;
                slot = j.saturating_sub(2);
                slot += 1;
            } else {
                break;
            }
        } else {
            // currently at non-home
            if next_loc_h == home {
                // other → home
                counts[state_idx(h, non_typical) * N_STATES + state_idx(next_h, typical)] += 1.0;
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
                let next_state_h = (h + tau) % 24;
                counts[state_idx(h, non_typical) * N_STATES
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
pub fn markov_diary_normalize_impl(counts: &mut [f64]) {
    for row in counts.chunks_mut(N_STATES) {
        let total: f64 = row.iter().sum();
        if total > 0.0 {
            for v in row.iter_mut() {
                *v /= total;
            }
        }
    }
}

/// Convert normalized probability matrix to CDF matrix (each row is cumulative).
pub fn markov_diary_build_cdf_impl(probs: &[f64]) -> Vec<f64> {
    let mut cdf = probs.to_vec();
    for row in cdf.chunks_mut(N_STATES) {
        let mut cumsum = 0.0;
        for v in row.iter_mut() {
            cumsum += *v;
            *v = cumsum;
        }
    }
    cdf
}

/// Generate one diary from the CDF matrix.
/// Returns parallel arrays: (unix_timestamps_seconds, abstract_locations).
///
/// Mirrors `_generate_list` logic:
/// - Start state: (hour=0, typicality=home=1)
/// - Abstract location: 0 = home, 1,2,3,… = other locations in encounter order
/// - The returned diary is "short" (consecutive identical locations are deduplicated)
pub fn markov_diary_generate_impl(
    cdf_matrix: &[f64],
    diary_length: usize,
    start_ts: i64,
    seed: u64,
) -> (Vec<i64>, Vec<i32>) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    // --- generate state sequence V ---
    let mut v: Vec<(usize, usize)> = Vec::with_capacity(diary_length / 4 + 4);
    let prev_state = (0usize, 1usize); // hour=0, home=typical
    v.push(prev_state);
    let mut prev_state = prev_state;
    let mut i: usize = 0;

    while i < diary_length {
        let h = i % 24;
        let row_start = state_idx(prev_state.0, prev_state.1) * N_STATES;
        let row = &cdf_matrix[row_start..row_start + N_STATES];
        let row_total = row[N_STATES - 1];

        let next_state_idx = if row_total <= 0.0 || !row_total.is_finite() {
            // fallback: advance to next hour, same typicality
            state_idx((prev_state.0 + 1) % 24, prev_state.1)
        } else {
            cdf_choice(&mut rng, row)
        };

        let next_h = next_state_idx / 2;
        let next_r = next_state_idx % 2;
        let next_state = (next_h, next_r);
        v.push(next_state);

        let j = next_state.0;
        let step = if j > h { j - h } else { 24 - h + j };
        i += step;
        prev_state = next_state;
    }

    // --- build full hourly diary ---
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
            current_ts += 3600;
            diary.push((current_ts, 0));
            other_count = 1;
        } else {
            let steps = if h_next > h_prev {
                h_next - h_prev
            } else {
                24 - h_prev + h_next
            };
            for _ in 0..steps {
                current_ts += 3600;
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

/// Batch-generate diaries for `n_agents` in parallel (independent per agent).
/// Returns flat arrays + per-agent slice boundaries.
pub fn markov_diary_batch_generate_impl(
    cdf_matrix: &[f64],
    diary_length: usize,
    start_ts: i64,
    n_agents: usize,
    master_seed: u64,
) -> (Vec<i64>, Vec<i32>, Vec<usize>, Vec<usize>) {
    let results: Vec<(Vec<i64>, Vec<i32>)> = (0..n_agents)
        .into_par_iter()
        .map(|agent| {
            let seed = derive_agent_seed(master_seed, agent, 2);
            markov_diary_generate_impl(cdf_matrix, diary_length, start_ts, seed)
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
