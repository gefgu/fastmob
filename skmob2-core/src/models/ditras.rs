use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use crate::models::od::validate_equal_lengths;
use crate::models::shared::{cdf_choice, derive_agent_seed, weighted_choice_excluding};

type DitrasResult = Result<(Vec<i64>, Vec<f64>, Vec<f64>, Vec<i64>), String>;

struct DitrasThreadState {
    visited_locs: Vec<usize>,
    visit_counts: Vec<u32>,
    total_visits: f64,
}

impl DitrasThreadState {
    fn new(n: usize) -> Self {
        Self {
            visited_locs: Vec::with_capacity(200),
            visit_counts: vec![0u32; n],
            total_visits: 0.0,
        }
    }
}

fn record_visits(visited: &mut Vec<usize>, counts: &mut [u32], total: &mut f64, loc: usize) {
    if counts[loc] == 0 {
        visited.push(loc);
    }
    counts[loc] += 1;
    *total += 1.0;
}

fn clear_visits(visited: &mut Vec<usize>, counts: &mut [u32], total: &mut f64) {
    for &l in visited.iter() {
        counts[l] = 0;
    }
    visited.clear();
    *total = 0.0;
}

/// Relevance-weighted choice from unvisited locations, excluding `home`.
fn explore_location(
    rng: &mut impl Rng,
    relevances: &[f64],
    visit_counts: &[u32],
    home: usize,
    n: usize,
) -> Option<usize> {
    let candidates: Vec<usize> = (0..n)
        .filter(|&j| visit_counts[j] == 0 && j != home)
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let scores: Vec<f64> = candidates.iter().map(|&j| relevances[j]).collect();
    let total: f64 = scores.iter().sum();
    if total <= 0.0 {
        return Some(candidates[rng.gen_range(0..candidates.len())]);
    }
    let mut cumsum = 0.0_f64;
    let cdf: Vec<f64> = scores
        .iter()
        .map(|&s| {
            cumsum += s;
            cumsum
        })
        .collect();
    Some(candidates[cdf_choice(rng, &cdf)])
}

/// Visit-count-weighted choice from non-home visited locations.
fn return_away(
    rng: &mut impl Rng,
    visited_locs: &[usize],
    visit_counts: &[u32],
    total_visits: f64,
    home: usize,
) -> Option<usize> {
    if !visited_locs.iter().any(|&l| l != home) {
        return None;
    }
    let loc = weighted_choice_excluding(rng, visited_locs, visit_counts, total_visits, home);
    if loc == home { None } else { Some(loc) }
}

/// Uniform choice from any unvisited location — last resort before staying put.
fn any_unvisited(rng: &mut impl Rng, visit_counts: &[u32], n: usize) -> Option<usize> {
    let unvisited: Vec<usize> = (0..n).filter(|&j| visit_counts[j] == 0).collect();
    if unvisited.is_empty() {
        None
    } else {
        Some(unvisited[rng.gen_range(0..unvisited.len())])
    }
}

#[allow(clippy::too_many_arguments)]
fn simulate_one_ditras_agent(
    agent_id: usize,
    home: usize,
    seed: u64,
    lats: &[f64],
    lngs: &[f64],
    relevances: &[f64],
    diary_timestamps: &[i64],
    diary_abs_locs: &[i32],
    diary_start: usize,
    diary_end: usize,
    rho: f64,
    gamma: f64,
    start_ts: i64,
    end_ts: i64,
    n: usize,
    out_agents: &mut Vec<i64>,
    out_ts: &mut Vec<i64>,
    out_lats: &mut Vec<f64>,
    out_lngs: &mut Vec<f64>,
    visited_locs: &mut Vec<usize>,
    visit_counts: &mut [u32],
    total_visits: &mut f64,
) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut cur_loc = home;

    // Record home at start_ts
    out_agents.push(agent_id as i64);
    out_ts.push(start_ts);
    out_lats.push(lats[cur_loc]);
    out_lngs.push(lngs[cur_loc]);
    record_visits(visited_locs, visit_counts, total_visits, cur_loc);

    // Diary index 0 is the start_ts home slot — iterate from index 1 onwards
    for diary_idx in (diary_start + 1)..diary_end {
        let ts = diary_timestamps[diary_idx];
        if ts >= end_ts {
            break;
        }

        let abs_loc = diary_abs_locs[diary_idx];
        let next_loc = if abs_loc == 0 {
            home
        } else {
            let s = visited_locs.len();
            let explore =
                s == 1 || (s < n && rng.gen_range(0.0_f64..1.0) <= rho * (s as f64).powf(-gamma));

            if explore {
                explore_location(&mut rng, relevances, visit_counts, home, n)
                    .or_else(|| {
                        return_away(&mut rng, visited_locs, visit_counts, *total_visits, home)
                    })
                    .or_else(|| any_unvisited(&mut rng, visit_counts, n))
                    .unwrap_or(cur_loc)
            } else {
                return_away(&mut rng, visited_locs, visit_counts, *total_visits, home)
                    .or_else(|| explore_location(&mut rng, relevances, visit_counts, home, n))
                    .or_else(|| any_unvisited(&mut rng, visit_counts, n))
                    .unwrap_or(cur_loc)
            }
        };

        cur_loc = next_loc;
        out_agents.push(agent_id as i64);
        out_ts.push(ts);
        out_lats.push(lats[cur_loc]);
        out_lngs.push(lngs[cur_loc]);
        record_visits(visited_locs, visit_counts, total_visits, cur_loc);
    }
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_ditras_agents_impl(
    lats: &[f64],
    lngs: &[f64],
    relevances: &[f64],
    diary_timestamps: &[i64],
    diary_abs_locs: &[i32],
    diary_starts: &[usize],
    diary_ends: &[usize],
    rho: f64,
    gamma: f64,
    start_ts: i64,
    end_ts: i64,
    n_agents: usize,
    master_seed: Option<u64>,
    starting_locs: Option<&[usize]>,
) -> DitrasResult {
    let n = validate_equal_lengths(&[
        ("latitudes", lats.len()),
        ("longitudes", lngs.len()),
        ("relevances", relevances.len()),
    ])?;
    if diary_starts.len() < n_agents || diary_ends.len() < n_agents {
        return Err(format!(
            "diary_starts/diary_ends must have at least {n_agents} entries"
        ));
    }
    if n == 0 || n_agents == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    }

    let master_seed = master_seed.unwrap_or_else(rand::random);

    let agent_trajectories: Vec<(Vec<i64>, Vec<f64>, Vec<f64>, Vec<i64>)> = (0..n_agents)
        .into_par_iter()
        .map_init(
            || DitrasThreadState::new(n),
            |state, agent| {
                let seed = derive_agent_seed(master_seed, agent, 0);
                let home = match starting_locs {
                    Some(sl) => sl[agent].min(n - 1),
                    None => {
                        let mut rng = Xoshiro256PlusPlus::seed_from_u64(derive_agent_seed(
                            master_seed,
                            agent,
                            1,
                        ));
                        rng.gen_range(0..n)
                    }
                };
                let mut out_agents = Vec::with_capacity(128);
                let mut out_ts = Vec::with_capacity(128);
                let mut out_lats = Vec::with_capacity(128);
                let mut out_lngs = Vec::with_capacity(128);
                simulate_one_ditras_agent(
                    agent + 1,
                    home,
                    seed,
                    lats,
                    lngs,
                    relevances,
                    diary_timestamps,
                    diary_abs_locs,
                    diary_starts[agent],
                    diary_ends[agent],
                    rho,
                    gamma,
                    start_ts,
                    end_ts,
                    n,
                    &mut out_agents,
                    &mut out_ts,
                    &mut out_lats,
                    &mut out_lngs,
                    &mut state.visited_locs,
                    &mut state.visit_counts,
                    &mut state.total_visits,
                );
                clear_visits(
                    &mut state.visited_locs,
                    &mut state.visit_counts,
                    &mut state.total_visits,
                );
                (out_agents, out_lats, out_lngs, out_ts)
            },
        )
        .collect();

    let total: usize = agent_trajectories.iter().map(|(a, _, _, _)| a.len()).sum();
    let mut out_agents = Vec::with_capacity(total);
    let mut out_lats = Vec::with_capacity(total);
    let mut out_lngs = Vec::with_capacity(total);
    let mut out_ts = Vec::with_capacity(total);
    for (agents, lats_c, lngs_c, ts) in agent_trajectories {
        out_agents.extend_from_slice(&agents);
        out_lats.extend_from_slice(&lats_c);
        out_lngs.extend_from_slice(&lngs_c);
        out_ts.extend_from_slice(&ts);
    }
    Ok((out_agents, out_lats, out_lngs, out_ts))
}
