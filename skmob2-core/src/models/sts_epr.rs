use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use crate::models::geosim::{
    GeoSimAgentState, GeoSimScratch, SocialMode, cosine_similarity_sparse, make_individual_return,
};
use crate::models::od::CachedGravityOdRows;
use crate::models::shared::{cdf_choice, derive_agent_seed};

type StsEprResult = Result<(Vec<i64>, Vec<f64>, Vec<f64>, Vec<i64>), String>;
const GRAVITY_REJECTION_ATTEMPTS: usize = 16;

// ---------------------------------------------------------------------------
// Diary state
// ---------------------------------------------------------------------------

struct DiaryState {
    diary_start: usize,
    diary_end: usize,
    diary_idx: usize,
}

impl DiaryState {
    fn current_ts(&self, diary_timestamps: &[i64]) -> Option<i64> {
        let idx = self.diary_start + self.diary_idx;
        if idx < self.diary_end {
            Some(diary_timestamps[idx])
        } else {
            None
        }
    }

    fn current_abstract_location(&self, diary_abs_locs: &[i32]) -> i32 {
        let idx = self.diary_start + self.diary_idx;
        if idx < self.diary_end {
            diary_abs_locs[idx]
        } else {
            0
        }
    }

    fn advance(&mut self, diary_timestamps: &[i64], end_ts: i64) -> i64 {
        self.diary_idx += 1;
        self.current_ts(diary_timestamps).unwrap_or(end_ts + 3600)
    }
}

// ---------------------------------------------------------------------------
// Per-agent state for the parallel window loop
// ---------------------------------------------------------------------------

struct AgentParData {
    rng: Xoshiro256PlusPlus,
    diary: DiaryState,
    scratch: GeoSimScratch,
    moves: Vec<(usize, i64)>,
    // This agent's own copy of its social-graph edge arrays.
    // Indices within these slices align with neighbor_indices.
    neighbor_indices: Vec<usize>,
    edge_sim: Vec<f64>,
    edge_upd: Vec<i64>,
}

// ---------------------------------------------------------------------------
// Gravity-based individual exploration (scratch-ified)
// ---------------------------------------------------------------------------

fn sts_epr_exploration(
    agent: usize,
    agents: &[GeoSimAgentState],
    distances: &[f64],
    od_rows: Option<&CachedGravityOdRows>,
    relevances: &[f64],
    n: usize,
    rng: &mut impl Rng,
    scratch: &mut GeoSimScratch,
) -> Option<usize> {
    let src = agents[agent].current_location;
    let home = agents[agent].home_location;
    let visit_counts = &agents[agent].visit_counts;

    if let Some(od_rows) = od_rows {
        let row = od_rows.get(src);
        if !row.is_empty() {
            let total = row[row.len() - 1];
            if total.is_finite() && total > 0.0 {
                for _ in 0..GRAVITY_REJECTION_ATTEMPTS {
                    let loc = cdf_choice(rng, &row);
                    if visit_counts[loc] == 0 && loc != src && loc != home {
                        return Some(loc);
                    }
                }
            }
            scratch.candidates.clear();
            scratch.cdf.clear();
            let mut prev = 0.0_f64;
            let mut cumsum = 0.0_f64;
            for (j, &value) in row.iter().enumerate() {
                let weight = value - prev;
                prev = value;
                if visit_counts[j] == 0
                    && j != src
                    && j != home
                    && weight.is_finite()
                    && weight > 0.0
                {
                    scratch.candidates.push(j);
                    cumsum += weight;
                    scratch.cdf.push(cumsum);
                }
            }
            if scratch.candidates.is_empty() {
                return None;
            }
            let total = scratch.cdf[scratch.cdf.len() - 1];
            let threshold = rng.gen_range(0.0_f64..1.0) * total;
            let idx = scratch
                .cdf
                .partition_point(|&v| v <= threshold)
                .min(scratch.candidates.len() - 1);
            return Some(scratch.candidates[idx]);
        }
    }

    let src_rel = relevances[src];
    scratch.candidates.clear();
    scratch.cdf.clear();
    let mut all_zero = true;
    let mut cumsum = 0.0_f64;
    for j in 0..n {
        if visit_counts[j] != 0 || j == src || j == home {
            continue;
        }
        let d = distances[src * n + j].max(0.001);
        let score = (1.0 / (d * d)) * relevances[j] * src_rel;
        scratch.candidates.push(j);
        if score > 0.0 {
            all_zero = false;
        }
        cumsum += score;
        scratch.cdf.push(cumsum);
    }
    if scratch.candidates.is_empty() {
        return None;
    }
    if all_zero {
        return Some(scratch.candidates[rng.gen_range(0..scratch.candidates.len())]);
    }
    let total = scratch.cdf[scratch.cdf.len() - 1];
    let threshold = rng.gen_range(0.0_f64..1.0) * total;
    let idx = scratch
        .cdf
        .partition_point(|&v| v <= threshold)
        .min(scratch.candidates.len() - 1);
    Some(scratch.candidates[idx])
}

// ---------------------------------------------------------------------------
// Social action using per-agent owned edge data (parallel-safe)
// ---------------------------------------------------------------------------

fn update_edge_sim_local(
    agent: usize,
    agents: &[GeoSimAgentState],
    neighbor_indices: &[usize],
    edge_sim: &mut [f64],
    edge_upd: &mut [i64],
    current_ts: i64,
    dt_update_s: i64,
) {
    for i in 0..neighbor_indices.len() {
        if edge_upd[i] <= current_ts {
            let nb = neighbor_indices[i];
            edge_sim[i] = cosine_similarity_sparse(
                &agents[agent].visited_locs,
                &agents[agent].visit_counts,
                &agents[nb].visited_locs,
                &agents[nb].visit_counts,
            );
            edge_upd[i] = current_ts + dt_update_s;
        }
    }
}

fn make_social_action_local(
    agent: usize,
    agents: &[GeoSimAgentState],
    neighbor_indices: &[usize],
    edge_sim: &mut [f64],
    edge_upd: &mut [i64],
    mode: SocialMode,
    rng: &mut impl Rng,
    current_ts: i64,
    dt_update_s: i64,
    scratch: &mut GeoSimScratch,
) -> Option<usize> {
    if neighbor_indices.is_empty() {
        return None;
    }
    update_edge_sim_local(agent, agents, neighbor_indices, edge_sim, edge_upd, current_ts, dt_update_s);

    scratch.sims.clear();
    scratch.sims.extend_from_slice(edge_sim);

    let contact_idx = if scratch.sims.iter().all(|&s| s == 0.0) {
        rng.gen_range(0..scratch.sims.len())
    } else {
        let total: f64 = scratch.sims.iter().sum();
        let threshold = rng.gen_range(0.0_f64..1.0) * total;
        let mut cumsum = 0.0_f64;
        let mut chosen = scratch.sims.len() - 1;
        for (i, &w) in scratch.sims.iter().enumerate() {
            cumsum += w;
            if cumsum > threshold {
                chosen = i;
                break;
            }
        }
        chosen
    };
    let contact = neighbor_indices[contact_idx];

    let agent_counts = &agents[agent].visit_counts;
    let contact_counts = &agents[contact].visit_counts;
    let contact_locs = &agents[contact].visited_locs;
    let agent_locs = &agents[agent].visited_locs;

    scratch.candidates.clear();
    scratch.cdf.clear();

    match mode {
        SocialMode::Exploration => {
            let mut cumsum = 0.0_f64;
            for &loc in contact_locs {
                if agent_counts[loc] == 0 {
                    scratch.candidates.push(loc);
                    cumsum += contact_counts[loc] as f64;
                    scratch.cdf.push(cumsum);
                }
            }
        }
        SocialMode::Return => {
            let mut cumsum = 0.0_f64;
            for &loc in agent_locs {
                let w = contact_counts[loc] as f64;
                if w > 0.0 {
                    scratch.candidates.push(loc);
                    cumsum += w;
                    scratch.cdf.push(cumsum);
                }
            }
        }
    }

    if scratch.candidates.is_empty() || scratch.cdf.last().copied().unwrap_or(0.0) <= 0.0 {
        return None;
    }

    let total = scratch.cdf[scratch.cdf.len() - 1];
    let threshold = rng.gen_range(0.0_f64..1.0) * total;
    let idx = scratch
        .cdf
        .partition_point(|&v| v <= threshold)
        .min(scratch.candidates.len() - 1);
    Some(scratch.candidates[idx])
}

// ---------------------------------------------------------------------------
// Choose location for STS-EPR (parallel version with local edge data)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn sts_epr_choose_location_local(
    agent: usize,
    agents: &[GeoSimAgentState],
    diary: &DiaryState,
    neighbor_indices: &[usize],
    edge_sim: &mut [f64],
    edge_upd: &mut [i64],
    rng: &mut impl Rng,
    rho: f64,
    gamma: f64,
    alpha: f64,
    n_locations: usize,
    current_ts: i64,
    dt_update_s: i64,
    distances: &[f64],
    od_rows: Option<&CachedGravityOdRows>,
    relevances: &[f64],
    diary_abs_locs: &[i32],
    scratch: &mut GeoSimScratch,
) -> usize {
    let abstract_location = diary.current_abstract_location(diary_abs_locs);
    if abstract_location == 0 {
        return agents[agent].home_location;
    }

    let s = agents[agent].s.max(1.0);
    let p_explore = rho * s.powf(-gamma);

    let explore = rng.gen_range(0.0_f64..1.0) < p_explore;
    let social = rng.gen_range(0.0_f64..1.0) < alpha;

    let location = if explore {
        if social {
            make_social_action_local(agent, agents, neighbor_indices, edge_sim, edge_upd, SocialMode::Exploration, rng, current_ts, dt_update_s, scratch)
                .or_else(|| sts_epr_exploration(agent, agents, distances, od_rows, relevances, n_locations, rng, scratch))
                .or_else(|| make_individual_return(agent, agents, rng, scratch))
        } else {
            sts_epr_exploration(agent, agents, distances, od_rows, relevances, n_locations, rng, scratch)
                .or_else(|| make_social_action_local(agent, agents, neighbor_indices, edge_sim, edge_upd, SocialMode::Exploration, rng, current_ts, dt_update_s, scratch))
                .or_else(|| make_individual_return(agent, agents, rng, scratch))
        }
    } else if social {
        make_social_action_local(agent, agents, neighbor_indices, edge_sim, edge_upd, SocialMode::Return, rng, current_ts, dt_update_s, scratch)
            .or_else(|| make_individual_return(agent, agents, rng, scratch))
            .or_else(|| sts_epr_exploration(agent, agents, distances, od_rows, relevances, n_locations, rng, scratch))
    } else {
        make_individual_return(agent, agents, rng, scratch)
            .or_else(|| make_social_action_local(agent, agents, neighbor_indices, edge_sim, edge_upd, SocialMode::Return, rng, current_ts, dt_update_s, scratch))
            .or_else(|| sts_epr_exploration(agent, agents, distances, od_rows, relevances, n_locations, rng, scratch))
    };

    location.unwrap_or(agents[agent].current_location)
}

// ---------------------------------------------------------------------------
// Starting location helpers
// ---------------------------------------------------------------------------

fn pick_starting_loc(relevances: &[f64], rng: &mut impl Rng, mode_relevance: bool) -> usize {
    let n = relevances.len();
    if !mode_relevance {
        return rng.gen_range(0..n);
    }
    let total: f64 = relevances.iter().sum();
    if total <= 0.0 {
        return rng.gen_range(0..n);
    }
    let threshold = rng.gen_range(0.0_f64..1.0) * total;
    let mut cumsum = 0.0;
    for (i, &r) in relevances.iter().enumerate() {
        cumsum += r;
        if cumsum > threshold {
            return i;
        }
    }
    n - 1
}

// ---------------------------------------------------------------------------
// Main STS-EPR simulation — parallel tumbling-window loop
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn simulate_sts_epr_impl(
    lats: &[f64],
    lngs: &[f64],
    relevances: &[f64],
    distances: &[f64],
    neighbor_starts: &[usize],
    neighbors: &[usize],
    diary_timestamps: &[i64],
    diary_abs_locs: &[i32],
    diary_starts: &[usize],
    diary_ends: &[usize],
    rho: f64,
    gamma: f64,
    alpha: f64,
    start_ts: i64,
    end_ts: i64,
    indipendency_window_s: i64,
    dt_update_mob_sim_s: i64,
    n_agents: usize,
    master_seed: Option<u64>,
    starting_locs: Option<&[usize]>,
    starting_locs_mode_relevance: bool,
) -> StsEprResult {
    let n_locations = lats.len();
    if n_locations < 3 {
        return Err("need at least 3 locations".to_string());
    }
    if !distances.is_empty() && distances.len() != n_locations * n_locations {
        return Err(format!(
            "distances must be empty or have length n_locations*n_locations={}, got {}",
            n_locations * n_locations,
            distances.len()
        ));
    }
    if neighbor_starts.len() != n_agents + 1 {
        return Err(format!(
            "neighbor_starts must have length n_agents+1={}, got {}",
            n_agents + 1,
            neighbor_starts.len()
        ));
    }

    let master_seed = master_seed.unwrap_or_else(rand::random);
    let od_rows = if distances.is_empty() {
        Some(CachedGravityOdRows::new(
            lats, lngs, relevances, "power_law", -2.0, 1.0, 1.0,
        ))
    } else {
        None
    };

    // Shared frozen public state (mutated only in the sequential commit phase).
    let mut agents: Vec<GeoSimAgentState> = (0..n_agents)
        .map(|_| GeoSimAgentState::new(n_locations, start_ts))
        .collect();

    // Build per-agent private state, including a local copy of edge arrays so
    // each agent can mutate its own edge similarity values without touching shared
    // memory.  This is the key data-structure move that enables safe parallelism:
    // disjoint ownership instead of a shared CSR with split_at_mut.
    let init_ts_edges = vec![start_ts; neighbors.len()];
    let mut par_data: Vec<AgentParData> = (0..n_agents)
        .map(|i| {
            let edge_start = neighbor_starts[i];
            let edge_end = neighbor_starts[i + 1];
            AgentParData {
                rng: Xoshiro256PlusPlus::seed_from_u64(derive_agent_seed(master_seed, i, 0)),
                diary: DiaryState {
                    diary_start: if i < diary_starts.len() { diary_starts[i] } else { 0 },
                    diary_end: if i < diary_ends.len() { diary_ends[i] } else { 0 },
                    diary_idx: 1,
                },
                scratch: GeoSimScratch::new(),
                moves: Vec::with_capacity(32),
                neighbor_indices: neighbors[edge_start..edge_end].to_vec(),
                edge_sim: vec![0.0_f64; edge_end - edge_start],
                edge_upd: init_ts_edges[edge_start..edge_end].to_vec(),
            }
        })
        .collect();

    // Set starting locations using the same per-agent RNG that will be used for
    // simulation, so the RNG sequence matches the original event-loop implementation.
    for (i, agent) in agents.iter_mut().enumerate() {
        let loc = if let Some(sl) = starting_locs {
            if i < sl.len() {
                sl[i].min(n_locations - 1)
            } else {
                pick_starting_loc(relevances, &mut par_data[i].rng, starting_locs_mode_relevance)
            }
        } else {
            pick_starting_loc(relevances, &mut par_data[i].rng, starting_locs_mode_relevance)
        };
        agent.current_location = loc;
        agent.home_location = loc;
        agent.visit(loc);
    }

    let mut out_agents: Vec<i64> = Vec::new();
    let mut out_ts: Vec<i64> = Vec::new();
    let mut out_lats_vec: Vec<f64> = Vec::new();
    let mut out_lngs_vec: Vec<f64> = Vec::new();

    for (i, agent) in agents.iter().enumerate() {
        out_agents.push(i as i64 + 1);
        out_ts.push(start_ts);
        out_lats_vec.push(lats[agent.current_location]);
        out_lngs_vec.push(lngs[agent.current_location]);
    }

    // Commit buffer: (timestamp, agent_id, location) — reused across windows.
    let mut commit_buf: Vec<(i64, usize, usize)> = Vec::new();

    let mut window_start = start_ts;
    while window_start < end_ts {
        let window_end = (window_start + indipendency_window_s).min(end_ts);

        // --- PARALLEL: each agent simulates its diary entries for this window ---
        par_data.par_iter_mut().enumerate().for_each(|(a, data)| {
            while let Some(ts) = data.diary.current_ts(diary_timestamps) {
                if ts >= window_end {
                    break;
                }

                let loc = sts_epr_choose_location_local(
                    a,
                    &agents,
                    &data.diary,
                    &data.neighbor_indices,
                    &mut data.edge_sim,
                    &mut data.edge_upd,
                    &mut data.rng,
                    rho,
                    gamma,
                    alpha,
                    n_locations,
                    ts,
                    dt_update_mob_sim_s,
                    distances,
                    od_rows.as_ref(),
                    relevances,
                    diary_abs_locs,
                    &mut data.scratch,
                );

                data.moves.push((loc, ts));
                data.diary.advance(diary_timestamps, end_ts);
            }
        });

        // --- SEQUENTIAL commit: sort + apply to public state + write output ---
        commit_buf.clear();
        for (a, data) in par_data.iter_mut().enumerate() {
            for &(loc, ts) in &data.moves {
                commit_buf.push((ts, a, loc));
            }
            data.moves.clear();
        }
        commit_buf.sort_unstable_by_key(|&(ts, a, _)| (ts, a));
        for &(ts, a, loc) in &commit_buf {
            agents[a].visit(loc);
            agents[a].current_location = loc;
            out_agents.push(a as i64 + 1);
            out_ts.push(ts);
            out_lats_vec.push(lats[loc]);
            out_lngs_vec.push(lngs[loc]);
        }

        window_start = window_end;
    }

    Ok((out_agents, out_lats_vec, out_lngs_vec, out_ts))
}
