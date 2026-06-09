use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::models::geosim::{
    apply_pending_moves, make_individual_return, make_social_action, GeoSimAgentState,
    PendingMove, SocialGraph, SocialMode,
};
use crate::models::shared::{cdf_choice, derive_agent_seed};

type StsEprResult = Result<(Vec<i64>, Vec<f64>, Vec<f64>, Vec<i64>), String>;

// ---------------------------------------------------------------------------
// Diary state (parallel to GeoSimAgentState)
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
// Gravity-based individual exploration
// ---------------------------------------------------------------------------

fn sts_epr_exploration(
    agent: usize,
    agents: &[GeoSimAgentState],
    distances: &[f64],
    relevances: &[f64],
    n: usize,
    rng: &mut impl Rng,
) -> Option<usize> {
    let src = agents[agent].current_location;
    let home = agents[agent].home_location;
    let lv = &agents[agent].location_vector;
    let src_rel = relevances[src];

    let feasible: Vec<usize> = (0..n)
        .filter(|&j| lv[j] == 0.0 && j != src && j != home)
        .collect();
    if feasible.is_empty() {
        return None;
    }

    let scores: Vec<f64> = feasible
        .iter()
        .map(|&j| {
            let d = distances[src * n + j].max(0.001);
            (1.0 / (d * d)) * relevances[j] * src_rel
        })
        .collect();

    if scores.iter().all(|&s| s == 0.0) {
        return Some(feasible[rng.gen_range(0..feasible.len())]);
    }

    // Build CDF from scores
    let mut cumsum = 0.0_f64;
    let cdf: Vec<f64> = scores
        .iter()
        .map(|&s| {
            cumsum += s;
            cumsum
        })
        .collect();
    Some(feasible[cdf_choice(rng, &cdf)])
}

// ---------------------------------------------------------------------------
// Choose location for STS-EPR (diary-gated)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn sts_epr_choose_location(
    agent: usize,
    agents: &[GeoSimAgentState],
    diaries: &[DiaryState],
    graph: &mut SocialGraph,
    rng: &mut impl Rng,
    rho: f64,
    gamma: f64,
    alpha: f64,
    n_locations: usize,
    current_ts: i64,
    dt_update_s: i64,
    distances: &[f64],
    relevances: &[f64],
    diary_abs_locs: &[i32],
) -> usize {
    let abstract_location = diaries[agent].current_abstract_location(diary_abs_locs);
    if abstract_location == 0 {
        return agents[agent].home_location;
    }

    let s = agents[agent].s.max(1.0);
    let p_explore = rho * s.powf(-gamma);
    let p_social = alpha;

    let explore = rng.gen_range(0.0_f64..1.0) < p_explore;
    let social = rng.gen_range(0.0_f64..1.0) < p_social;

    let location = if explore {
        if social {
            make_social_action(
                agent, agents, graph, SocialMode::Exploration, rng,
                n_locations, current_ts, dt_update_s,
            )
            .or_else(|| sts_epr_exploration(agent, agents, distances, relevances, n_locations, rng))
            .or_else(|| make_individual_return(agent, agents, rng))
        } else {
            sts_epr_exploration(agent, agents, distances, relevances, n_locations, rng)
                .or_else(|| make_social_action(
                    agent, agents, graph, SocialMode::Exploration, rng,
                    n_locations, current_ts, dt_update_s,
                ))
                .or_else(|| make_individual_return(agent, agents, rng))
        }
    } else if social {
        make_social_action(
            agent, agents, graph, SocialMode::Return, rng,
            n_locations, current_ts, dt_update_s,
        )
        .or_else(|| make_individual_return(agent, agents, rng))
        .or_else(|| sts_epr_exploration(agent, agents, distances, relevances, n_locations, rng))
    } else {
        make_individual_return(agent, agents, rng)
            .or_else(|| make_social_action(
                agent, agents, graph, SocialMode::Return, rng,
                n_locations, current_ts, dt_update_s,
            ))
            .or_else(|| sts_epr_exploration(agent, agents, distances, relevances, n_locations, rng))
    };

    location.unwrap_or(agents[agent].current_location)
}

// ---------------------------------------------------------------------------
// Starting location helpers
// ---------------------------------------------------------------------------

fn pick_starting_loc(
    relevances: &[f64],
    rng: &mut impl Rng,
    mode_relevance: bool,
) -> usize {
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
// Main STS-EPR simulation
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
    if neighbor_starts.len() != n_agents + 1 {
        return Err(format!(
            "neighbor_starts must have length n_agents+1={}, got {}",
            n_agents + 1,
            neighbor_starts.len()
        ));
    }

    let master_seed = master_seed.unwrap_or_else(rand::random);

    // Per-agent RNG
    let mut rngs: Vec<Xoshiro256PlusPlus> = (0..n_agents)
        .map(|i| Xoshiro256PlusPlus::seed_from_u64(derive_agent_seed(master_seed, i, 0)))
        .collect();

    // Init GeoSim agent states
    let mut agents: Vec<GeoSimAgentState> = (0..n_agents)
        .map(|_| GeoSimAgentState::new(n_locations, start_ts))
        .collect();

    // Init diary states
    let mut diaries: Vec<DiaryState> = (0..n_agents)
        .map(|i| DiaryState {
            diary_start: if i < diary_starts.len() { diary_starts[i] } else { 0 },
            diary_end: if i < diary_ends.len() { diary_ends[i] } else { 0 },
            diary_idx: 1, // skip the first entry (used for the initial timestamp)
        })
        .collect();

    // Init social graph
    let mut graph = SocialGraph::new(neighbor_starts.to_vec(), neighbors.to_vec(), start_ts);

    // Starting locations and initial time_next_move from diary
    for (i, agent) in agents.iter_mut().enumerate() {
        let loc = if let Some(sl) = starting_locs {
            if i < sl.len() { sl[i].min(n_locations - 1) }
            else { pick_starting_loc(relevances, &mut rngs[i], starting_locs_mode_relevance) }
        } else {
            pick_starting_loc(relevances, &mut rngs[i], starting_locs_mode_relevance)
        };
        agent.current_location = loc;
        agent.home_location = loc;
        agent.visit(loc);

        agent.time_next_move = diaries[i]
            .current_ts(diary_timestamps)
            .unwrap_or(end_ts + 3600);
    }

    // Output buffers
    let mut out_agents: Vec<i64> = Vec::new();
    let mut out_ts: Vec<i64> = Vec::new();
    let mut out_lats_vec: Vec<f64> = Vec::new();
    let mut out_lngs_vec: Vec<f64> = Vec::new();

    // Record initial positions
    for (i, agent) in agents.iter().enumerate() {
        out_agents.push(i as i64 + 1);
        out_ts.push(start_ts);
        out_lats_vec.push(lats[agent.current_location]);
        out_lngs_vec.push(lngs[agent.current_location]);
    }

    let mut pending: Vec<PendingMove> = Vec::new();

    // Priority queue
    let mut heap: BinaryHeap<(Reverse<i64>, usize)> = BinaryHeap::new();
    for (i, agent) in agents.iter().enumerate() {
        if agent.time_next_move < end_ts {
            heap.push((Reverse(agent.time_next_move), i));
        }
    }

    while let Some((Reverse(current_ts), agent_id)) = heap.peek().copied() {
        if current_ts >= end_ts {
            break;
        }
        heap.pop();

        // Flush old moves
        let cutoff = current_ts - indipendency_window_s;
        apply_pending_moves(
            &mut pending,
            &mut agents,
            cutoff,
            lats,
            lngs,
            &mut out_agents,
            &mut out_ts,
            &mut out_lats_vec,
            &mut out_lngs_vec,
        );

        // Choose location
        let loc = sts_epr_choose_location(
            agent_id,
            &agents,
            &diaries,
            &mut graph,
            &mut rngs[agent_id],
            rho,
            gamma,
            alpha,
            n_locations,
            current_ts,
            dt_update_mob_sim_s,
            distances,
            relevances,
            diary_abs_locs,
        );

        pending.push(PendingMove {
            agent: agent_id,
            location: loc,
            timestamp: current_ts,
        });

        // Advance diary
        let next_ts = diaries[agent_id].advance(diary_timestamps, end_ts);
        agents[agent_id].time_next_move = next_ts;
        if next_ts < end_ts {
            heap.push((Reverse(next_ts), agent_id));
        }
    }

    // Flush remaining pending moves
    apply_pending_moves(
        &mut pending,
        &mut agents,
        end_ts,
        lats,
        lngs,
        &mut out_agents,
        &mut out_ts,
        &mut out_lats_vec,
        &mut out_lngs_vec,
    );

    Ok((out_agents, out_lats_vec, out_lngs_vec, out_ts))
}
