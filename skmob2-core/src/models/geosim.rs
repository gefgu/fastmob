use std::cmp::Reverse;
use std::collections::BinaryHeap;

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::models::shared::{derive_agent_seed, sample_tpl_rng};

type GeoSimResult = Result<(Vec<i64>, Vec<f64>, Vec<f64>, Vec<i64>), String>;

// ---------------------------------------------------------------------------
// Social graph (CSR adjacency with per-edge mobility similarity)
// ---------------------------------------------------------------------------

pub(crate) struct SocialGraph {
    pub neighbor_starts: Vec<usize>,
    pub neighbors: Vec<usize>,
    pub mobility_similarity: Vec<f64>,
    pub next_update_s: Vec<i64>,
}

impl SocialGraph {
    pub fn new(neighbor_starts: Vec<usize>, neighbors: Vec<usize>, init_ts: i64) -> Self {
        let n_edges = neighbors.len();
        Self {
            neighbor_starts,
            neighbors,
            mobility_similarity: vec![0.0_f64; n_edges],
            next_update_s: vec![init_ts; n_edges],
        }
    }

    pub fn edge_range(&self, agent: usize) -> std::ops::Range<usize> {
        if agent + 1 < self.neighbor_starts.len() {
            self.neighbor_starts[agent]..self.neighbor_starts[agent + 1]
        } else {
            0..0
        }
    }
}

// ---------------------------------------------------------------------------
// Agent state
// ---------------------------------------------------------------------------

pub(crate) struct GeoSimAgentState {
    pub current_location: usize,
    pub home_location: usize,
    pub location_vector: Vec<f64>,
    pub s: f64,
    pub time_next_move: i64,
}

impl GeoSimAgentState {
    pub fn new(n_locations: usize, start_ts: i64) -> Self {
        Self {
            current_location: 0,
            home_location: 0,
            location_vector: vec![0.0_f64; n_locations],
            s: 0.0,
            time_next_move: start_ts,
        }
    }

    pub fn visit(&mut self, loc: usize) {
        if self.location_vector[loc] == 0.0 {
            self.s += 1.0;
        }
        self.location_vector[loc] += 1.0;
    }
}

// ---------------------------------------------------------------------------
// Pending moves (staged until indipendency_window elapses)
// ---------------------------------------------------------------------------

pub(crate) struct PendingMove {
    pub agent: usize,
    pub location: usize,
    pub timestamp: i64,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub(crate) fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

fn weighted_choice(rng: &mut impl Rng, weights: &[f64]) -> Option<usize> {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return None;
    }
    let threshold = rng.gen_range(0.0_f64..1.0) * total;
    let mut cumsum = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cumsum += w;
        if cumsum > threshold {
            return Some(i);
        }
    }
    Some(weights.len() - 1)
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) enum SocialMode {
    Exploration,
    Return,
}

fn update_edge_similarity(
    agent: usize,
    agents: &[GeoSimAgentState],
    graph: &mut SocialGraph,
    current_ts: i64,
    dt_update_s: i64,
) {
    for edge_idx in graph.edge_range(agent) {
        if graph.next_update_s[edge_idx] <= current_ts {
            let neighbor = graph.neighbors[edge_idx];
            graph.mobility_similarity[edge_idx] =
                cosine_similarity(&agents[agent].location_vector, &agents[neighbor].location_vector);
            graph.next_update_s[edge_idx] = current_ts + dt_update_s;
        }
    }
}

pub(crate) fn make_social_action(
    agent: usize,
    agents: &[GeoSimAgentState],
    graph: &mut SocialGraph,
    mode: SocialMode,
    rng: &mut impl Rng,
    n_locations: usize,
    current_ts: i64,
    dt_update_s: i64,
) -> Option<usize> {
    // Update stale edges for this agent
    update_edge_similarity(agent, agents, graph, current_ts, dt_update_s);

    let range = graph.edge_range(agent);
    if range.is_empty() {
        return None;
    }

    let sims: Vec<f64> = range.clone().map(|e| graph.mobility_similarity[e]).collect();
    let neighbors: Vec<usize> = range.map(|e| graph.neighbors[e]).collect();

    // Choose contact weighted by mobility similarity (or uniform if all zero)
    let contact_idx = if sims.iter().all(|&s| s == 0.0) {
        rng.gen_range(0..sims.len())
    } else {
        weighted_choice(rng, &sims).unwrap_or(0)
    };
    let contact = neighbors[contact_idx];

    let agent_lv = &agents[agent].location_vector;
    let contact_lv = &agents[contact].location_vector;

    let feasible: Vec<usize> = (0..n_locations)
        .filter(|&loc| match mode {
            SocialMode::Exploration => agent_lv[loc] == 0.0,
            SocialMode::Return => agent_lv[loc] >= 1.0,
        })
        .collect();
    if feasible.is_empty() {
        return None;
    }

    let weights: Vec<f64> = feasible.iter().map(|&loc| contact_lv[loc]).collect();
    if weights.iter().all(|&w| w == 0.0) {
        return None;
    }
    weighted_choice(rng, &weights).map(|idx| feasible[idx])
}

pub(crate) fn make_individual_return(
    agent: usize,
    agents: &[GeoSimAgentState],
    rng: &mut impl Rng,
) -> Option<usize> {
    let lv = &agents[agent].location_vector;
    let feasible: Vec<usize> = (0..lv.len()).filter(|&loc| lv[loc] >= 1.0).collect();
    if feasible.is_empty() {
        return None;
    }
    let weights: Vec<f64> = feasible.iter().map(|&loc| lv[loc]).collect();
    weighted_choice(rng, &weights).map(|idx| feasible[idx])
}

pub(crate) fn make_individual_exploration(
    agent: usize,
    agents: &[GeoSimAgentState],
    rng: &mut impl Rng,
) -> Option<usize> {
    let lv = &agents[agent].location_vector;
    let feasible: Vec<usize> = (0..lv.len()).filter(|&loc| lv[loc] == 0.0).collect();
    if feasible.is_empty() {
        return None;
    }
    Some(feasible[rng.gen_range(0..feasible.len())])
}

/// Choose location following the EPR+social decision tree with fallback chain.
/// Always returns a valid location index.
pub(crate) fn choose_location_geosim(
    agent: usize,
    agents: &[GeoSimAgentState],
    graph: &mut SocialGraph,
    rng: &mut impl Rng,
    rho: f64,
    gamma: f64,
    alpha: f64,
    n_locations: usize,
    current_ts: i64,
    dt_update_s: i64,
) -> usize {
    let s = agents[agent].s.max(1.0);
    let p_explore = rho * s.powf(-gamma);
    let p_social = alpha;

    let explore = rng.gen_range(0.0_f64..1.0) < p_explore;
    let social = rng.gen_range(0.0_f64..1.0) < p_social;

    let location = if explore {
        if social {
            // social exploration → individual exploration → individual return
            make_social_action(
                agent,
                agents,
                graph,
                SocialMode::Exploration,
                rng,
                n_locations,
                current_ts,
                dt_update_s,
            )
            .or_else(|| make_individual_exploration(agent, agents, rng))
            .or_else(|| make_individual_return(agent, agents, rng))
        } else {
            // individual exploration → social exploration → individual return
            make_individual_exploration(agent, agents, rng)
                .or_else(|| {
                    make_social_action(
                        agent,
                        agents,
                        graph,
                        SocialMode::Exploration,
                        rng,
                        n_locations,
                        current_ts,
                        dt_update_s,
                    )
                })
                .or_else(|| make_individual_return(agent, agents, rng))
        }
    } else if social {
        // social return → individual return → individual exploration
        make_social_action(
            agent,
            agents,
            graph,
            SocialMode::Return,
            rng,
            n_locations,
            current_ts,
            dt_update_s,
        )
        .or_else(|| make_individual_return(agent, agents, rng))
        .or_else(|| make_individual_exploration(agent, agents, rng))
    } else {
        // individual return → social return → individual exploration
        make_individual_return(agent, agents, rng)
            .or_else(|| {
                make_social_action(
                    agent,
                    agents,
                    graph,
                    SocialMode::Return,
                    rng,
                    n_locations,
                    current_ts,
                    dt_update_s,
                )
            })
            .or_else(|| make_individual_exploration(agent, agents, rng))
    };

    // Last resort: stay at current location
    location.unwrap_or(agents[agent].current_location)
}

// ---------------------------------------------------------------------------
// Apply pending moves whose timestamps are old enough
// ---------------------------------------------------------------------------

pub(crate) fn apply_pending_moves(
    pending: &mut Vec<PendingMove>,
    agents: &mut [GeoSimAgentState],
    cutoff_ts: i64,
    lats: &[f64],
    lngs: &[f64],
    out_agents: &mut Vec<i64>,
    out_ts: &mut Vec<i64>,
    out_lats: &mut Vec<f64>,
    out_lngs: &mut Vec<f64>,
) {
    let mut keep = Vec::with_capacity(pending.len());
    let mut apply = Vec::new();
    for mv in pending.drain(..) {
        if mv.timestamp <= cutoff_ts {
            apply.push(mv);
        } else {
            keep.push(mv);
        }
    }
    // sort apply by timestamp then agent for determinism
    apply.sort_by_key(|m| (m.timestamp, m.agent));
    for mv in apply {
        agents[mv.agent].visit(mv.location);
        agents[mv.agent].current_location = mv.location;
        out_agents.push(mv.agent as i64 + 1);
        out_ts.push(mv.timestamp);
        out_lats.push(lats[mv.location]);
        out_lngs.push(lngs[mv.location]);
    }
    *pending = keep;
}

// ---------------------------------------------------------------------------
// Main GeoSim simulation
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn simulate_geosim_impl(
    lats: &[f64],
    lngs: &[f64],
    neighbor_starts: &[usize],
    neighbors: &[usize],
    rho: f64,
    gamma: f64,
    alpha: f64,
    beta: f64,
    tau: f64,
    xmin: f64,
    start_ts: i64,
    end_ts: i64,
    indipendency_window_s: i64,
    dt_update_mob_sim_s: i64,
    n_agents: usize,
    master_seed: Option<u64>,
    starting_locs: Option<&[usize]>,
) -> GeoSimResult {
    let n_locations = lats.len();
    if n_locations < 2 {
        return Err("need at least 2 locations".to_string());
    }
    if neighbor_starts.len() != n_agents + 1 {
        return Err(format!(
            "neighbor_starts must have length n_agents+1={}, got {}",
            n_agents + 1,
            neighbor_starts.len()
        ));
    }

    let master_seed = master_seed.unwrap_or_else(rand::random);
    let alpha_param = 1.0 + beta;
    let lambda_ = 1.0 / tau;

    // Init agents
    let mut agents: Vec<GeoSimAgentState> = (0..n_agents)
        .map(|_| GeoSimAgentState::new(n_locations, start_ts))
        .collect();

    // Init social graph
    let mut graph = SocialGraph::new(neighbor_starts.to_vec(), neighbors.to_vec(), start_ts);

    // Starting locations
    for (i, agent) in agents.iter_mut().enumerate() {
        let loc = match starting_locs {
            Some(sl) if i < sl.len() => sl[i].min(n_locations - 1),
            _ => {
                let mut rng = Xoshiro256PlusPlus::seed_from_u64(derive_agent_seed(master_seed, i, 1));
                rng.gen_range(0..n_locations)
            }
        };
        agent.current_location = loc;
        agent.home_location = loc;
        agent.visit(loc);

        // Sample initial waiting time
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(derive_agent_seed(master_seed, i, 0));
        let wait_h = sample_tpl_rng(&mut rng, xmin, alpha_param, lambda_);
        agent.time_next_move = start_ts + (wait_h * 3600.0) as i64;
    }

    // Output trajectory buffers
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

    // Per-agent RNG (one per agent, for the movement decision)
    let mut rngs: Vec<Xoshiro256PlusPlus> = (0..n_agents)
        .map(|i| Xoshiro256PlusPlus::seed_from_u64(derive_agent_seed(master_seed, i, 0)))
        .collect();

    // Priority queue: (time_next_move, agent_id)
    let mut heap: BinaryHeap<(Reverse<i64>, usize)> = BinaryHeap::new();
    for (i, agent) in agents.iter().enumerate() {
        heap.push((Reverse(agent.time_next_move), i));
    }

    let mut pending: Vec<PendingMove> = Vec::new();

    while let Some((Reverse(current_ts), agent)) = heap.peek().copied() {
        if current_ts >= end_ts {
            break;
        }
        heap.pop();

        // Flush moves older than the indipendency window
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

        // Choose location for this agent
        let loc = choose_location_geosim(
            agent,
            &agents,
            &mut graph,
            &mut rngs[agent],
            rho,
            gamma,
            alpha,
            n_locations,
            current_ts,
            dt_update_mob_sim_s,
        );

        // Stage the move
        pending.push(PendingMove {
            agent,
            location: loc,
            timestamp: current_ts,
        });

        // Sample next waiting time
        let wait_h = sample_tpl_rng(&mut rngs[agent], xmin, alpha_param, lambda_);
        let next_ts = current_ts + (wait_h * 3600.0) as i64;
        agents[agent].time_next_move = next_ts;

        if next_ts < end_ts {
            heap.push((Reverse(next_ts), agent));
        }
    }

    // Flush all remaining pending moves
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
