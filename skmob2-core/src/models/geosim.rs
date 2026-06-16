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
// Reusable scratch buffers — one instance per simulation, passed by &mut
// ---------------------------------------------------------------------------

pub(crate) struct GeoSimScratch {
    pub candidates: Vec<usize>,
    pub cdf: Vec<f64>,
    pub sims: Vec<f64>,
    pub neighbors: Vec<usize>,
}

impl GeoSimScratch {
    pub fn new() -> Self {
        Self {
            candidates: Vec::with_capacity(200),
            cdf: Vec::with_capacity(200),
            sims: Vec::with_capacity(64),
            neighbors: Vec::with_capacity(64),
        }
    }
}

// ---------------------------------------------------------------------------
// Agent state (sparse visit representation)
// ---------------------------------------------------------------------------

pub(crate) struct GeoSimAgentState {
    pub current_location: usize,
    pub home_location: usize,
    pub visited_locs: Vec<usize>,  // sparse list of visited location indices
    pub visit_counts: Vec<u32>,    // dense counts[loc]; 0 = unvisited
    pub total_visits: f64,
    pub s: f64,                    // unique-location count (len of visited_locs)
    pub time_next_move: i64,
}

impl GeoSimAgentState {
    pub fn new(n_locations: usize, start_ts: i64) -> Self {
        Self {
            current_location: 0,
            home_location: 0,
            visited_locs: Vec::with_capacity(200),
            visit_counts: vec![0u32; n_locations],
            total_visits: 0.0,
            s: 0.0,
            time_next_move: start_ts,
        }
    }

    pub fn visit(&mut self, loc: usize) {
        if self.visit_counts[loc] == 0 {
            self.s += 1.0;
            self.visited_locs.push(loc);
        }
        self.visit_counts[loc] += 1;
        self.total_visits += 1.0;
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
// Internal helpers
// ---------------------------------------------------------------------------

// O(min(|a|,|b|)) cosine similarity over sparse visited sets.
pub(crate) fn cosine_similarity_sparse(
    a_locs: &[usize],
    a_counts: &[u32],
    b_locs: &[usize],
    b_counts: &[u32],
) -> f64 {
    let dot: f64 = a_locs
        .iter()
        .map(|&l| (a_counts[l] as f64) * (b_counts[l] as f64))
        .sum();
    let norm_a: f64 = a_locs
        .iter()
        .map(|&l| (a_counts[l] as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let norm_b: f64 = b_locs
        .iter()
        .map(|&l| (b_counts[l] as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

// Sample from a strictly-increasing CDF stored in `cdf`, returning the
// corresponding index into `candidates`.
fn cdf_sample(rng: &mut impl Rng, candidates: &[usize], cdf: &[f64]) -> usize {
    let total = cdf[cdf.len() - 1];
    let threshold = rng.gen_range(0.0_f64..1.0) * total;
    let idx = cdf
        .partition_point(|&v| v <= threshold)
        .min(candidates.len() - 1);
    candidates[idx]
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
            graph.mobility_similarity[edge_idx] = cosine_similarity_sparse(
                &agents[agent].visited_locs,
                &agents[agent].visit_counts,
                &agents[neighbor].visited_locs,
                &agents[neighbor].visit_counts,
            );
            graph.next_update_s[edge_idx] = current_ts + dt_update_s;
        }
    }
}

// ---------------------------------------------------------------------------
// Public hot-path helpers
// ---------------------------------------------------------------------------

pub(crate) fn make_social_action(
    agent: usize,
    agents: &[GeoSimAgentState],
    graph: &mut SocialGraph,
    mode: SocialMode,
    rng: &mut impl Rng,
    current_ts: i64,
    dt_update_s: i64,
    scratch: &mut GeoSimScratch,
) -> Option<usize> {
    update_edge_similarity(agent, agents, graph, current_ts, dt_update_s);

    let range = graph.edge_range(agent);
    if range.is_empty() {
        return None;
    }

    // Collect edge data into scratch without allocating.
    scratch.sims.clear();
    scratch.neighbors.clear();
    for e in range {
        scratch.sims.push(graph.mobility_similarity[e]);
        scratch.neighbors.push(graph.neighbors[e]);
    }

    // Choose contact weighted by similarity, uniform fallback when all zero.
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
    let contact = scratch.neighbors[contact_idx];

    let agent_counts = &agents[agent].visit_counts;
    let contact_counts = &agents[contact].visit_counts;
    let contact_locs = &agents[contact].visited_locs;
    let agent_locs = &agents[agent].visited_locs;

    scratch.candidates.clear();
    scratch.cdf.clear();

    match mode {
        SocialMode::Exploration => {
            // Feasible: contact's visited locations the agent hasn't been to.
            // All entries in contact_locs have count >= 1, so weight is always > 0.
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
            // Feasible: agent's visited locations the contact has also visited.
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

    Some(cdf_sample(rng, &scratch.candidates, &scratch.cdf))
}

pub(crate) fn make_individual_return(
    agent: usize,
    agents: &[GeoSimAgentState],
    rng: &mut impl Rng,
    scratch: &mut GeoSimScratch,
) -> Option<usize> {
    let state = &agents[agent];
    if state.visited_locs.is_empty() {
        return None;
    }
    scratch.candidates.clear();
    scratch.cdf.clear();
    let mut cumsum = 0.0_f64;
    for &loc in &state.visited_locs {
        scratch.candidates.push(loc);
        cumsum += state.visit_counts[loc] as f64;
        scratch.cdf.push(cumsum);
    }
    if cumsum <= 0.0 {
        return None;
    }
    Some(cdf_sample(rng, &scratch.candidates, &scratch.cdf))
}

pub(crate) fn make_individual_exploration(
    agent: usize,
    agents: &[GeoSimAgentState],
    n_locations: usize,
    rng: &mut impl Rng,
    scratch: &mut GeoSimScratch,
) -> Option<usize> {
    let counts = &agents[agent].visit_counts;
    scratch.candidates.clear();
    scratch.candidates.extend((0..n_locations).filter(|&j| counts[j] == 0));
    if scratch.candidates.is_empty() {
        return None;
    }
    Some(scratch.candidates[rng.gen_range(0..scratch.candidates.len())])
}

/// Choose location following the EPR+social decision tree with fallback chain.
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
    scratch: &mut GeoSimScratch,
) -> usize {
    let s = agents[agent].s.max(1.0);
    let p_explore = rho * s.powf(-gamma);

    let explore = rng.gen_range(0.0_f64..1.0) < p_explore;
    let social = rng.gen_range(0.0_f64..1.0) < alpha;

    let location = if explore {
        if social {
            make_social_action(agent, agents, graph, SocialMode::Exploration, rng, current_ts, dt_update_s, scratch)
                .or_else(|| make_individual_exploration(agent, agents, n_locations, rng, scratch))
                .or_else(|| make_individual_return(agent, agents, rng, scratch))
        } else {
            make_individual_exploration(agent, agents, n_locations, rng, scratch)
                .or_else(|| make_social_action(agent, agents, graph, SocialMode::Exploration, rng, current_ts, dt_update_s, scratch))
                .or_else(|| make_individual_return(agent, agents, rng, scratch))
        }
    } else if social {
        make_social_action(agent, agents, graph, SocialMode::Return, rng, current_ts, dt_update_s, scratch)
            .or_else(|| make_individual_return(agent, agents, rng, scratch))
            .or_else(|| make_individual_exploration(agent, agents, n_locations, rng, scratch))
    } else {
        make_individual_return(agent, agents, rng, scratch)
            .or_else(|| make_social_action(agent, agents, graph, SocialMode::Return, rng, current_ts, dt_update_s, scratch))
            .or_else(|| make_individual_exploration(agent, agents, n_locations, rng, scratch))
    };

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

    let mut agents: Vec<GeoSimAgentState> = (0..n_agents)
        .map(|_| GeoSimAgentState::new(n_locations, start_ts))
        .collect();

    let mut graph = SocialGraph::new(neighbor_starts.to_vec(), neighbors.to_vec(), start_ts);

    for (i, agent) in agents.iter_mut().enumerate() {
        let loc = match starting_locs {
            Some(sl) if i < sl.len() => sl[i].min(n_locations - 1),
            _ => {
                let mut rng =
                    Xoshiro256PlusPlus::seed_from_u64(derive_agent_seed(master_seed, i, 1));
                rng.gen_range(0..n_locations)
            }
        };
        agent.current_location = loc;
        agent.home_location = loc;
        agent.visit(loc);

        let mut rng =
            Xoshiro256PlusPlus::seed_from_u64(derive_agent_seed(master_seed, i, 0));
        let wait_h = sample_tpl_rng(&mut rng, xmin, alpha_param, lambda_);
        agent.time_next_move = start_ts + (wait_h * 3600.0) as i64;
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

    let mut rngs: Vec<Xoshiro256PlusPlus> = (0..n_agents)
        .map(|i| Xoshiro256PlusPlus::seed_from_u64(derive_agent_seed(master_seed, i, 0)))
        .collect();

    let mut heap: BinaryHeap<(Reverse<i64>, usize)> = BinaryHeap::new();
    for (i, agent) in agents.iter().enumerate() {
        heap.push((Reverse(agent.time_next_move), i));
    }

    let mut pending: Vec<PendingMove> = Vec::new();
    let mut scratch = GeoSimScratch::new();

    while let Some((Reverse(current_ts), agent)) = heap.peek().copied() {
        if current_ts >= end_ts {
            break;
        }
        heap.pop();

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
            &mut scratch,
        );

        pending.push(PendingMove {
            agent,
            location: loc,
            timestamp: current_ts,
        });

        let wait_h = sample_tpl_rng(&mut rngs[agent], xmin, alpha_param, lambda_);
        let next_ts = current_ts + (wait_h * 3600.0) as i64;
        agents[agent].time_next_move = next_ts;

        if next_ts < end_ts {
            heap.push((Reverse(next_ts), agent));
        }
    }

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
