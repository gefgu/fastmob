use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use crate::models::od::{validate_equal_lengths, CachedGravityOdRows};
use crate::utils::haversine::haversine_km;

type RadiationResult = Result<(Vec<usize>, Vec<usize>, Vec<f64>), String>;
type EprResult = Result<(Vec<i64>, Vec<f64>, Vec<f64>, Vec<i64>), String>;

// ---------------------------------------------------------------------------
// Radiation model
// ---------------------------------------------------------------------------

pub fn model_radiation_probabilities_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    relevances: &[f64],
    tot_outflows: &[f64],
) -> RadiationResult {
    let n = validate_equal_lengths(&[
        ("longitudes", longitudes.len()),
        ("relevances", relevances.len()),
        ("tot_outflows", tot_outflows.len()),
        ("latitudes", latitudes.len()),
    ])?;
    let total_relevance: f64 = relevances.iter().sum();

    let per_origin: Vec<Vec<(usize, usize, f64)>> = (0..n)
        .into_par_iter()
        .map(|origin| {
            if tot_outflows[origin] <= 0.0 {
                return Vec::new();
            }
            let origin_relevance = relevances[origin];
            let normalization_factor = 1.0 / (1.0 - origin_relevance / total_relevance);
            let mut destinations_and_distances: Vec<(usize, f64)> = (0..n)
                .filter(|&destination| destination != origin)
                .map(|destination| {
                    (
                        destination,
                        haversine_km(
                            latitudes[origin],
                            longitudes[origin],
                            latitudes[destination],
                            longitudes[destination],
                        ),
                    )
                })
                .collect();
            destinations_and_distances.sort_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut sum_inside = 0.0;
            let mut rows = Vec::with_capacity(n.saturating_sub(1));
            for (destination, _) in destinations_and_distances {
                let destination_relevance = relevances[destination];
                let prob = normalization_factor * (origin_relevance * destination_relevance)
                    / ((origin_relevance + sum_inside)
                        * (origin_relevance + sum_inside + destination_relevance));
                sum_inside += destination_relevance;
                rows.push((origin, destination, prob));
            }
            rows
        })
        .collect();

    let total_len: usize = per_origin.iter().map(Vec::len).sum();
    let mut origins = Vec::with_capacity(total_len);
    let mut destinations = Vec::with_capacity(total_len);
    let mut probabilities = Vec::with_capacity(total_len);
    for rows in per_origin {
        for (origin, destination, probability) in rows {
            origins.push(origin);
            destinations.push(destination);
            probabilities.push(probability);
        }
    }
    Ok((origins, destinations, probabilities))
}

// ---------------------------------------------------------------------------
// Private helpers shared by EPR kernels
// ---------------------------------------------------------------------------

fn sample_tpl_rng(rng: &mut impl Rng, xmin: f64, alpha: f64, lambda_: f64) -> f64 {
    loop {
        let u: f64 = rng.gen_range(0.0_f64..1.0);
        let x = xmin - (1.0 / lambda_) * (1.0 - u).ln();
        if rng.gen_range(0.0_f64..1.0) < (x / xmin).powf(-alpha) {
            return x;
        }
    }
}

fn cdf_choice(rng: &mut impl Rng, cdf: &[f64]) -> usize {
    let n = cdf.len();
    if n == 0 {
        return 0;
    }
    let total = cdf[n - 1];
    if total <= 0.0 || !total.is_finite() {
        return rng.gen_range(0..n);
    }
    let threshold = rng.gen_range(0.0_f64..1.0) * total;
    cdf.partition_point(|value| value.is_finite() && *value < threshold)
        .min(n - 1)
}

fn weighted_choice_excluding(
    rng: &mut impl Rng,
    visited_locs: &[usize],
    visit_counts: &[u32],
    total_visits: f64,
    exclude: usize,
) -> usize {
    let total = total_visits - visit_counts[exclude] as f64;
    if total <= 0.0 {
        return exclude;
    }
    let threshold = rng.gen_range(0.0_f64..1.0) * total;
    let mut cumsum = 0.0;
    let mut fallback = exclude;
    for &loc in visited_locs {
        if loc == exclude {
            continue;
        }
        fallback = loc;
        cumsum += visit_counts[loc] as f64;
        if cumsum > threshold {
            return loc;
        }
    }
    fallback
}

#[inline(always)]
fn derive_agent_seed(master_seed: u64, agent: usize, stream: u64) -> u64 {
    let mut value = master_seed
        ^ ((agent as u64).wrapping_add(1)).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ stream.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[inline(always)]
fn estimate_records_per_agent(start_ts: i64, end_ts: i64, xmin: f64) -> usize {
    if end_ts <= start_ts || xmin <= 0.0 || !xmin.is_finite() {
        return 2;
    }
    let duration_h = (end_ts - start_ts) as f64 / 3600.0;
    ((duration_h / xmin).ceil() as usize + 2).clamp(2, 4096)
}

fn validate_starting_locs_length(starts: Option<&[i64]>, n_agents: usize) -> Result<(), String> {
    match starts {
        Some(values) if values.len() < n_agents => {
            Err("starting_locs length must be at least n_agents".to_string())
        }
        _ => Ok(()),
    }
}

#[derive(Clone, Copy)]
pub struct EprConfig<'a> {
    pub od_rows: &'a CachedGravityOdRows<'a>,
    pub n: usize,
    pub rho: f64,
    pub gamma: f64,
    pub alpha: f64,
    pub lambda_: f64,
    pub xmin: f64,
    pub start_ts: u64,
    pub end_ts: u64,
}

struct AgentThreadState {
    visited_locs: Vec<usize>,
    visit_counts: Vec<u32>,
    total_visits: f64,
}

impl AgentThreadState {
    fn new(n: usize) -> Self {
        Self {
            visited_locs: Vec::with_capacity(300),
            visit_counts: vec![0u32; n],
            total_visits: 0.0,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_one_epr_agent(
    agent_id: usize,
    starting_loc: usize,
    seed: u64,
    config: EprConfig,
    lats: &[f64],
    lons: &[f64],
    out_agents: &mut Vec<i64>,
    out_ts: &mut Vec<i64>,
    out_lats: &mut Vec<f64>,
    out_lons: &mut Vec<f64>,
    visited_locs: &mut Vec<usize>,
    visit_counts: &mut [u32],
    total_visits: &mut f64,
) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let mut cur_loc = starting_loc.min(config.n.saturating_sub(1));
    let mut cur_ts = config.start_ts;
    out_agents.push(agent_id as i64);
    out_ts.push(cur_ts as i64);
    out_lats.push(lats[cur_loc]);
    out_lons.push(lons[cur_loc]);
    record_visits(visited_locs, visit_counts, total_visits, cur_loc);

    let wait_h = sample_tpl_rng(&mut rng, config.xmin, config.alpha, config.lambda_);
    cur_ts += (wait_h * 3600.0) as u64;

    while cur_ts < config.end_ts {
        let n_visited = visited_locs.len();
        let p_new: f64 = rng.gen_range(0.0_f64..1.0);
        let explore = n_visited == 1
            || (n_visited < config.n
                && p_new <= config.rho * (n_visited as f64).powf(-config.gamma));
        let next_loc = if explore {
            let row = config.od_rows.get(cur_loc);
            cdf_choice(&mut rng, &row)
        } else {
            weighted_choice_excluding(&mut rng, visited_locs, visit_counts, *total_visits, cur_loc)
        };
        cur_loc = next_loc;
        out_agents.push(agent_id as i64);
        out_ts.push(cur_ts as i64);
        out_lats.push(lats[cur_loc]);
        out_lons.push(lons[cur_loc]);
        record_visits(visited_locs, visit_counts, total_visits, cur_loc);
        let wait_h = sample_tpl_rng(&mut rng, config.xmin, config.alpha, config.lambda_);
        cur_ts += (wait_h * 3600.0) as u64;
    }
}

#[inline(always)]
fn record_visits(
    visited_locs: &mut Vec<usize>,
    visit_counts: &mut [u32],
    total_visits: &mut f64,
    loc: usize,
) {
    if visit_counts[loc] == 0 {
        visited_locs.push(loc);
    }
    visit_counts[loc] += 1;
    *total_visits += 1.0;
}

#[inline(always)]
fn clear_visits(
    visited_locs: &mut Vec<usize>,
    visit_counts: &mut [u32],
    total_visits: &mut f64,
) {
    for &loc in visited_locs.iter() {
        visit_counts[loc] = 0;
    }
    visited_locs.clear();
    *total_visits = 0.0;
}

// ---------------------------------------------------------------------------
// EPR simulation (single driver, cached OD via gravity model)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn simulate_epr_agents_from_cached_od_impl(
    lats: &[f64],
    lons: &[f64],
    rels: &[f64],
    deterrence_type: &str,
    deterrence_arg: f64,
    origin_exp: f64,
    destination_exp: f64,
    rho: f64,
    gamma: f64,
    beta: f64,
    tau: f64,
    xmin: f64,
    start_ts: i64,
    end_ts: i64,
    n_agents: usize,
    master_seed: Option<u64>,
    starting_locs_slice: Option<&[i64]>,
) -> EprResult {
    let n = validate_equal_lengths(&[
        ("longitudes", lons.len()),
        ("relevances", rels.len()),
        ("latitudes", lats.len()),
    ])?;
    validate_starting_locs_length(starting_locs_slice, n_agents)?;
    if n == 0 || n_agents == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    }
    let master_seed = master_seed.unwrap_or_else(rand::random);

    let od_rows = CachedGravityOdRows::new(
        lats,
        lons,
        rels,
        deterrence_type,
        deterrence_arg,
        origin_exp,
        destination_exp,
    );
    let records_per_agent = estimate_records_per_agent(start_ts, end_ts, xmin);

    let epr_config = EprConfig {
        od_rows: &od_rows,
        n,
        rho,
        gamma,
        alpha: 1.0 + beta,
        lambda_: 1.0 / tau,
        xmin,
        start_ts: start_ts as u64,
        end_ts: end_ts as u64,
    };

    let agent_trajectories: Vec<(Vec<i64>, Vec<f64>, Vec<f64>, Vec<i64>)> = (0..n_agents)
        .into_par_iter()
        .map_init(
            || AgentThreadState::new(n),
            |state, agent| {
                let seed = derive_agent_seed(master_seed, agent, 0);
                let sl = match starting_locs_slice {
                    Some(starts) => (starts[agent].max(0) as usize).min(n - 1),
                    None => {
                        let mut start_rng = Xoshiro256PlusPlus::seed_from_u64(
                            derive_agent_seed(master_seed, agent, 1),
                        );
                        start_rng.gen_range(0..n)
                    }
                };
                let mut out_agents = Vec::with_capacity(records_per_agent);
                let mut out_lats = Vec::with_capacity(records_per_agent);
                let mut out_lons = Vec::with_capacity(records_per_agent);
                let mut out_ts = Vec::with_capacity(records_per_agent);
                simulate_one_epr_agent(
                    agent + 1,
                    sl,
                    seed,
                    epr_config,
                    lats,
                    lons,
                    &mut out_agents,
                    &mut out_ts,
                    &mut out_lats,
                    &mut out_lons,
                    &mut state.visited_locs,
                    &mut state.visit_counts,
                    &mut state.total_visits,
                );
                clear_visits(
                    &mut state.visited_locs,
                    &mut state.visit_counts,
                    &mut state.total_visits,
                );
                (out_agents, out_lats, out_lons, out_ts)
            },
        )
        .collect();

    let total: usize = agent_trajectories
        .iter()
        .map(|(a, _, _, _)| a.len())
        .sum();
    let mut out_agents: Vec<i64> = Vec::with_capacity(total);
    let mut out_lats: Vec<f64> = Vec::with_capacity(total);
    let mut out_lons_: Vec<f64> = Vec::with_capacity(total);
    let mut out_ts: Vec<i64> = Vec::with_capacity(total);

    for (agents, lats_chunk, lons_chunk, ts) in agent_trajectories {
        out_agents.extend_from_slice(&agents);
        out_lats.extend_from_slice(&lats_chunk);
        out_lons_.extend_from_slice(&lons_chunk);
        out_ts.extend_from_slice(&ts);
    }

    Ok((out_agents, out_lats, out_lons_, out_ts))
}

// ---------------------------------------------------------------------------
// Truncated power law batch sampler
// ---------------------------------------------------------------------------

pub fn model_truncated_power_law_samples(
    xmin: f64,
    alpha: f64,
    lambda_: f64,
    n: usize,
    seed: u64,
) -> Vec<f64> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut result = Vec::with_capacity(n);
    while result.len() < n {
        let u: f64 = rng.gen_range(0.0_f64..1.0);
        let x = xmin - (1.0 / lambda_) * (1.0 - u).ln();
        if rng.gen_range(0.0_f64..1.0) < (x / xmin).powf(-alpha) {
            result.push(x);
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Distance matrix
// ---------------------------------------------------------------------------

pub fn model_distance_matrix_impl(
    latitudes: &[f64],
    longitudes: &[f64],
) -> Result<Vec<f64>, String> {
    let n = validate_equal_lengths(&[
        ("longitudes", longitudes.len()),
        ("latitudes", latitudes.len()),
    ])?;
    let flat: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            (0..n).map(move |j| {
                if i == j {
                    0.0
                } else {
                    haversine_km(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
                }
            })
        })
        .collect();
    Ok(flat)
}
