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

fn weighted_choice_slice(rng: &mut impl Rng, weights: &[f64]) -> usize {
    let n = weights.len();
    if n == 0 {
        return 0;
    }
    let total: f64 = weights.iter().sum();
    if total == 0.0 {
        return rng.gen_range(0..n);
    }
    let threshold = rng.gen_range(0.0_f64..1.0) * total;
    let mut cumsum = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        cumsum += w;
        if cumsum > threshold {
            return i;
        }
    }
    n - 1
}

fn weighted_choice_excluding(rng: &mut impl Rng, visits: &[(usize, u32)], exclude: usize) -> usize {
    let mut pairs: Vec<(usize, f64)> = Vec::new();
    for (loc, cnt) in visits.iter() {
        if *loc != exclude {
            pairs.push((*loc, *cnt as f64));
        }
    }
    if pairs.is_empty() {
        return exclude;
    }
    let total: f64 = pairs.iter().map(|(_, w)| w).sum();
    let threshold = rng.gen_range(0.0_f64..1.0) * total;
    let mut cumsum = 0.0;
    for &(loc, w) in &pairs {
        cumsum += w;
        if cumsum > threshold {
            return loc;
        }
    }
    pairs.last().unwrap().0
}

#[derive(Clone, Copy)]
pub enum EprOdRows<'a> {
    Dense(&'a [Vec<f64>]),
    Cached(&'a CachedGravityOdRows<'a>),
}

#[derive(Clone, Copy)]
pub struct EprConfig<'a> {
    pub od_rows: EprOdRows<'a>,
    pub n: usize,
    pub rho: f64,
    pub gamma: f64,
    pub alpha: f64,
    pub lambda_: f64,
    pub xmin: f64,
    pub start_ts: u64,
    pub end_ts: u64,
}

#[allow(clippy::too_many_arguments)]
pub fn simulate_one_epr_agent(
    agent_id: usize,
    starting_loc: usize,
    seed: u64,
    config: EprConfig,
    out_agents: &mut Vec<usize>,
    out_timestamps: &mut Vec<u64>,
    out_loc_indices: &mut Vec<usize>,
    visit_cache: &mut Vec<(usize, u32)>,
) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let mut cur_loc = starting_loc.min(config.n.saturating_sub(1));
    let mut cur_ts = config.start_ts;
    out_agents.push(agent_id);
    out_timestamps.push(cur_ts);
    out_loc_indices.push(cur_loc);
    record_visits(visit_cache, cur_loc);

    let wait_h = sample_tpl_rng(&mut rng, config.xmin, config.alpha, config.lambda_);
    cur_ts += (wait_h * 3600.0) as u64;

    while cur_ts < config.end_ts {
        let n_visited = visit_cache.len();
        let p_new: f64 = rng.gen_range(0.0_f64..1.0);
        let explore = n_visited == 1
            || (n_visited < config.n
                && p_new <= config.rho * (n_visited as f64).powf(-config.gamma));
        let next_loc = if explore {
            match config.od_rows {
                EprOdRows::Dense(rows) => weighted_choice_slice(&mut rng, &rows[cur_loc]),
                EprOdRows::Cached(rows) => {
                    let row = rows.get(cur_loc);
                    weighted_choice_slice(&mut rng, &row)
                }
            }
        } else {
            weighted_choice_excluding(&mut rng, visit_cache, cur_loc)
        };
        cur_loc = next_loc;
        out_agents.push(agent_id);
        out_timestamps.push(cur_ts);
        out_loc_indices.push(cur_loc);
        record_visits(visit_cache, cur_loc);
        let wait_h = sample_tpl_rng(&mut rng, config.xmin, config.alpha, config.lambda_);
        cur_ts += (wait_h * 3600.0) as u64;
    }
}

#[inline(always)]
fn record_visits(visits: &mut Vec<(usize, u32)>, loc: usize) {
    for (l, cnt) in visits.iter_mut() {
        if *l == loc {
            *cnt += 1;
            return;
        }
    }
    visits.push((loc, 1));
}

// ---------------------------------------------------------------------------
// EPR simulation from dense OD matrix
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn simulate_epr_agents_from_od_impl(
    lats: &[f64],
    lons: &[f64],
    od_rows: Vec<Vec<f64>>,
    rho: f64,
    gamma: f64,
    beta: f64,
    tau: f64,
    xmin: f64,
    start_ts: i64,
    end_ts: i64,
    seeds_slice: &[i64],
    starting_locs_slice: &[i64],
) -> EprResult {
    let n = lats.len();
    let n_agents = seeds_slice.len();
    if starting_locs_slice.len() != n_agents {
        return Err("starting_locs length must equal seeds length".to_string());
    }
    if lons.len() != n {
        return Err("longitudes must have the same length as latitudes".to_string());
    }
    if od_rows.len() != n || od_rows.iter().any(|row| row.len() != n) {
        return Err("od_matrix must be square with one row per coordinate".to_string());
    }
    if n == 0 || n_agents == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    }

    let worker_count = rayon::current_num_threads().max(1);
    let target_chunks = (worker_count * 4).min(n_agents).max(1);
    let chunk_size = n_agents.div_ceil(target_chunks);
    let agent_chunks: Vec<_> = (0..n_agents)
        .step_by(chunk_size)
        .map(|start| start..(start + chunk_size).min(n_agents))
        .collect();

    let epr_config = EprConfig {
        od_rows: EprOdRows::Dense(&od_rows),
        n,
        rho,
        gamma,
        alpha: 1.0 + beta,
        lambda_: 1.0 / tau,
        xmin,
        start_ts: start_ts as u64,
        end_ts: end_ts as u64,
    };

    let agent_results = agent_chunks
        .into_par_iter()
        .map(|chunk| {
            let mut out_agents = Vec::with_capacity(chunk.len() * 50);
            let mut out_timestamps = Vec::with_capacity(chunk.len() * 50);
            let mut out_loc_indices = Vec::with_capacity(chunk.len() * 50);
            let mut visit_cache = Vec::with_capacity(300);

            for agent in chunk {
                let sl = (starting_locs_slice[agent].max(0) as usize).min(n - 1);
                let seed = seeds_slice[agent] as u64;
                simulate_one_epr_agent(
                    agent + 1,
                    sl,
                    seed,
                    epr_config,
                    &mut out_agents,
                    &mut out_timestamps,
                    &mut out_loc_indices,
                    &mut visit_cache,
                );

                visit_cache.clear();
            }

            (out_agents, out_timestamps, out_loc_indices)
        })
        .collect::<Vec<_>>();

    let total: usize = agent_results
        .iter()
        .map(|(agents, _, _)| agents.len())
        .sum();
    let mut out_agents: Vec<i64> = Vec::with_capacity(total);
    let mut out_lats: Vec<f64> = Vec::with_capacity(total);
    let mut out_lons_: Vec<f64> = Vec::with_capacity(total);
    let mut out_ts: Vec<i64> = Vec::with_capacity(total);

    for (chunk_agents, timestamps, loc_indices) in agent_results {
        for ((agent_id, ts), loc) in chunk_agents.into_iter().zip(timestamps).zip(loc_indices) {
            out_agents.push(agent_id as i64);
            out_lats.push(lats[loc]);
            out_lons_.push(lons[loc]);
            out_ts.push(ts as i64);
        }
    }

    Ok((out_agents, out_lats, out_lons_, out_ts))
}

// ---------------------------------------------------------------------------
// EPR simulation from cached OD (gravity model)
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
    seeds_slice: &[i64],
    starting_locs_slice: &[i64],
) -> EprResult {
    let n = validate_equal_lengths(&[
        ("longitudes", lons.len()),
        ("relevances", rels.len()),
        ("latitudes", lats.len()),
    ])?;
    let n_agents = seeds_slice.len();
    if starting_locs_slice.len() != n_agents {
        return Err("starting_locs length must equal seeds length".to_string());
    }
    if n == 0 || n_agents == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new(), Vec::new()));
    }

    let od_rows = CachedGravityOdRows::new(
        lats,
        lons,
        rels,
        deterrence_type,
        deterrence_arg,
        origin_exp,
        destination_exp,
    );
    let worker_count = rayon::current_num_threads().max(1);
    let target_chunks = (worker_count * 4).min(n_agents).max(1);
    let chunk_size = n_agents.div_ceil(target_chunks);
    let agent_chunks: Vec<_> = (0..n_agents)
        .step_by(chunk_size)
        .map(|start| start..(start + chunk_size).min(n_agents))
        .collect();

    let epr_config = EprConfig {
        od_rows: EprOdRows::Cached(&od_rows),
        n,
        rho,
        gamma,
        alpha: 1.0 + beta,
        lambda_: 1.0 / tau,
        xmin,
        start_ts: start_ts as u64,
        end_ts: end_ts as u64,
    };

    let agent_results = agent_chunks
        .into_par_iter()
        .map(|chunk| {
            let mut out_agents = Vec::with_capacity(chunk.len() * 50);
            let mut out_timestamps = Vec::with_capacity(chunk.len() * 50);
            let mut out_loc_indices = Vec::with_capacity(chunk.len() * 50);
            let mut visit_cache = Vec::with_capacity(300);

            for agent in chunk {
                let sl = (starting_locs_slice[agent].max(0) as usize).min(n - 1);
                let seed = seeds_slice[agent] as u64;
                simulate_one_epr_agent(
                    agent + 1,
                    sl,
                    seed,
                    epr_config,
                    &mut out_agents,
                    &mut out_timestamps,
                    &mut out_loc_indices,
                    &mut visit_cache,
                );

                visit_cache.clear();
            }

            (out_agents, out_timestamps, out_loc_indices)
        })
        .collect::<Vec<_>>();

    let total: usize = agent_results
        .iter()
        .map(|(agents, _, _)| agents.len())
        .sum();
    let mut out_agents: Vec<i64> = Vec::with_capacity(total);
    let mut out_lats: Vec<f64> = Vec::with_capacity(total);
    let mut out_lons_: Vec<f64> = Vec::with_capacity(total);
    let mut out_ts: Vec<i64> = Vec::with_capacity(total);

    for (chunk_agents, timestamps, loc_indices) in agent_results {
        for ((agent_id, ts), loc) in chunk_agents.into_iter().zip(timestamps).zip(loc_indices) {
            out_agents.push(agent_id as i64);
            out_lats.push(lats[loc]);
            out_lons_.push(lons[loc]);
            out_ts.push(ts as i64);
        }
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

pub fn model_distance_matrix_impl(latitudes: &[f64], longitudes: &[f64]) -> Result<Vec<f64>, String> {
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
