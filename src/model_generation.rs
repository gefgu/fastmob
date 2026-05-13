use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use std::collections::HashMap;

const EARTH_RADIUS_KM: f64 = 6371.01;

fn validate_equal_lengths(arrays: &[(&str, usize)]) -> PyResult<usize> {
    let Some((_, n)) = arrays.first() else {
        return Ok(0);
    };
    for (name, len) in arrays {
        if len != n {
            return Err(PyValueError::new_err(format!(
                "{name} must have the same length as the coordinate arrays"
            )));
        }
    }
    Ok(*n)
}

fn haversine_km_manual(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let lat1 = lat1.to_radians();
    let lon1 = lon1.to_radians();
    let lat2 = lat2.to_radians();
    let lon2 = lon2.to_radians();
    let dlat = lat1 - lat2;
    let dlon = lon1 - lon2;
    let ds = 2.0
        * ((dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2))
            .sqrt()
            .asin();
    EARTH_RADIUS_KM * ds
}

fn deterrence(distance: f64, deterrence_type: &str, arg: f64) -> f64 {
    if deterrence_type == "exponential" {
        (-distance * arg).exp()
    } else {
        distance.powf(arg)
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (latitudes, longitudes, relevances, tot_outflows, deterrence_type, deterrence_arg, origin_exp, destination_exp, gravity_type, out_format))]
pub(crate) fn model_gravity_matrix_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    relevances: PyReadonlyArray1<'py, f64>,
    tot_outflows: PyReadonlyArray1<'py, f64>,
    deterrence_type: &str,
    deterrence_arg: f64,
    origin_exp: f64,
    destination_exp: f64,
    gravity_type: &str,
    out_format: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let latitudes = latitudes.as_slice()?;
    let longitudes = longitudes.as_slice()?;
    let relevances = relevances.as_slice()?;
    let tot_outflows = tot_outflows.as_slice()?;
    let n = validate_equal_lengths(&[
        ("longitudes", longitudes.len()),
        ("relevances", relevances.len()),
        ("tot_outflows", tot_outflows.len()),
        ("latitudes", latitudes.len()),
    ])?;

    let mut matrix: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            (0..n).map(move |j| {
                if i == j {
                    return 0.0;
                }
                let distance =
                    haversine_km_manual(latitudes[i], longitudes[i], latitudes[j], longitudes[j]);
                let score = deterrence(distance, deterrence_type, deterrence_arg)
                    * relevances[j].powf(destination_exp)
                    * relevances[i].powf(origin_exp);
                if score.is_finite() {
                    score
                } else {
                    0.0
                }
            })
        })
        .collect();

    if gravity_type == "globally constrained" {
        let total: f64 = matrix.iter().sum();
        if total != 0.0 {
            for value in &mut matrix {
                *value /= total;
            }
        }
        if out_format == "flows" {
            let total_outflow: f64 = tot_outflows.iter().sum();
            for value in &mut matrix {
                *value *= total_outflow;
            }
        }
    } else {
        matrix.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            let row_sum: f64 = row.iter().sum();
            if row_sum != 0.0 {
                for value in row.iter_mut() {
                    *value /= row_sum;
                    if out_format == "flows" {
                        *value *= tot_outflows[i];
                    }
                }
            } else {
                for value in row.iter_mut() {
                    *value = 0.0;
                }
            }
        });
    }

    Ok(matrix.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn model_gravity_od_row_numpy<'py>(
    py: Python<'py>,
    origin: usize,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    relevances: PyReadonlyArray1<'py, f64>,
    deterrence_type: &str,
    deterrence_arg: f64,
    origin_exp: f64,
    destination_exp: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let latitudes = latitudes.as_slice()?;
    let longitudes = longitudes.as_slice()?;
    let relevances = relevances.as_slice()?;
    let n = validate_equal_lengths(&[
        ("longitudes", longitudes.len()),
        ("relevances", relevances.len()),
        ("latitudes", latitudes.len()),
    ])?;
    if origin >= n {
        return Err(PyValueError::new_err(
            "origin must be within coordinate bounds",
        ));
    }

    let mut row: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|j| {
            if j == origin {
                return 0.0;
            }
            let distance = haversine_km_manual(
                latitudes[origin],
                longitudes[origin],
                latitudes[j],
                longitudes[j],
            );
            let score = deterrence(distance, deterrence_type, deterrence_arg)
                * relevances[j].powf(destination_exp)
                * relevances[origin].powf(origin_exp);
            if score.is_finite() {
                score
            } else {
                0.0
            }
        })
        .collect();
    let total: f64 = row.iter().sum();
    if total != 0.0 {
        for value in &mut row {
            *value /= total;
        }
    } else if n > 0 {
        let uniform = 1.0 / n as f64;
        row.fill(uniform);
    }
    Ok(row.into_pyarray(py))
}

#[pyfunction]
pub(crate) fn model_radiation_probabilities(
    latitudes: PyReadonlyArray1<f64>,
    longitudes: PyReadonlyArray1<f64>,
    relevances: PyReadonlyArray1<f64>,
    tot_outflows: PyReadonlyArray1<f64>,
) -> PyResult<(Vec<usize>, Vec<usize>, Vec<f64>)> {
    let latitudes = latitudes.as_slice()?;
    let longitudes = longitudes.as_slice()?;
    let relevances = relevances.as_slice()?;
    let tot_outflows = tot_outflows.as_slice()?;
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
                        haversine_km_manual(
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
// Private helpers shared by new kernels
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

// Sequential (non-Rayon) OD row — used inside the parallel agent loop to avoid
// nested parallelism contention when outer par_iter already owns all threads.
#[allow(clippy::too_many_arguments)]
fn gravity_od_row_seq(
    origin: usize,
    lats: &[f64],
    lons: &[f64],
    rels: &[f64],
    deterrence_type: &str,
    deterrence_arg: f64,
    origin_exp: f64,
    dest_exp: f64,
) -> Vec<f64> {
    let n = lats.len();
    let mut row: Vec<f64> = (0..n)
        .map(|j| {
            if j == origin {
                return 0.0;
            }
            let d = haversine_km_manual(lats[origin], lons[origin], lats[j], lons[j]);
            let s = deterrence(d, deterrence_type, deterrence_arg)
                * rels[j].powf(dest_exp)
                * rels[origin].powf(origin_exp);
            if s.is_finite() { s } else { 0.0 }
        })
        .collect();
    let total: f64 = row.iter().sum();
    if total != 0.0 {
        row.iter_mut().for_each(|v| *v /= total);
    } else if n > 0 {
        row.fill(1.0 / n as f64);
    }
    row
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

fn weighted_choice_excluding(
    rng: &mut impl Rng,
    visits: &HashMap<usize, usize>,
    exclude: usize,
) -> usize {
    let mut pairs: Vec<(usize, f64)> = Vec::new();
    for (&loc, &cnt) in visits.iter() {
        if loc != exclude {
            pairs.push((loc, cnt as f64));
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

#[allow(clippy::too_many_arguments)]
fn simulate_one_epr_agent(
    starting_loc: usize,
    od_rows: &[Vec<f64>],
    n: usize,
    rho: f64,
    gamma: f64,
    beta: f64,
    tau: f64,
    xmin: f64,
    start_ts: i64,
    end_ts: i64,
    seed: u64,
) -> (Vec<i64>, Vec<usize>) {
    let alpha = 1.0 + beta;
    let lambda_ = 1.0 / tau;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut timestamps: Vec<i64> = Vec::new();
    let mut loc_indices: Vec<usize> = Vec::new();
    let mut visits: HashMap<usize, usize> = HashMap::new();

    let mut cur_loc = starting_loc.min(n.saturating_sub(1));
    let mut cur_ts = start_ts;
    timestamps.push(cur_ts);
    loc_indices.push(cur_loc);
    *visits.entry(cur_loc).or_insert(0) += 1;

    let wait_h = sample_tpl_rng(&mut rng, xmin, alpha, lambda_);
    cur_ts += (wait_h * 3600.0) as i64;

    while cur_ts < end_ts {
        let n_visited = visits.len();
        let p_new: f64 = rng.gen_range(0.0_f64..1.0);
        let explore = n_visited == 1
            || (n_visited < n && p_new <= rho * (n_visited as f64).powf(-gamma));
        let next_loc = if explore {
            weighted_choice_slice(&mut rng, &od_rows[cur_loc])
        } else {
            weighted_choice_excluding(&mut rng, &visits, cur_loc)
        };
        cur_loc = next_loc;
        timestamps.push(cur_ts);
        loc_indices.push(cur_loc);
        *visits.entry(cur_loc).or_insert(0) += 1;
        let wait_h = sample_tpl_rng(&mut rng, xmin, alpha, lambda_);
        cur_ts += (wait_h * 3600.0) as i64;
    }
    (timestamps, loc_indices)
}

// ---------------------------------------------------------------------------
// Iteration 2: Truncated power law batch sampler
// ---------------------------------------------------------------------------

#[pyfunction]
pub(crate) fn model_truncated_power_law_samples(
    xmin: f64,
    alpha: f64,
    lambda_: f64,
    n: usize,
    seed: u64,
) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
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
// Iteration 3: Full distance matrix (Rayon-parallel, for STS_epr init)
// ---------------------------------------------------------------------------

#[pyfunction]
pub(crate) fn model_distance_matrix_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let lats = latitudes.as_slice()?;
    let lons = longitudes.as_slice()?;
    let n = validate_equal_lengths(&[("longitudes", lons.len()), ("latitudes", lats.len())])?;
    let flat: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            (0..n).map(move |j| {
                if i == j {
                    0.0
                } else {
                    haversine_km_manual(lats[i], lons[i], lats[j], lons[j])
                }
            })
        })
        .collect();
    Ok(flat.into_pyarray(py))
}

// ---------------------------------------------------------------------------
// Iteration 4: Parallel EPR agent simulation
// ---------------------------------------------------------------------------

#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyo3(signature = (
    latitudes, longitudes, relevances,
    rho, gamma, beta, tau, xmin,
    start_ts, end_ts,
    seeds, starting_locs,
    deterrence_type, deterrence_arg, origin_exp, destination_exp
))]
pub(crate) fn model_epr_simulate_agents<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    relevances: PyReadonlyArray1<'py, f64>,
    rho: f64,
    gamma: f64,
    beta: f64,
    tau: f64,
    xmin: f64,
    start_ts: i64,
    end_ts: i64,
    seeds: PyReadonlyArray1<'py, i64>,
    starting_locs: PyReadonlyArray1<'py, i64>,
    deterrence_type: &str,
    deterrence_arg: f64,
    origin_exp: f64,
    destination_exp: f64,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let lats = latitudes.as_slice()?;
    let lons = longitudes.as_slice()?;
    let rels = relevances.as_slice()?;
    let seeds_slice = seeds.as_slice()?;
    let starting_locs_slice = starting_locs.as_slice()?;
    let n = validate_equal_lengths(&[
        ("longitudes", lons.len()),
        ("relevances", rels.len()),
        ("latitudes", lats.len()),
    ])?;
    let n_agents = seeds_slice.len();
    if starting_locs_slice.len() != n_agents {
        return Err(PyValueError::new_err(
            "starting_locs length must equal seeds length",
        ));
    }
    if n == 0 || n_agents == 0 {
        let empty_i64 = Vec::<i64>::new().into_pyarray(py);
        let empty_f64 = Vec::<f64>::new().into_pyarray(py);
        return Ok((empty_i64, empty_f64.clone(), empty_f64, Vec::<i64>::new().into_pyarray(py)));
    }

    // Pre-compute all N OD rows in parallel (outer Rayon), sequential inner per row.
    let od_rows: Vec<Vec<f64>> = (0..n)
        .into_par_iter()
        .map(|i| {
            gravity_od_row_seq(
                i,
                lats,
                lons,
                rels,
                deterrence_type,
                deterrence_arg,
                origin_exp,
                destination_exp,
            )
        })
        .collect();

    // Simulate all agents in parallel; each agent owns its own RNG.
    let agent_results: Vec<(Vec<i64>, Vec<usize>)> = (0..n_agents)
        .into_par_iter()
        .map(|i| {
            let sl = (starting_locs_slice[i].max(0) as usize).min(n - 1);
            let seed = seeds_slice[i] as u64;
            simulate_one_epr_agent(
                sl, &od_rows, n, rho, gamma, beta, tau, xmin, start_ts, end_ts, seed,
            )
        })
        .collect();

    // Flatten into parallel output arrays.
    let total: usize = agent_results.iter().map(|(ts, _)| ts.len()).sum();
    let mut out_agents: Vec<i64> = Vec::with_capacity(total);
    let mut out_lats: Vec<f64> = Vec::with_capacity(total);
    let mut out_lons_: Vec<f64> = Vec::with_capacity(total);
    let mut out_ts: Vec<i64> = Vec::with_capacity(total);

    for (agent_idx, (timestamps, loc_indices)) in agent_results.into_iter().enumerate() {
        let agent_id = (agent_idx + 1) as i64;
        for (ts, loc) in timestamps.into_iter().zip(loc_indices) {
            out_agents.push(agent_id);
            out_lats.push(lats[loc]);
            out_lons_.push(lons[loc]);
            out_ts.push(ts);
        }
    }

    Ok((
        out_agents.into_pyarray(py),
        out_lats.into_pyarray(py),
        out_lons_.into_pyarray(py),
        out_ts.into_pyarray(py),
    ))
}
