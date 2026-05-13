use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

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
