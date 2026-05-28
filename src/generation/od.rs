use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::generation::model_generation::validate_equal_lengths;
use crate::haversine::haversine_km;

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
                    haversine_km(latitudes[i], longitudes[i], latitudes[j], longitudes[j]);
                let score = deterrence(distance, deterrence_type, deterrence_arg)
                    * relevances[j].powf(destination_exp)
                    * relevances[i].powf(origin_exp);
                if score.is_finite() { score } else { 0.0 }
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
            let distance = haversine_km(
                latitudes[origin],
                longitudes[origin],
                latitudes[j],
                longitudes[j],
            );
            let score = deterrence(distance, deterrence_type, deterrence_arg)
                * relevances[j].powf(destination_exp)
                * relevances[origin].powf(origin_exp);
            if score.is_finite() { score } else { 0.0 }
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

// Sequential (non-Rayon) OD row: used inside the parallel agent loop to avoid
// nested parallelism contention when outer par_iter already owns all threads.
#[allow(clippy::too_many_arguments)]
pub(crate) fn gravity_od_row_seq(
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
            let d = haversine_km(lats[origin], lons[origin], lats[j], lons[j]);
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
