use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use skmob2_core::models::model_generation::{
    model_distance_matrix_impl, model_radiation_probabilities_impl,
    model_truncated_power_law_samples as core_model_truncated_power_law_samples,
    simulate_epr_agents_from_cached_od_impl, simulate_epr_agents_from_od_impl,
};
use skmob2_core::models::od::validate_equal_lengths;

#[pyfunction]
pub fn model_radiation_probabilities(
    latitudes: PyReadonlyArray1<f64>,
    longitudes: PyReadonlyArray1<f64>,
    relevances: PyReadonlyArray1<f64>,
    tot_outflows: PyReadonlyArray1<f64>,
) -> PyResult<(Vec<usize>, Vec<usize>, Vec<f64>)> {
    model_radiation_probabilities_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        relevances.as_slice()?,
        tot_outflows.as_slice()?,
    )
    .map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn model_truncated_power_law_samples(
    xmin: f64,
    alpha: f64,
    lambda_: f64,
    n: usize,
    seed: u64,
) -> Vec<f64> {
    core_model_truncated_power_law_samples(xmin, alpha, lambda_, n, seed)
}

#[pyfunction]
pub fn model_distance_matrix_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let lats = latitudes.as_slice()?;
    let lons = longitudes.as_slice()?;
    let flat = model_distance_matrix_impl(lats, lons).map_err(PyValueError::new_err)?;
    Ok(flat.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyo3(signature = (
    latitudes, longitudes, relevances,
    rho, gamma, beta, tau, xmin,
    start_ts, end_ts,
    seeds, starting_locs,
    deterrence_type, deterrence_arg, origin_exp, destination_exp
))]
pub fn model_epr_simulate_agents<'py>(
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
    let (out_agents, out_lats, out_lons, out_ts) = simulate_epr_agents_from_cached_od_impl(
        lats,
        lons,
        rels,
        deterrence_type,
        deterrence_arg,
        origin_exp,
        destination_exp,
        rho,
        gamma,
        beta,
        tau,
        xmin,
        start_ts,
        end_ts,
        seeds_slice,
        starting_locs_slice,
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        out_agents.into_pyarray(py),
        out_lats.into_pyarray(py),
        out_lons.into_pyarray(py),
        out_ts.into_pyarray(py),
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyo3(signature = (
    latitudes, longitudes, od_matrix,
    rho, gamma, beta, tau, xmin,
    start_ts, end_ts,
    seeds, starting_locs
))]
pub fn model_epr_simulate_agents_from_od<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    od_matrix: PyReadonlyArray1<'py, f64>,
    rho: f64,
    gamma: f64,
    beta: f64,
    tau: f64,
    xmin: f64,
    start_ts: i64,
    end_ts: i64,
    seeds: PyReadonlyArray1<'py, i64>,
    starting_locs: PyReadonlyArray1<'py, i64>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let lats = latitudes.as_slice()?;
    let lons = longitudes.as_slice()?;
    let od = od_matrix.as_slice()?;
    let seeds_slice = seeds.as_slice()?;
    let starting_locs_slice = starting_locs.as_slice()?;
    let n = validate_equal_lengths(&[("longitudes", lons.len()), ("latitudes", lats.len())])
        .map_err(PyValueError::new_err)?;
    if od.len() != n * n {
        return Err(PyValueError::new_err(
            "od_matrix must contain n_locations * n_locations values",
        ));
    }
    let od_rows = od.chunks(n).map(|row| row.to_vec()).collect();
    let (out_agents, out_lats, out_lons, out_ts) = simulate_epr_agents_from_od_impl(
        lats,
        lons,
        od_rows,
        rho,
        gamma,
        beta,
        tau,
        xmin,
        start_ts,
        end_ts,
        seeds_slice,
        starting_locs_slice,
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        out_agents.into_pyarray(py),
        out_lats.into_pyarray(py),
        out_lons.into_pyarray(py),
        out_ts.into_pyarray(py),
    ))
}
