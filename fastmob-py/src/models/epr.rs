use fastmob_core::models::epr::{
    model_truncated_power_law_samples as core_model_truncated_power_law_samples,
    simulate_epr_agents_from_cached_od_impl,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_i64_values, arrow_values, as_f64_array, as_i64_array, f64_results_into_arrow,
    i64_results_into_arrow,
};

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
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
#[pyo3(signature = (
    latitudes, longitudes, relevances,
    rho, gamma, beta, tau, xmin,
    start_ts, end_ts,
    deterrence_type, deterrence_arg, origin_exp, destination_exp,
    n_agents, random_state=None, starting_locs=None
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
    deterrence_type: &str,
    deterrence_arg: f64,
    origin_exp: f64,
    destination_exp: f64,
    n_agents: usize,
    random_state: Option<u64>,
    starting_locs: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let lats = latitudes.as_slice()?;
    let lons = longitudes.as_slice()?;
    let rels = relevances.as_slice()?;
    let starting_locs_slice = match &starting_locs {
        Some(values) => Some(values.as_slice()?),
        None => None,
    };
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
        n_agents,
        random_state,
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
    latitudes, longitudes, relevances,
    rho, gamma, beta, tau, xmin,
    start_ts, end_ts,
    deterrence_type, deterrence_arg, origin_exp, destination_exp,
    n_agents, random_state=None, starting_locs=None
))]
pub fn model_epr_simulate_agents_arrow<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    relevances: ArrowPyArray,
    rho: f64,
    gamma: f64,
    beta: f64,
    tau: f64,
    xmin: f64,
    start_ts: i64,
    end_ts: i64,
    deterrence_type: &str,
    deterrence_arg: f64,
    origin_exp: f64,
    destination_exp: f64,
    n_agents: usize,
    random_state: Option<u64>,
    starting_locs: Option<ArrowPyArray>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let lats = as_f64_array(latitudes, "latitudes")?;
    let lons = as_f64_array(longitudes, "longitudes")?;
    let rels = as_f64_array(relevances, "relevances")?;
    let starts = starting_locs
        .map(|values| as_i64_array(values, "starting_locs"))
        .transpose()?;
    let (out_agents, out_lats, out_lons, out_ts) = simulate_epr_agents_from_cached_od_impl(
        arrow_values(&lats),
        arrow_values(&lons),
        arrow_values(&rels),
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
        n_agents,
        random_state,
        starts.as_ref().map(arrow_i64_values),
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        Py::new(py, i64_results_into_arrow(out_agents))?.into_any(),
        Py::new(py, f64_results_into_arrow(out_lats))?.into_any(),
        Py::new(py, f64_results_into_arrow(out_lons))?.into_any(),
        Py::new(py, i64_results_into_arrow(out_ts))?.into_any(),
    ))
}
