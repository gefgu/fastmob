use fastmob_core::models::geosim::simulate_geosim_impl;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_i64_values, arrow_values, as_f64_array, as_i64_array, f64_results_into_arrow,
    i64_results_into_arrow,
};

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    latitudes, longitudes, neighbor_starts, neighbors,
    rho, gamma, alpha, beta, tau, xmin,
    start_ts, end_ts, indipendency_window_s, dt_update_mob_sim_s,
    n_agents, master_seed=None, starting_locs=None
))]
pub fn model_geosim_simulate_agents<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    neighbor_starts: PyReadonlyArray1<'py, i64>,
    neighbors: PyReadonlyArray1<'py, i64>,
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
    starting_locs: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let lats = latitudes.as_slice()?;
    let lngs = longitudes.as_slice()?;
    let ns_raw = neighbor_starts.as_slice()?;
    let nb_raw = neighbors.as_slice()?;
    let ns: Vec<usize> = ns_raw.iter().map(|&v| v.max(0) as usize).collect();
    let nb: Vec<usize> = nb_raw.iter().map(|&v| v.max(0) as usize).collect();
    let sl_buf: Vec<usize>;
    let sl: Option<&[usize]> = match &starting_locs {
        Some(v) => {
            sl_buf = v.as_slice()?.iter().map(|&x| x.max(0) as usize).collect();
            Some(&sl_buf)
        }
        None => None,
    };
    let (out_agents, out_lats, out_lngs, out_ts) = simulate_geosim_impl(
        lats,
        lngs,
        &ns,
        &nb,
        rho,
        gamma,
        alpha,
        beta,
        tau,
        xmin,
        start_ts,
        end_ts,
        indipendency_window_s,
        dt_update_mob_sim_s,
        n_agents,
        master_seed,
        sl,
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        out_agents.into_pyarray(py),
        out_lats.into_pyarray(py),
        out_lngs.into_pyarray(py),
        out_ts.into_pyarray(py),
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    latitudes, longitudes, neighbor_starts, neighbors,
    rho, gamma, alpha, beta, tau, xmin,
    start_ts, end_ts, indipendency_window_s, dt_update_mob_sim_s,
    n_agents, master_seed=None, starting_locs=None
))]
pub fn model_geosim_simulate_agents_arrow<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    neighbor_starts: ArrowPyArray,
    neighbors: ArrowPyArray,
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
    starting_locs: Option<ArrowPyArray>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let lats = as_f64_array(latitudes, "latitudes")?;
    let lngs = as_f64_array(longitudes, "longitudes")?;
    let neighbor_starts = as_i64_array(neighbor_starts, "neighbor_starts")?;
    let neighbors = as_i64_array(neighbors, "neighbors")?;
    let starts: Vec<usize> = arrow_i64_values(&neighbor_starts)
        .iter()
        .map(|&v| v.max(0) as usize)
        .collect();
    let neighbors: Vec<usize> = arrow_i64_values(&neighbors)
        .iter()
        .map(|&v| v.max(0) as usize)
        .collect();
    let starting_locs = starting_locs
        .map(|values| as_i64_array(values, "starting_locs"))
        .transpose()?;
    let starting_locs: Option<Vec<usize>> = starting_locs.as_ref().map(|array| {
        arrow_i64_values(array)
            .iter()
            .map(|&v| v.max(0) as usize)
            .collect()
    });
    let (agents, out_lats, out_lngs, timestamps) = simulate_geosim_impl(
        arrow_values(&lats),
        arrow_values(&lngs),
        &starts,
        &neighbors,
        rho,
        gamma,
        alpha,
        beta,
        tau,
        xmin,
        start_ts,
        end_ts,
        indipendency_window_s,
        dt_update_mob_sim_s,
        n_agents,
        master_seed,
        starting_locs.as_deref(),
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        Py::new(py, i64_results_into_arrow(agents))?.into_any(),
        Py::new(py, f64_results_into_arrow(out_lats))?.into_any(),
        Py::new(py, f64_results_into_arrow(out_lngs))?.into_any(),
        Py::new(py, i64_results_into_arrow(timestamps))?.into_any(),
    ))
}
