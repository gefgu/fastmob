use fastmob_core::models::sts_epr::simulate_sts_epr_impl;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_i32_values, arrow_i64_values, arrow_values, as_f64_array, as_i32_array, as_i64_array,
    f64_results_into_arrow, i64_results_into_arrow,
};

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    latitudes, longitudes, relevances, distances,
    neighbor_starts, neighbors,
    diary_timestamps, diary_abs_locs, diary_starts, diary_ends,
    rho, gamma, alpha,
    start_ts, end_ts, indipendency_window_s, dt_update_mob_sim_s,
    n_agents, master_seed=None, starting_locs=None,
    starting_locs_mode_relevance=false
))]
pub fn model_sts_epr_simulate_agents<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    relevances: PyReadonlyArray1<'py, f64>,
    distances: PyReadonlyArray1<'py, f64>,
    neighbor_starts: PyReadonlyArray1<'py, i64>,
    neighbors: PyReadonlyArray1<'py, i64>,
    diary_timestamps: PyReadonlyArray1<'py, i64>,
    diary_abs_locs: PyReadonlyArray1<'py, i32>,
    diary_starts: PyReadonlyArray1<'py, i64>,
    diary_ends: PyReadonlyArray1<'py, i64>,
    rho: f64,
    gamma: f64,
    alpha: f64,
    start_ts: i64,
    end_ts: i64,
    indipendency_window_s: i64,
    dt_update_mob_sim_s: i64,
    n_agents: usize,
    master_seed: Option<u64>,
    starting_locs: Option<PyReadonlyArray1<'py, i64>>,
    starting_locs_mode_relevance: bool,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    let lats = latitudes.as_slice()?;
    let lngs = longitudes.as_slice()?;
    let rels = relevances.as_slice()?;
    let dists = distances.as_slice()?;
    let ns_raw = neighbor_starts.as_slice()?;
    let nb_raw = neighbors.as_slice()?;
    let dt_raw = diary_timestamps.as_slice()?;
    let da_raw = diary_abs_locs.as_slice()?;
    let ds_raw = diary_starts.as_slice()?;
    let de_raw = diary_ends.as_slice()?;

    let ns: Vec<usize> = ns_raw.iter().map(|&v| v.max(0) as usize).collect();
    let nb: Vec<usize> = nb_raw.iter().map(|&v| v.max(0) as usize).collect();
    let ds: Vec<usize> = ds_raw.iter().map(|&v| v.max(0) as usize).collect();
    let de: Vec<usize> = de_raw.iter().map(|&v| v.max(0) as usize).collect();

    let sl_buf: Vec<usize>;
    let sl: Option<&[usize]> = match &starting_locs {
        Some(v) => {
            sl_buf = v.as_slice()?.iter().map(|&x| x.max(0) as usize).collect();
            Some(&sl_buf)
        }
        None => None,
    };

    let (out_agents, out_lats, out_lngs, out_ts) = simulate_sts_epr_impl(
        lats,
        lngs,
        rels,
        dists,
        &ns,
        &nb,
        dt_raw,
        da_raw,
        &ds,
        &de,
        rho,
        gamma,
        alpha,
        start_ts,
        end_ts,
        indipendency_window_s,
        dt_update_mob_sim_s,
        n_agents,
        master_seed,
        sl,
        starting_locs_mode_relevance,
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
    latitudes, longitudes, relevances, distances,
    neighbor_starts, neighbors,
    diary_timestamps, diary_abs_locs, diary_starts, diary_ends,
    rho, gamma, alpha,
    start_ts, end_ts, indipendency_window_s, dt_update_mob_sim_s,
    n_agents, master_seed=None, starting_locs=None,
    starting_locs_mode_relevance=false
))]
pub fn model_sts_epr_simulate_agents_arrow<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    relevances: ArrowPyArray,
    distances: ArrowPyArray,
    neighbor_starts: ArrowPyArray,
    neighbors: ArrowPyArray,
    diary_timestamps: ArrowPyArray,
    diary_abs_locs: ArrowPyArray,
    diary_starts: ArrowPyArray,
    diary_ends: ArrowPyArray,
    rho: f64,
    gamma: f64,
    alpha: f64,
    start_ts: i64,
    end_ts: i64,
    indipendency_window_s: i64,
    dt_update_mob_sim_s: i64,
    n_agents: usize,
    master_seed: Option<u64>,
    starting_locs: Option<ArrowPyArray>,
    starting_locs_mode_relevance: bool,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let lats = as_f64_array(latitudes, "latitudes")?;
    let lngs = as_f64_array(longitudes, "longitudes")?;
    let relevances = as_f64_array(relevances, "relevances")?;
    let distances = as_f64_array(distances, "distances")?;
    let neighbor_starts = as_i64_array(neighbor_starts, "neighbor_starts")?;
    let neighbors = as_i64_array(neighbors, "neighbors")?;
    let diary_timestamps = as_i64_array(diary_timestamps, "diary_timestamps")?;
    let diary_locations = as_i32_array(diary_abs_locs, "diary_abs_locs")?;
    let diary_starts = as_i64_array(diary_starts, "diary_starts")?;
    let diary_ends = as_i64_array(diary_ends, "diary_ends")?;
    let starts: Vec<usize> = arrow_i64_values(&neighbor_starts)
        .iter()
        .map(|&v| v.max(0) as usize)
        .collect();
    let neighbors: Vec<usize> = arrow_i64_values(&neighbors)
        .iter()
        .map(|&v| v.max(0) as usize)
        .collect();
    let diary_starts: Vec<usize> = arrow_i64_values(&diary_starts)
        .iter()
        .map(|&v| v.max(0) as usize)
        .collect();
    let diary_ends: Vec<usize> = arrow_i64_values(&diary_ends)
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
    let (agents, out_lats, out_lngs, timestamps) = simulate_sts_epr_impl(
        arrow_values(&lats),
        arrow_values(&lngs),
        arrow_values(&relevances),
        arrow_values(&distances),
        &starts,
        &neighbors,
        arrow_i64_values(&diary_timestamps),
        arrow_i32_values(&diary_locations),
        &diary_starts,
        &diary_ends,
        rho,
        gamma,
        alpha,
        start_ts,
        end_ts,
        indipendency_window_s,
        dt_update_mob_sim_s,
        n_agents,
        master_seed,
        starting_locs.as_deref(),
        starting_locs_mode_relevance,
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        Py::new(py, i64_results_into_arrow(agents))?.into_any(),
        Py::new(py, f64_results_into_arrow(out_lats))?.into_any(),
        Py::new(py, f64_results_into_arrow(out_lngs))?.into_any(),
        Py::new(py, i64_results_into_arrow(timestamps))?.into_any(),
    ))
}
