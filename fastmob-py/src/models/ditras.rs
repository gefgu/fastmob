use fastmob_core::models::ditras::simulate_ditras_agents_impl;
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
    latitudes, longitudes, relevances,
    diary_timestamps, diary_abs_locs, diary_starts, diary_ends,
    deterrence_type, deterrence_arg, origin_exp, destination_exp,
    rho, gamma, start_ts, end_ts,
    n_agents, master_seed=None, starting_locs=None
))]
pub fn model_ditras_simulate_agents<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    relevances: PyReadonlyArray1<'py, f64>,
    diary_timestamps: PyReadonlyArray1<'py, i64>,
    diary_abs_locs: PyReadonlyArray1<'py, i32>,
    diary_starts: PyReadonlyArray1<'py, i64>,
    diary_ends: PyReadonlyArray1<'py, i64>,
    deterrence_type: &str,
    deterrence_arg: f64,
    origin_exp: f64,
    destination_exp: f64,
    rho: f64,
    gamma: f64,
    start_ts: i64,
    end_ts: i64,
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
    let rels = relevances.as_slice()?;
    let dt_raw = diary_timestamps.as_slice()?;
    let da_raw = diary_abs_locs.as_slice()?;
    let ds_raw = diary_starts.as_slice()?;
    let de_raw = diary_ends.as_slice()?;

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

    let (out_agents, out_lats, out_lngs, out_ts) = simulate_ditras_agents_impl(
        lats,
        lngs,
        rels,
        dt_raw,
        da_raw,
        &ds,
        &de,
        deterrence_type,
        deterrence_arg,
        origin_exp,
        destination_exp,
        rho,
        gamma,
        start_ts,
        end_ts,
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
    latitudes, longitudes, relevances,
    diary_timestamps, diary_abs_locs, diary_starts, diary_ends,
    deterrence_type, deterrence_arg, origin_exp, destination_exp,
    rho, gamma, start_ts, end_ts,
    n_agents, master_seed=None, starting_locs=None
))]
pub fn model_ditras_simulate_agents_arrow<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    relevances: ArrowPyArray,
    diary_timestamps: ArrowPyArray,
    diary_abs_locs: ArrowPyArray,
    diary_starts: ArrowPyArray,
    diary_ends: ArrowPyArray,
    deterrence_type: &str,
    deterrence_arg: f64,
    origin_exp: f64,
    destination_exp: f64,
    rho: f64,
    gamma: f64,
    start_ts: i64,
    end_ts: i64,
    n_agents: usize,
    master_seed: Option<u64>,
    starting_locs: Option<ArrowPyArray>,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let lats = as_f64_array(latitudes, "latitudes")?;
    let lngs = as_f64_array(longitudes, "longitudes")?;
    let rels = as_f64_array(relevances, "relevances")?;
    let timestamps = as_i64_array(diary_timestamps, "diary_timestamps")?;
    let locations = as_i32_array(diary_abs_locs, "diary_abs_locs")?;
    let starts = as_i64_array(diary_starts, "diary_starts")?;
    let ends = as_i64_array(diary_ends, "diary_ends")?;
    let starts: Vec<usize> = arrow_i64_values(&starts)
        .iter()
        .map(|&v| v.max(0) as usize)
        .collect();
    let ends: Vec<usize> = arrow_i64_values(&ends)
        .iter()
        .map(|&v| v.max(0) as usize)
        .collect();
    let start_locations = starting_locs
        .map(|values| as_i64_array(values, "starting_locs"))
        .transpose()?;
    let start_locations: Option<Vec<usize>> = start_locations.as_ref().map(|array| {
        arrow_i64_values(array)
            .iter()
            .map(|&v| v.max(0) as usize)
            .collect()
    });
    let (agents, out_lats, out_lngs, out_ts) = simulate_ditras_agents_impl(
        arrow_values(&lats),
        arrow_values(&lngs),
        arrow_values(&rels),
        arrow_i64_values(&timestamps),
        arrow_i32_values(&locations),
        &starts,
        &ends,
        deterrence_type,
        deterrence_arg,
        origin_exp,
        destination_exp,
        rho,
        gamma,
        start_ts,
        end_ts,
        n_agents,
        master_seed,
        start_locations.as_deref(),
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        Py::new(py, i64_results_into_arrow(agents))?.into_any(),
        Py::new(py, f64_results_into_arrow(out_lats))?.into_any(),
        Py::new(py, f64_results_into_arrow(out_lngs))?.into_any(),
        Py::new(py, i64_results_into_arrow(out_ts))?.into_any(),
    ))
}
