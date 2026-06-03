use std::time::Instant;

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use super::filter_traj::{FilterConfig, filter_trajectory_impl, filter_trajectory_indexed_impl};
use crate::utils::{arrow_values, as_f64_array, bool_results_into_arrow};

#[pyfunction]
pub(crate) fn filter_trajectory(
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    timestamps_s: Vec<f64>,
    ranges: Vec<(usize, usize)>,
    config: FilterConfig,
) -> PyResult<Vec<bool>> {
    Ok(filter_trajectory_impl(
        &latitudes,
        &longitudes,
        &timestamps_s,
        &ranges,
        &config,
    ))
}

#[pyfunction]
pub(crate) fn filter_trajectory_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    timestamps_s: PyReadonlyArray1<'py, f64>,
    ranges: Vec<(usize, usize)>,
    config: FilterConfig,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    // 1. Extract raw Rust slices while we hold the GIL
    let lats = latitudes.as_slice()?;
    let lngs = longitudes.as_slice()?;
    let times = timestamps_s.as_slice()?;

    // 2. Release the GIL and blast all CPU cores
    let keep_mask = py.detach(|| filter_trajectory_impl(lats, lngs, times, &ranges, &config));

    // 3. Convert back to Python (GIL is automatically re-acquired here)
    Ok(PyArray1::from_vec(py, keep_mask))
}

#[pyfunction]
pub(crate) fn filter_trajectory_arrow(
    py: Python<'_>, // <-- Added the Python token here
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    ranges: Vec<(usize, usize)>,
    config: FilterConfig,
) -> PyResult<ArrowPyArray> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;

    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    let times = arrow_values(&timestamps_s);

    // Release the GIL
    let keep_mask = py.detach(|| filter_trajectory_impl(lats, lngs, times, &ranges, &config));

    Ok(bool_results_into_arrow(keep_mask))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_trajectory_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    timestamps_s: PyReadonlyArray1<'py, f64>,
    sorted_indices: PyReadonlyArray1<'py, usize>,
    starts: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    config: FilterConfig,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let start = Instant::now();
    let lats = latitudes.as_slice()?;
    let lngs = longitudes.as_slice()?;
    let times = timestamps_s.as_slice()?;
    let indices = sorted_indices.as_slice()?;
    let starts_slice = starts.as_slice()?;
    let ends_slice = ends.as_slice()?;

    let ranges: Vec<(usize, usize)> = starts_slice
        .iter()
        .copied()
        .zip(ends_slice.iter().copied())
        .collect();

    // Release the GIL
    let keep_mask =
        py.detach(|| filter_trajectory_indexed_impl(lats, lngs, times, indices, &ranges, &config));

    // let keep_mask = filter_trajectory_indexed_impl(lats, lngs, times, indices, &ranges, &config);

    println!("Total RUST time: {:.2?}", start.elapsed());

    Ok(PyArray1::from_vec(py, keep_mask))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_trajectory_indexed_arrow(
    py: Python<'_>, // <-- Added the Python token here
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    sorted_indices: PyReadonlyArray1<usize>,
    starts: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
    config: FilterConfig,
) -> PyResult<ArrowPyArray> {
    let start = Instant::now();
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;

    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    let times = arrow_values(&timestamps_s);
    let indices = sorted_indices.as_slice()?;

    let ranges: Vec<(usize, usize)> = starts
        .as_slice()?
        .iter()
        .copied()
        .zip(ends.as_slice()?.iter().copied())
        .collect();

    // Release the GIL
    let keep_mask =
        py.detach(|| filter_trajectory_indexed_impl(lats, lngs, times, indices, &ranges, &config));
    // let keep_mask = filter_trajectory_indexed_impl(lats, lngs, times, indices, &ranges, &config);

    println!("Total RUST time: {:.2?}", start.elapsed());

    Ok(bool_results_into_arrow(keep_mask))
}
