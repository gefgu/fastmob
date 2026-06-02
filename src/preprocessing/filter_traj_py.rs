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
    Ok(PyArray1::from_vec(
        py,
        filter_trajectory_impl(
            latitudes.as_slice()?,
            longitudes.as_slice()?,
            timestamps_s.as_slice()?,
            &ranges,
            &config,
        ),
    ))
}

#[pyfunction]
pub(crate) fn filter_trajectory_arrow(
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    ranges: Vec<(usize, usize)>,
    config: FilterConfig,
) -> PyResult<ArrowPyArray> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;
    Ok(bool_results_into_arrow(filter_trajectory_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        arrow_values(&timestamps_s),
        &ranges,
        &config,
    )))
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
    let ranges: Vec<(usize, usize)> = starts
        .as_slice()?
        .iter()
        .copied()
        .zip(ends.as_slice()?.iter().copied())
        .collect();

    Ok(PyArray1::from_vec(
        py,
        filter_trajectory_indexed_impl(
            latitudes.as_slice()?,
            longitudes.as_slice()?,
            timestamps_s.as_slice()?,
            sorted_indices.as_slice()?,
            &ranges,
            &config,
        ),
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_trajectory_indexed_arrow(
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    sorted_indices: PyReadonlyArray1<usize>,
    starts: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
    config: FilterConfig,
) -> PyResult<ArrowPyArray> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;
    let ranges: Vec<(usize, usize)> = starts
        .as_slice()?
        .iter()
        .copied()
        .zip(ends.as_slice()?.iter().copied())
        .collect();

    Ok(bool_results_into_arrow(filter_trajectory_indexed_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        arrow_values(&timestamps_s),
        sorted_indices.as_slice()?,
        &ranges,
        &config,
    )))
}
