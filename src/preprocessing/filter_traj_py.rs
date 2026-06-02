use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use super::filter_traj::{FilterConfig, filter_trajectory_impl};

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
