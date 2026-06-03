use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use skmob2_core::measures::individual::waiting_times::{
    waiting_times_flat_impl, waiting_times_impl, waiting_times_indexed_flat_impl,
    waiting_times_indexed_impl, waiting_times_seconds as core_waiting_times_seconds,
};

use crate::utils::{arrow_values, as_f64_array, ranges_from_starts_ends};

#[pyfunction]
pub fn waiting_times_seconds(
    timestamps_s: Vec<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<Vec<f64>>> {
    core_waiting_times_seconds(timestamps_s, ranges).map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn waiting_times_numpy(
    timestamps_s: PyReadonlyArray1<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<Vec<f64>>> {
    waiting_times_impl(timestamps_s.as_slice()?, &ranges).map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn waiting_times_flat_numpy(
    timestamps_s: PyReadonlyArray1<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    waiting_times_flat_impl(timestamps_s.as_slice()?, &ranges).map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn waiting_times_indexed_numpy(
    timestamps_s: PyReadonlyArray1<f64>,
    indices: PyReadonlyArray1<usize>,
    starts: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<Vec<Vec<f64>>> {
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    waiting_times_indexed_impl(timestamps_s.as_slice()?, indices.as_slice()?, &ranges)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn waiting_times_indexed_flat_numpy(
    timestamps_s: PyReadonlyArray1<f64>,
    indices: PyReadonlyArray1<usize>,
    starts: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<Vec<f64>> {
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    waiting_times_indexed_flat_impl(timestamps_s.as_slice()?, indices.as_slice()?, &ranges)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn waiting_times_arrow(
    timestamps_s: PyArray,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<Vec<f64>>> {
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;
    waiting_times_impl(arrow_values(&timestamps_s), &ranges).map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn waiting_times_flat_arrow(
    timestamps_s: PyArray,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;
    waiting_times_flat_impl(arrow_values(&timestamps_s), &ranges).map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn waiting_times_indexed_arrow(
    timestamps_s: PyArray,
    indices: PyReadonlyArray1<usize>,
    starts: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<Vec<Vec<f64>>> {
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    waiting_times_indexed_impl(arrow_values(&timestamps_s), indices.as_slice()?, &ranges)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn waiting_times_indexed_flat_arrow(
    timestamps_s: PyArray,
    indices: PyReadonlyArray1<usize>,
    starts: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<Vec<f64>> {
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    waiting_times_indexed_flat_impl(arrow_values(&timestamps_s), indices.as_slice()?, &ranges)
        .map_err(PyValueError::new_err)
}
