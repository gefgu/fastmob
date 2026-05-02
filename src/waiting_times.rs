use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use rayon::prelude::*;

use crate::utils::{as_f64_array, arrow_values, validate_ranges};

fn waiting_times_for_range(timestamps_s: &[f64], start: usize, end: usize) -> Vec<f64> {
    if end - start < 2 {
        return Vec::new();
    }
    (start + 1..end)
        .map(|idx| timestamps_s[idx] - timestamps_s[idx - 1])
        .collect()
}

fn waiting_times_impl(timestamps_s: &[f64], ranges: &[(usize, usize)]) -> PyResult<Vec<Vec<f64>>> {
    validate_ranges(timestamps_s.len(), ranges)?;

    Ok(ranges
        .par_iter()
        .map(|&(start, end)| waiting_times_for_range(timestamps_s, start, end))
        .collect())
}

fn waiting_times_flat_impl(timestamps_s: &[f64], ranges: &[(usize, usize)]) -> PyResult<Vec<f64>> {
    validate_ranges(timestamps_s.len(), ranges)?;

    Ok(ranges
        .iter()
        .flat_map(|&(start, end)| waiting_times_for_range(timestamps_s, start, end))
        .collect())
}

#[pyfunction]
pub(crate) fn waiting_times_seconds(
    timestamps_s: Vec<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<Vec<f64>>> {
    waiting_times_impl(&timestamps_s, &ranges)
}

#[pyfunction]
pub(crate) fn waiting_times_numpy(
    timestamps_s: PyReadonlyArray1<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<Vec<f64>>> {
    waiting_times_impl(timestamps_s.as_slice()?, &ranges)
}

#[pyfunction]
pub(crate) fn waiting_times_flat_numpy(
    timestamps_s: PyReadonlyArray1<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    waiting_times_flat_impl(timestamps_s.as_slice()?, &ranges)
}

#[pyfunction]
pub(crate) fn waiting_times_arrow(
    timestamps_s: PyArray,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<Vec<f64>>> {
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;
    waiting_times_impl(arrow_values(&timestamps_s), &ranges)
}

#[pyfunction]
pub(crate) fn waiting_times_flat_arrow(
    timestamps_s: PyArray,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;
    waiting_times_flat_impl(arrow_values(&timestamps_s), &ranges)
}
