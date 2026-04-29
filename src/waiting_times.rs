use arrow_array::{Array, Float64Array, PrimitiveArray, types::Float64Type};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use rayon::prelude::*;

fn validate_inputs(timestamps_s: &[f64], ranges: &[(usize, usize)]) -> PyResult<()> {
    let n = timestamps_s.len();
    for &(start, end) in ranges {
        if start > end {
            return Err(PyValueError::new_err(
                "range start must be less than or equal to range end",
            ));
        }
        if end > n {
            return Err(PyValueError::new_err(
                "range end must be within timestamp array bounds",
            ));
        }
    }
    Ok(())
}

fn waiting_times_for_range(timestamps_s: &[f64], start: usize, end: usize) -> Vec<f64> {
    if end - start < 2 {
        return Vec::new();
    }
    (start + 1..end)
        .map(|idx| timestamps_s[idx] - timestamps_s[idx - 1])
        .collect()
}

fn waiting_times_impl(timestamps_s: &[f64], ranges: &[(usize, usize)]) -> PyResult<Vec<Vec<f64>>> {
    validate_inputs(timestamps_s, ranges)?;

    Ok(ranges
        .par_iter()
        .map(|&(start, end)| waiting_times_for_range(timestamps_s, start, end))
        .collect())
}

fn waiting_times_flat_impl(timestamps_s: &[f64], ranges: &[(usize, usize)]) -> PyResult<Vec<f64>> {
    validate_inputs(timestamps_s, ranges)?;

    Ok(ranges
        .iter()
        .flat_map(|&(start, end)| waiting_times_for_range(timestamps_s, start, end))
        .collect())
}

fn as_f64_array(arr: PyArray, name: &str) -> PyResult<PrimitiveArray<Float64Type>> {
    let (array_ref, _field) = arr.into_inner();
    let array = array_ref
        .as_any()
        .downcast_ref::<Float64Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected float64 Arrow array for {name}")))?;

    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }

    Ok(array)
}

fn arrow_values(array: &PrimitiveArray<Float64Type>) -> &[f64] {
    let start = array.offset();
    let end = start + array.len();
    &array.values()[start..end]
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
