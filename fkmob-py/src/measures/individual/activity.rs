use arrow_array::{Array, BooleanArray, Int64Array, UInt64Array};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use fkmob_core::measures::individual::activity::{
    activity_counts, activity_transition_counts, daily_activity_percentages,
};

use crate::utils::{f64_results_into_arrow, u64_results_into_arrow};

fn arrow_u64_values(array: PyArray, name: &str) -> PyResult<Vec<u64>> {
    let (array, _) = array.into_inner();
    let array = array
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| PyValueError::new_err(format!("expected uint64 Arrow array for {name}")))?;
    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }
    Ok(array.values()[array.offset()..array.offset() + array.len()].to_vec())
}

fn arrow_i64_values(array: PyArray, name: &str) -> PyResult<Vec<i64>> {
    let (array, _) = array.into_inner();
    let array = array
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| PyValueError::new_err(format!("expected int64 Arrow array for {name}")))?;
    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }
    Ok(array.values()[array.offset()..array.offset() + array.len()].to_vec())
}

fn arrow_bool_values(array: PyArray, name: &str) -> PyResult<Vec<bool>> {
    let (array, _) = array.into_inner();
    let array = array
        .as_any()
        .downcast_ref::<BooleanArray>()
        .ok_or_else(|| PyValueError::new_err(format!("expected boolean Arrow array for {name}")))?;
    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }
    Ok((0..array.len()).map(|index| array.value(index)).collect())
}

#[pyfunction]
pub fn activity_counts_numpy<'py>(
    py: Python<'py>,
    codes: PyReadonlyArray1<'py, u64>,
    n_activities: usize,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let codes = codes.as_slice()?.to_vec();
    Ok(py
        .detach(move || activity_counts(&codes, n_activities))
        .map_err(PyValueError::new_err)?
        .into_pyarray(py))
}

#[pyfunction]
pub fn activity_counts_arrow(
    py: Python<'_>,
    codes: PyArray,
    n_activities: usize,
) -> PyResult<PyArray> {
    let codes = arrow_u64_values(codes, "codes")?;
    Ok(u64_results_into_arrow(
        py.detach(move || activity_counts(&codes, n_activities))
            .map_err(PyValueError::new_err)?,
    ))
}

#[pyfunction]
pub fn activity_transition_counts_numpy<'py>(
    py: Python<'py>,
    codes: PyReadonlyArray1<'py, u64>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    n_activities: usize,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let codes = codes.as_slice()?.to_vec();
    let indices = indices.as_slice()?.to_vec();
    let ends = ends.as_slice()?.to_vec();
    Ok(py
        .detach(move || activity_transition_counts(&codes, &indices, &ends, n_activities))
        .map_err(PyValueError::new_err)?
        .into_pyarray(py))
}

#[pyfunction]
pub fn activity_transition_counts_arrow(
    py: Python<'_>,
    codes: PyArray,
    indices: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
    n_activities: usize,
) -> PyResult<PyArray> {
    let codes = arrow_u64_values(codes, "codes")?;
    let indices = indices.as_slice()?.to_vec();
    let ends = ends.as_slice()?.to_vec();
    Ok(u64_results_into_arrow(
        py.detach(move || activity_transition_counts(&codes, &indices, &ends, n_activities))
            .map_err(PyValueError::new_err)?,
    ))
}

#[pyfunction]
pub fn daily_activity_percentages_numpy<'py>(
    py: Python<'py>,
    codes: PyReadonlyArray1<'py, u64>,
    start_minutes: PyReadonlyArray1<'py, i64>,
    end_minutes: PyReadonlyArray1<'py, i64>,
    valid_rows: PyReadonlyArray1<'py, bool>,
    n_activities: usize,
    bin_size_minutes: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let codes = codes.as_slice()?.to_vec();
    let start_minutes = start_minutes.as_slice()?.to_vec();
    let end_minutes = end_minutes.as_slice()?.to_vec();
    let valid_rows = valid_rows.as_slice()?.to_vec();
    Ok(py
        .detach(move || {
            daily_activity_percentages(
                &codes,
                &start_minutes,
                &end_minutes,
                &valid_rows,
                n_activities,
                bin_size_minutes,
            )
        })
        .map_err(PyValueError::new_err)?
        .into_pyarray(py))
}

#[pyfunction]
pub fn daily_activity_percentages_arrow(
    py: Python<'_>,
    codes: PyArray,
    start_minutes: PyArray,
    end_minutes: PyArray,
    valid_rows: PyArray,
    n_activities: usize,
    bin_size_minutes: usize,
) -> PyResult<PyArray> {
    let codes = arrow_u64_values(codes, "codes")?;
    let start_minutes = arrow_i64_values(start_minutes, "start_minutes")?;
    let end_minutes = arrow_i64_values(end_minutes, "end_minutes")?;
    let valid_rows = arrow_bool_values(valid_rows, "valid_rows")?;
    Ok(f64_results_into_arrow(
        py.detach(move || {
            daily_activity_percentages(
                &codes,
                &start_minutes,
                &end_minutes,
                &valid_rows,
                n_activities,
                bin_size_minutes,
            )
        })
        .map_err(PyValueError::new_err)?,
    ))
}
