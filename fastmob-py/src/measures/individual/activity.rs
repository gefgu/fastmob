use arrow_array::{Array, BooleanArray, Int64Array, UInt64Array};
use fastmob_core::measures::individual::activity::{
    activity_counts as activity_counts_impl,
    activity_transition_counts as activity_transition_counts_impl,
    daily_activity_percentages as daily_activity_percentages_impl,
};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{f64_results_into_arrow, u64_results_into_arrow};

const ACTIVITY_BACKEND_ERROR: &str = "activity arrays must be NumPy arrays or Arrow arrays";

fn is_arrow_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.hasattr("__arrow_c_array__")
}

fn numpy_u64_output(py: Python<'_>, values: Vec<u64>) -> Py<PyAny> {
    values.into_pyarray(py).into_any().unbind()
}

fn numpy_f64_output(py: Python<'_>, values: Vec<f64>) -> Py<PyAny> {
    values.into_pyarray(py).into_any().unbind()
}

fn arrow_u64_output(py: Python<'_>, values: Vec<u64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, u64_results_into_arrow(values))?.into_any())
}

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

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
    Ok((0..array.len()).map(|idx| array.value(idx)).collect())
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
    Ok((0..array.len()).map(|idx| array.value(idx)).collect())
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
    Ok((0..array.len()).map(|idx| array.value(idx)).collect())
}

#[pyfunction]
pub fn activity_counts<'py>(
    py: Python<'py>,
    codes: &Bound<'py, PyAny>,
    n_activities: usize,
) -> PyResult<Py<PyAny>> {
    if let Ok(codes) = codes.extract::<PyReadonlyArray1<u64>>() {
        let codes = codes.as_slice()?.to_vec();
        let values = py
            .detach(move || activity_counts_impl(&codes, n_activities))
            .map_err(PyValueError::new_err)?;
        return Ok(numpy_u64_output(py, values));
    }

    if is_arrow_array(codes)? {
        let codes = arrow_u64_values(codes.extract::<PyArray>()?, "codes")?;
        let values = py
            .detach(move || activity_counts_impl(&codes, n_activities))
            .map_err(PyValueError::new_err)?;
        return arrow_u64_output(py, values);
    }

    Err(PyTypeError::new_err(ACTIVITY_BACKEND_ERROR))
}

#[pyfunction]
pub fn activity_transition_counts<'py>(
    py: Python<'py>,
    codes: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    n_activities: usize,
) -> PyResult<Py<PyAny>> {
    let indices = indices.as_slice()?.to_vec();
    let ends = ends.as_slice()?.to_vec();

    if let Ok(codes) = codes.extract::<PyReadonlyArray1<u64>>() {
        let codes = codes.as_slice()?.to_vec();
        let values = py
            .detach(move || activity_transition_counts_impl(&codes, &indices, &ends, n_activities))
            .map_err(PyValueError::new_err)?;
        return Ok(numpy_u64_output(py, values));
    }

    if is_arrow_array(codes)? {
        let codes = arrow_u64_values(codes.extract::<PyArray>()?, "codes")?;
        let values = py
            .detach(move || activity_transition_counts_impl(&codes, &indices, &ends, n_activities))
            .map_err(PyValueError::new_err)?;
        return arrow_u64_output(py, values);
    }

    Err(PyTypeError::new_err(ACTIVITY_BACKEND_ERROR))
}

#[pyfunction]
pub fn daily_activity_percentages<'py>(
    py: Python<'py>,
    codes: &Bound<'py, PyAny>,
    start_minutes: &Bound<'py, PyAny>,
    end_minutes: &Bound<'py, PyAny>,
    valid_rows: &Bound<'py, PyAny>,
    n_activities: usize,
    bin_size_minutes: usize,
) -> PyResult<Py<PyAny>> {
    if let (Ok(codes), Ok(start_minutes), Ok(end_minutes), Ok(valid_rows)) = (
        codes.extract::<PyReadonlyArray1<u64>>(),
        start_minutes.extract::<PyReadonlyArray1<i64>>(),
        end_minutes.extract::<PyReadonlyArray1<i64>>(),
        valid_rows.extract::<PyReadonlyArray1<bool>>(),
    ) {
        let codes = codes.as_slice()?.to_vec();
        let start_minutes = start_minutes.as_slice()?.to_vec();
        let end_minutes = end_minutes.as_slice()?.to_vec();
        let valid_rows = valid_rows.as_slice()?.to_vec();
        let values = py
            .detach(move || {
                daily_activity_percentages_impl(
                    &codes,
                    &start_minutes,
                    &end_minutes,
                    &valid_rows,
                    n_activities,
                    bin_size_minutes,
                )
            })
            .map_err(PyValueError::new_err)?;
        return Ok(numpy_f64_output(py, values));
    }

    if is_arrow_array(codes)?
        && is_arrow_array(start_minutes)?
        && is_arrow_array(end_minutes)?
        && is_arrow_array(valid_rows)?
    {
        let codes = arrow_u64_values(codes.extract::<PyArray>()?, "codes")?;
        let start_minutes = arrow_i64_values(start_minutes.extract::<PyArray>()?, "start_minutes")?;
        let end_minutes = arrow_i64_values(end_minutes.extract::<PyArray>()?, "end_minutes")?;
        let valid_rows = arrow_bool_values(valid_rows.extract::<PyArray>()?, "valid_rows")?;
        let values = py
            .detach(move || {
                daily_activity_percentages_impl(
                    &codes,
                    &start_minutes,
                    &end_minutes,
                    &valid_rows,
                    n_activities,
                    bin_size_minutes,
                )
            })
            .map_err(PyValueError::new_err)?;
        return arrow_f64_output(py, values);
    }

    Err(PyTypeError::new_err(ACTIVITY_BACKEND_ERROR))
}
