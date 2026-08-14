use crate::utils::ArrowUsizeArrayExt;
use fastmob_core::measures::individual::activity::{
    activity_counts as activity_counts_impl,
    activity_transition_counts as activity_transition_counts_impl,
    daily_activity_percentages as daily_activity_percentages_impl,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_bool_values, arrow_i64_values, arrow_u32_values, as_bool_array, as_i64_array,
    as_u32_array, f64_results_into_arrow, u64_results_into_arrow,
};

fn arrow_u64_output(py: Python<'_>, values: Vec<u64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, u64_results_into_arrow(values))?.into_any())
}

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

#[pyfunction]
pub fn activity_counts(
    py: Python<'_>,
    codes: ArrowPyArray,
    n_activities: usize,
) -> PyResult<Py<PyAny>> {
    let codes = as_u32_array(codes, "codes")?;
    let codes = arrow_u32_values(&codes).to_vec();
    let values = py
        .detach(move || activity_counts_impl(&codes, n_activities))
        .map_err(PyValueError::new_err)?;
    arrow_u64_output(py, values)
}

#[pyfunction]
pub fn activity_transition_counts<'py>(
    py: Python<'py>,
    codes: ArrowPyArray,
    indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
    n_activities: usize,
) -> PyResult<Py<PyAny>> {
    let indices = indices.as_slice()?.to_vec();
    let ends = ends.as_slice()?.to_vec();
    let codes = as_u32_array(codes, "codes")?;
    let codes = arrow_u32_values(&codes).to_vec();
    let values = py
        .detach(move || activity_transition_counts_impl(&codes, &indices, &ends, n_activities))
        .map_err(PyValueError::new_err)?;
    arrow_u64_output(py, values)
}

#[pyfunction]
pub fn daily_activity_percentages(
    py: Python<'_>,
    codes: ArrowPyArray,
    start_minutes: ArrowPyArray,
    end_minutes: ArrowPyArray,
    valid_rows: ArrowPyArray,
    n_activities: usize,
    bin_size_minutes: usize,
) -> PyResult<Py<PyAny>> {
    let codes = as_u32_array(codes, "codes")?;
    let start_minutes = as_i64_array(start_minutes, "start_minutes")?;
    let end_minutes = as_i64_array(end_minutes, "end_minutes")?;
    let valid_rows = as_bool_array(valid_rows, "valid_rows")?;
    let codes = arrow_u32_values(&codes).to_vec();
    let start_minutes = arrow_i64_values(&start_minutes).to_vec();
    let end_minutes = arrow_i64_values(&end_minutes).to_vec();
    let valid_rows = arrow_bool_values(&valid_rows);
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
    arrow_f64_output(py, values)
}
