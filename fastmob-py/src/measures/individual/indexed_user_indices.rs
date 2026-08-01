use fastmob_core::utils::{split_user_index_ranges, user_indices_for_u64_codes};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{arrow_u64_values, as_u64_array, extract_arrow_array, u64_results_into_arrow};

#[pyfunction]
pub fn single_user_indices(length: usize) -> (PyArray, PyArray) {
    let indices = (0..length as u64).collect();
    let ends = if length == 0 {
        Vec::new()
    } else {
        vec![length as u64]
    };
    (
        u64_results_into_arrow(indices),
        u64_results_into_arrow(ends),
    )
}

#[pyfunction]
pub fn indexed_user_indices(
    py: Python<'_>,
    uids: &Bound<'_, PyAny>,
    num_groups: usize,
) -> PyResult<(PyArray, PyArray)> {
    let uids = as_u64_array(extract_arrow_array(uids, "uids")?, "uids")?;
    let grouped = py
        .detach(|| user_indices_for_u64_codes(arrow_u64_values(&uids), num_groups))
        .map_err(PyValueError::new_err)?;
    let (indices, ends) = split_user_index_ranges(grouped);
    Ok((
        u64_results_into_arrow(indices.into_iter().map(|value| value as u64).collect()),
        u64_results_into_arrow(ends.into_iter().map(|value| value as u64).collect()),
    ))
}
