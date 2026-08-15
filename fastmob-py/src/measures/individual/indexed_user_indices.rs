use fastmob_core::utils::{split_user_index_ranges, user_indices_for_u32_codes};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{arrow_u32_values, as_u32_array, extract_arrow_array, u64_results_into_arrow};

#[pyfunction]
pub fn single_user_indices(length: usize) -> PyResult<(PyArray, PyArray)> {
    let length_u64 = u64::try_from(length)
        .map_err(|_| PyValueError::new_err("row count exceeds the UInt64 index limit"))?;
    let indices = (0..length).map(|value| value as u64).collect();
    let ends = if length == 0 {
        Vec::new()
    } else {
        vec![length_u64]
    };
    Ok((
        u64_results_into_arrow(indices),
        u64_results_into_arrow(ends),
    ))
}

#[pyfunction]
pub fn indexed_user_indices(
    py: Python<'_>,
    uids: &Bound<'_, PyAny>,
    num_groups: usize,
) -> PyResult<(PyArray, PyArray)> {
    let uids = as_u32_array(extract_arrow_array(uids, "uids")?, "uids")?;
    let grouped = py
        .detach(|| user_indices_for_u32_codes(arrow_u32_values(&uids), num_groups))
        .map_err(PyValueError::new_err)?;
    let (indices, ends) = split_user_index_ranges(grouped);
    Ok((
        u64_results_into_arrow(indices),
        u64_results_into_arrow(ends),
    ))
}
