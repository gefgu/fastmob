use fastmob_core::measures::individual::time_ordering::{
    presorted_ranges_for_u32_codes, split_ordered_index_ranges, time_ordered_indices_for_u32_codes,
    time_ordered_indices_single_user,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{
    arrow_u32_values, arrow_values, as_nullable_f64_array, as_u32_array, extract_arrow_array,
    u64_results_into_arrow,
};

fn index_results(values: Vec<usize>) -> PyArray {
    u64_results_into_arrow(values.into_iter().map(|value| value as u64).collect())
}

#[pyfunction]
pub fn presorted_user_starts_ends(
    py: Python<'_>,
    uids: &Bound<'_, PyAny>,
) -> PyResult<(PyArray, PyArray)> {
    let uids = as_u32_array(extract_arrow_array(uids, "uids")?, "uids")?;
    let (starts, ends) = py.detach(|| presorted_ranges_for_u32_codes(arrow_u32_values(&uids)));
    Ok((index_results(starts), index_results(ends)))
}

#[pyfunction]
#[pyo3(signature = (uids, timestamps, num_groups = None))]
pub fn time_ordered_user_indices(
    py: Python<'_>,
    uids: Option<&Bound<'_, PyAny>>,
    timestamps: &Bound<'_, PyAny>,
    num_groups: Option<usize>,
) -> PyResult<(PyArray, PyArray)> {
    let timestamps =
        as_nullable_f64_array(extract_arrow_array(timestamps, "timestamps")?, "timestamps")?;
    let ordered = if let Some(uids) = uids {
        let uids = as_u32_array(extract_arrow_array(uids, "uids")?, "uids")?;
        let num_groups = num_groups.ok_or_else(|| {
            PyValueError::new_err("num_groups is required when uids are provided")
        })?;
        py.detach(|| {
            time_ordered_indices_for_u32_codes(
                arrow_u32_values(&uids),
                arrow_values(&timestamps),
                num_groups,
            )
        })
        .map_err(PyValueError::new_err)?
    } else {
        py.detach(|| time_ordered_indices_single_user(arrow_values(&timestamps)))
    };
    let (indices, ends) = split_ordered_index_ranges(ordered);
    Ok((index_results(indices), index_results(ends)))
}
