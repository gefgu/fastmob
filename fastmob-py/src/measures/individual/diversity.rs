use fastmob_core::measures::individual::diversity::diversity_users as core_diversity_users;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{as_u32_array, ranges_from_ends};

#[pyfunction]
pub fn diversity_users(
    py: Python<'_>,
    location_ids: ArrowPyArray,
    ends: ArrowPyArray,
) -> PyResult<Vec<f64>> {
    let location_ids = as_u32_array(location_ids, "location_ids")?;
    let n_location_ids = location_ids.len();
    let location_ids = location_ids.values().to_vec();
    let ranges = ranges_from_ends(ends, n_location_ids)?;
    let result = py.detach(move || core_diversity_users(location_ids, ranges));
    result.map_err(PyValueError::new_err)
}
