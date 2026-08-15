use fastmob_core::measures::individual::entropy::{
    real_entropy_users as core_real_entropy_users,
    trajectory_predictability_batch as core_trajectory_predictability_batch,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{as_u32_array, ranges_from_ends};

type PredictabilityBatchResult = (Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>);

#[pyfunction]
pub fn real_entropy_users(
    py: Python<'_>,
    location_ids: ArrowPyArray,
    ends: ArrowPyArray,
    normalized: bool,
) -> PyResult<Vec<f64>> {
    let location_ids = as_u32_array(location_ids, "location_ids")?;
    let n_location_ids = location_ids.len();
    let location_ids = location_ids.values().to_vec();
    let ranges = ranges_from_ends(ends, n_location_ids)?
        .into_iter()
        .map(|(start, end)| (start as u64, end as u64))
        .collect();
    let result = py.detach(move || core_real_entropy_users(location_ids, ranges, normalized));
    result.map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn trajectory_predictability_batch(
    py: Python<'_>,
    location_ids: ArrowPyArray,
    ends: ArrowPyArray,
) -> PyResult<PredictabilityBatchResult> {
    let location_ids = as_u32_array(location_ids, "location_ids")?;
    let n_location_ids = location_ids.len();
    let location_ids = location_ids.values().to_vec();
    let ranges = ranges_from_ends(ends, n_location_ids)?
        .into_iter()
        .map(|(start, end)| (start as u64, end as u64))
        .collect();
    let result = py.detach(move || core_trajectory_predictability_batch(location_ids, ranges));
    result.map_err(PyValueError::new_err)
}
