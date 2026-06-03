use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use skmob2_core::measures::individual::entropy::{
    real_entropy_batch as core_real_entropy_batch,
    trajectory_entropy_batch as core_trajectory_entropy_batch,
    trajectory_predictability_batch as core_trajectory_predictability_batch,
};

type PredictabilityBatchResult = (Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>);

#[pyfunction]
pub fn real_entropy_batch(
    py: Python<'_>,
    tokens: Vec<String>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    let result = py.detach(move || core_real_entropy_batch(tokens, ranges));
    result.map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn trajectory_entropy_batch(
    py: Python<'_>,
    tokens: Vec<String>,
    ranges: Vec<(usize, usize)>,
    normalized: bool,
) -> PyResult<Vec<f64>> {
    let result = py.detach(move || core_trajectory_entropy_batch(tokens, ranges, normalized));
    result.map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn trajectory_predictability_batch(
    py: Python<'_>,
    tokens: Vec<String>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<PredictabilityBatchResult> {
    let result = py.detach(move || core_trajectory_predictability_batch(tokens, ranges));
    result.map_err(PyValueError::new_err)
}
