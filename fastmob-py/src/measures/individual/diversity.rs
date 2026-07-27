use fastmob_core::measures::individual::diversity::diversity_batch as core_diversity_batch;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub fn diversity_batch(
    py: Python<'_>,
    tokens: Vec<String>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    let result = py.detach(move || core_diversity_batch(tokens, ranges));
    result.map_err(PyValueError::new_err)
}
