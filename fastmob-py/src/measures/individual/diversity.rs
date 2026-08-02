use fastmob_core::measures::individual::diversity::diversity_batch as core_diversity_batch;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::ranges_from_ends;

#[pyfunction]
pub fn diversity_batch(
    py: Python<'_>,
    tokens: Vec<String>,
    ends: ArrowPyArray,
) -> PyResult<Vec<f64>> {
    let ranges = ranges_from_ends(ends, tokens.len())?;
    let result = py.detach(move || core_diversity_batch(tokens, ranges));
    result.map_err(PyValueError::new_err)
}
