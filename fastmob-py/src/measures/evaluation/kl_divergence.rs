use fastmob_core::measures::evaluation::kl_divergence::kullback_leibler_divergence_impl;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{arrow_values, as_f64_array};

/// Compute KL divergence from paired Arrow float64 arrays in Rust.
#[pyfunction]
pub fn kullback_leibler_divergence(a: ArrowPyArray, b: ArrowPyArray) -> PyResult<f64> {
    let a = as_f64_array(a, "true")?;
    let b = as_f64_array(b, "pred")?;
    kullback_leibler_divergence_impl(arrow_values(&a), arrow_values(&b))
        .map_err(PyValueError::new_err)
}
