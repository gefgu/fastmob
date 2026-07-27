use fastmob_core::measures::evaluation::wasserstein::empirical_wasserstein_1d;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{arrow_values, as_f64_array};

#[pyfunction]
pub fn wasserstein(a: ArrowPyArray, b: ArrowPyArray) -> PyResult<f64> {
    let a = as_f64_array(a, "a")?;
    let b = as_f64_array(b, "b")?;
    empirical_wasserstein_1d(arrow_values(&a), arrow_values(&b)).map_err(PyValueError::new_err)
}
