use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use skmob2_core::measures::evaluation::wasserstein::empirical_wasserstein_1d;

use crate::utils::{arrow_values, as_f64_array};

#[pyfunction]
pub fn wasserstein_numpy(a: PyReadonlyArray1<f64>, b: PyReadonlyArray1<f64>) -> PyResult<f64> {
    empirical_wasserstein_1d(a.as_slice()?, b.as_slice()?).map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn wasserstein_arrow(a: PyArray, b: PyArray) -> PyResult<f64> {
    let a = as_f64_array(a, "a")?;
    let b = as_f64_array(b, "b")?;
    empirical_wasserstein_1d(arrow_values(&a), arrow_values(&b)).map_err(PyValueError::new_err)
}
