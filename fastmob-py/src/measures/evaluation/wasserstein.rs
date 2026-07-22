use fastmob_core::measures::evaluation::wasserstein::empirical_wasserstein_1d;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{arrow_values, as_f64_array};

fn is_arrow_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.hasattr("__arrow_c_array__")
}

#[pyfunction]
pub fn wasserstein<'py>(a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<f64> {
    if let (Ok(a), Ok(b)) = (
        a.extract::<PyReadonlyArray1<f64>>(),
        b.extract::<PyReadonlyArray1<f64>>(),
    ) {
        return empirical_wasserstein_1d(a.as_slice()?, b.as_slice()?)
            .map_err(PyValueError::new_err);
    }

    if is_arrow_array(a)? && is_arrow_array(b)? {
        let a = as_f64_array(a.extract::<PyArray>()?, "a")?;
        let b = as_f64_array(b.extract::<PyArray>()?, "b")?;
        return empirical_wasserstein_1d(arrow_values(&a), arrow_values(&b))
            .map_err(PyValueError::new_err);
    }

    Err(PyTypeError::new_err(
        "a and b must both be NumPy arrays or both be Arrow arrays",
    ))
}
