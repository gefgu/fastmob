use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use fastmob_core::measures::individual::uncorrelated_entropy::uncorrelated_entropy_indexed_impl;

use crate::utils::{arrow_valid_rows, arrow_values, as_nullable_f64_array, f64_results_into_arrow};

#[pyfunction]
pub fn uncorrelated_entropy_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    normalize: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(uncorrelated_entropy_indexed_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        indices.as_slice()?,
        ends.as_slice()?,
        normalize,
        None,
    )
    .map_err(PyValueError::new_err)?
    .into_pyarray(py))
}

#[pyfunction]
pub fn uncorrelated_entropy_indexed_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    indices: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
    normalize: bool,
) -> PyResult<PyArray> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    Ok(f64_results_into_arrow(
        uncorrelated_entropy_indexed_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            indices.as_slice()?,
            ends.as_slice()?,
            normalize,
            valid_rows.as_deref(),
        )
        .map_err(PyValueError::new_err)?,
    ))
}
