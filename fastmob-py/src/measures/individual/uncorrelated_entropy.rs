use fastmob_core::measures::individual::uncorrelated_entropy::uncorrelated_entropy_indexed_impl;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::adapters::trajectory::run_indexed_coordinate_arrow;
use crate::utils::f64_results_into_arrow;

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

#[pyfunction]
pub fn uncorrelated_entropy_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    normalize: bool,
) -> PyResult<Py<PyAny>> {
    let values = run_indexed_coordinate_arrow(py, latitudes, longitudes, indices, ends, |view| {
        uncorrelated_entropy_indexed_impl(
            view.coordinates.latitudes,
            view.coordinates.longitudes,
            view.indices,
            view.ends,
            normalize,
            view.valid_rows,
        )
    })?
    .map_err(PyValueError::new_err)?;
    arrow_f64_output(py, values)
}
