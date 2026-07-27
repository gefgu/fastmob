use fastmob_core::measures::individual::total_distance::{
    total_distance_indexed_impl, total_distance_presorted_impl,
};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::adapters::trajectory::{run_indexed_coordinate_arrow, run_presorted_coordinate_arrow};
use crate::utils::f64_results_into_arrow;

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

#[pyfunction]
pub fn total_distance_presorted<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<Py<PyAny>> {
    let values =
        run_presorted_coordinate_arrow(py, latitudes, longitudes, ends, |coords, ends| {
            total_distance_presorted_impl(coords.latitudes, coords.longitudes, ends)
        })?
        .map_err(PyValueError::new_err)?;
    arrow_f64_output(py, values)
}

#[pyfunction]
pub fn total_distance_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<Py<PyAny>> {
    let values = run_indexed_coordinate_arrow(py, latitudes, longitudes, indices, ends, |view| {
        total_distance_indexed_impl(
            view.coordinates.latitudes,
            view.coordinates.longitudes,
            view.indices,
            view.ends,
            view.valid_rows,
        )
    })?
    .map_err(PyValueError::new_err)?;
    arrow_f64_output(py, values)
}
