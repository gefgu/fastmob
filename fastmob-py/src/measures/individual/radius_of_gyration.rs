use fastmob_core::measures::individual::radius_of_gyration::{
    radius_of_gyration_indexed_impl, radius_of_gyration_presorted_impl,
};
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::adapters::trajectory::{run_indexed_coordinate_arrow, run_presorted_coordinate_arrow};
use crate::utils::f64_results_into_arrow;

type PyRogResult<'py> = (Py<PyAny>, Bound<'py, PyArray1<bool>>);

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

#[pyfunction]
pub fn radius_of_gyration_presorted<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
) -> PyResult<PyRogResult<'py>> {
    let (values, validity) =
        run_presorted_coordinate_arrow(py, latitudes, longitudes, ends, |coords, ends| {
            radius_of_gyration_presorted_impl(coords.latitudes, coords.longitudes, ends)
        })?;
    Ok((arrow_f64_output(py, values)?, validity.into_pyarray(py)))
}

#[pyfunction]
pub fn radius_of_gyration_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
) -> PyResult<PyRogResult<'py>> {
    let (values, validity) =
        run_indexed_coordinate_arrow(py, latitudes, longitudes, indices, ends, |view| {
            radius_of_gyration_indexed_impl(
                view.coordinates.latitudes,
                view.coordinates.longitudes,
                view.indices,
                view.ends,
                view.valid_rows,
            )
        })?;
    Ok((arrow_f64_output(py, values)?, validity.into_pyarray(py)))
}
