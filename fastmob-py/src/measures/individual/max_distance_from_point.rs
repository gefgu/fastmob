use fastmob_core::measures::individual::max_distance_from_point::{
    max_distance_from_point_from_ends_impl, max_distance_from_point_indexed_impl,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::adapters::trajectory::{
    run_indexed_group_coordinate_arrow, run_presorted_group_coordinate_arrow,
};
use crate::utils::f64_results_into_arrow;

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn max_distance_from_point_presorted<'py>(
    py: Python<'py>,
    home_lats: ArrowPyArray,
    home_lngs: ArrowPyArray,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
) -> PyResult<Py<PyAny>> {
    let values = run_presorted_group_coordinate_arrow(
        py,
        home_lats,
        home_lngs,
        latitudes,
        longitudes,
        ends,
        |coords, ends| {
            max_distance_from_point_from_ends_impl(
                coords.group_latitudes,
                coords.group_longitudes,
                coords.row_latitudes,
                coords.row_longitudes,
                ends,
                coords.valid_rows,
            )
        },
    )?
    .map_err(PyValueError::new_err)?;
    arrow_f64_output(py, values)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn max_distance_from_point_indexed<'py>(
    py: Python<'py>,
    home_lats: ArrowPyArray,
    home_lngs: ArrowPyArray,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
) -> PyResult<Py<PyAny>> {
    let values = run_indexed_group_coordinate_arrow(
        py,
        home_lats,
        home_lngs,
        latitudes,
        longitudes,
        indices,
        ends,
        |view| {
            max_distance_from_point_indexed_impl(
                view.coordinates.group_latitudes,
                view.coordinates.group_longitudes,
                view.coordinates.row_latitudes,
                view.coordinates.row_longitudes,
                view.indices,
                view.ends,
                view.valid_rows,
            )
        },
    )?
    .map_err(PyValueError::new_err)?;
    arrow_f64_output(py, values)
}
