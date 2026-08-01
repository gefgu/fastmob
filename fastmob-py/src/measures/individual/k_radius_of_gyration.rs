use fastmob_core::measures::individual::k_radius_of_gyration::{
    k_radius_of_gyration_from_ends_impl, k_radius_of_gyration_indexed_impl,
    k_radius_of_gyration_km as core_k_rog_km,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::adapters::trajectory::{
    run_indexed_timed_coordinate_arrow, run_presorted_timed_coordinate_arrow,
};
use crate::utils::f64_results_into_arrow;

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

#[pyfunction]
pub fn k_radius_of_gyration_km(
    coords: Vec<(f64, f64)>,
    visit_counts: Vec<u64>,
    k: usize,
) -> PyResult<f64> {
    core_k_rog_km(coords, visit_counts, k).map_err(PyValueError::new_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn k_radius_of_gyration_presorted<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
    k: usize,
) -> PyResult<Py<PyAny>> {
    let values = run_presorted_timed_coordinate_arrow(
        py,
        latitudes,
        longitudes,
        timestamps,
        ends,
        |coords, ends| {
            k_radius_of_gyration_from_ends_impl(
                coords.latitudes,
                coords.longitudes,
                coords.times,
                ends,
                k,
            )
        },
    )?
    .map_err(PyValueError::new_err)?;
    arrow_f64_output(py, values)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn k_radius_of_gyration_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps: ArrowPyArray,
    indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
    k: usize,
) -> PyResult<Py<PyAny>> {
    let values = run_indexed_timed_coordinate_arrow(
        py,
        latitudes,
        longitudes,
        timestamps,
        indices,
        ends,
        |view| {
            k_radius_of_gyration_indexed_impl(
                view.coordinates.latitudes,
                view.coordinates.longitudes,
                view.coordinates.times,
                view.indices,
                view.ends,
                k,
                view.valid_rows,
            )
        },
    )?
    .map_err(PyValueError::new_err)?;
    arrow_f64_output(py, values)
}
