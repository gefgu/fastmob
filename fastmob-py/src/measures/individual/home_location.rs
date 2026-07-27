use fastmob_core::measures::individual::home_location::{
    home_location_from_ends_impl, home_location_indexed_impl,
};
use numpy::PyReadonlyArray1;
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
#[allow(clippy::too_many_arguments)]
pub fn home_location_presorted<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    hours: ArrowPyArray,
    ends: PyReadonlyArray1<'py, usize>,
    start_night: f64,
    end_night: f64,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let (lats, lngs) = run_presorted_timed_coordinate_arrow(
        py,
        latitudes,
        longitudes,
        hours,
        ends,
        |coords, ends| {
            home_location_from_ends_impl(
                coords.latitudes,
                coords.longitudes,
                coords.times,
                ends,
                start_night,
                end_night,
            )
        },
    )?
    .map_err(PyValueError::new_err)?;
    Ok((arrow_f64_output(py, lats)?, arrow_f64_output(py, lngs)?))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn home_location_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    hours: ArrowPyArray,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    start_night: f64,
    end_night: f64,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let (lats, lngs) = run_indexed_timed_coordinate_arrow(
        py,
        latitudes,
        longitudes,
        hours,
        indices,
        ends,
        |view| {
            home_location_indexed_impl(
                view.coordinates.latitudes,
                view.coordinates.longitudes,
                view.coordinates.times,
                view.indices,
                view.ends,
                start_night,
                end_night,
                view.valid_rows,
            )
        },
    )?
    .map_err(PyValueError::new_err)?;
    Ok((arrow_f64_output(py, lats)?, arrow_f64_output(py, lngs)?))
}
