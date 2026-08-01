use std::str::FromStr;

use fastmob_core::trajectory::interpolate_at::{
    interpolate_at_indexed_impl, interpolate_at_presorted_impl, PositionQueryMethod,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
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

type InterpolateAtOutput<'py> = (Py<PyAny>, Py<PyAny>, Bound<'py, PyArray1<bool>>);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn interpolate_at_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    sorted_indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
    query_times_s: PyReadonlyArray1<'py, f64>,
    method: &str,
) -> PyResult<InterpolateAtOutput<'py>> {
    let query_times_s = query_times_s.as_slice()?;
    let method = PositionQueryMethod::from_str(method).map_err(PyValueError::new_err)?;
    let (out_lats, out_lngs, out_valid) = run_indexed_timed_coordinate_arrow(
        py,
        latitudes,
        longitudes,
        timestamps_s,
        sorted_indices,
        ends,
        |view| {
            interpolate_at_indexed_impl(
                view.coordinates.latitudes,
                view.coordinates.longitudes,
                view.coordinates.times,
                view.indices,
                view.ends,
                view.valid_rows,
                query_times_s,
                method,
            )
        },
    )?
    .map_err(PyValueError::new_err)?;
    Ok((
        arrow_f64_output(py, out_lats)?,
        arrow_f64_output(py, out_lngs)?,
        out_valid.into_pyarray(py),
    ))
}

#[pyfunction]
pub fn interpolate_at_presorted<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
    query_times_s: PyReadonlyArray1<'py, f64>,
    method: &str,
) -> PyResult<InterpolateAtOutput<'py>> {
    let query_times_s = query_times_s.as_slice()?;
    let method = PositionQueryMethod::from_str(method).map_err(PyValueError::new_err)?;
    let (out_lats, out_lngs, out_valid) = run_presorted_timed_coordinate_arrow(
        py,
        latitudes,
        longitudes,
        timestamps_s,
        ends,
        |view, ends| {
            interpolate_at_presorted_impl(
                view.latitudes,
                view.longitudes,
                view.times,
                ends,
                query_times_s,
                method,
            )
        },
    )?
    .map_err(PyValueError::new_err)?;
    Ok((
        arrow_f64_output(py, out_lats)?,
        arrow_f64_output(py, out_lngs)?,
        out_valid.into_pyarray(py),
    ))
}
