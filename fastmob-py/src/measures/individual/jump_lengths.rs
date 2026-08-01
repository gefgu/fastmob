use fastmob_core::measures::individual::jump_lengths::{
    jump_lengths_indexed_impl, jump_lengths_km as core_jump_lengths_km, jump_lengths_presorted_impl,
};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::adapters::trajectory::{run_indexed_coordinate_arrow, run_presorted_coordinate_arrow};
use crate::utils::f64_results_into_arrow;

type PyJumpLengths<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Py<PyAny>,
);

#[pyfunction]
pub fn jump_lengths_km(latitudes: Vec<f64>, longitudes: Vec<f64>) -> PyResult<Vec<f64>> {
    core_jump_lengths_km(latitudes, longitudes).map_err(PyValueError::new_err)
}

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

#[pyfunction]
pub fn jump_lengths_presorted<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
) -> PyResult<PyJumpLengths<'py>> {
    let (starts, ends, values) =
        run_presorted_coordinate_arrow(py, latitudes, longitudes, ends, |coords, ends| {
            jump_lengths_presorted_impl(coords.latitudes, coords.longitudes, ends)
        })?
        .map_err(PyValueError::new_err)?;
    Ok((
        starts.into_pyarray(py),
        ends.into_pyarray(py),
        arrow_f64_output(py, values)?,
    ))
}

#[pyfunction]
pub fn jump_lengths_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
) -> PyResult<PyJumpLengths<'py>> {
    let (starts, ends, values) =
        run_indexed_coordinate_arrow(py, latitudes, longitudes, indices, ends, |view| {
            jump_lengths_indexed_impl(
                view.coordinates.latitudes,
                view.coordinates.longitudes,
                view.indices,
                view.ends,
                view.valid_rows,
            )
        })?
        .map_err(PyValueError::new_err)?;
    Ok((
        starts.into_pyarray(py),
        ends.into_pyarray(py),
        arrow_f64_output(py, values)?,
    ))
}
