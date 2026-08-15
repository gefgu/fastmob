use fastmob_core::measures::individual::jump_lengths::{
    jump_lengths_indexed_impl, jump_lengths_km as core_jump_lengths_km, jump_lengths_presorted_impl,
};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::adapters::trajectory::run_presorted_coordinate_arrow;
use crate::utils::{
    ArrowU64ArrayExt, arrow_valid_rows, arrow_values, as_nullable_f64_array,
    f64_results_into_arrow, validate_indexed_ends_u64,
};

type PyJumpLengths<'py> = (
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Py<PyAny>,
);

#[pyfunction]
pub fn jump_lengths_km(latitudes: Vec<f64>, longitudes: Vec<f64>) -> PyResult<Vec<f64>> {
    core_jump_lengths_km(latitudes, longitudes).map_err(PyValueError::new_err)
}

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

fn u64_indices(values: Vec<usize>) -> Vec<u64> {
    values.into_iter().map(|value| value as u64).collect()
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
        u64_indices(starts).into_pyarray(py),
        u64_indices(ends).into_pyarray(py),
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
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }
    let indices = indices.as_u64_slice()?;
    let ends = ends.as_u64_slice()?;
    validate_indexed_ends_u64(latitudes.len(), indices, ends)?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    let (starts, ends, values) = py
        .detach(|| jump_lengths_indexed_impl(lats, lngs, indices, ends, valid_rows.as_deref()))
        .map_err(PyValueError::new_err)?;
    Ok((
        u64_indices(starts).into_pyarray(py),
        u64_indices(ends).into_pyarray(py),
        arrow_f64_output(py, values)?,
    ))
}
