use fastmob_core::measures::individual::jump_lengths::{
    jump_lengths_indexed_impl, jump_lengths_km as core_jump_lengths_km, jump_lengths_presorted_impl,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::{PyArray, PyArray as ArrowPyArray};

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, f64_results_into_arrow,
    validate_indexed_ends,
};

pub type PyNonOrderedJumpLengths<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<f64>>,
);
pub type PyNonOrderedJumpLengthsArrow<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    PyArray,
);

type PyJumpLengths<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Py<PyAny>,
);
type PyNonOrderedJumpLengthsLogical<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Py<PyAny>,
);

#[pyfunction]
pub fn jump_lengths_km(latitudes: Vec<f64>, longitudes: Vec<f64>) -> PyResult<Vec<f64>> {
    core_jump_lengths_km(latitudes, longitudes).map_err(PyValueError::new_err)
}

fn is_arrow_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.hasattr("__arrow_c_array__")
}

fn numpy_f64_output<'py>(py: Python<'py>, values: Vec<f64>) -> Py<PyAny> {
    values.into_pyarray(py).into_any().unbind()
}

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

#[pyfunction]
pub fn jump_lengths_presorted<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyJumpLengths<'py>> {
    if let (Ok(latitudes), Ok(longitudes)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let (starts, ends, values) = jump_lengths_presorted_impl(
            latitudes.as_slice()?,
            longitudes.as_slice()?,
            ends.as_slice()?,
        )
        .map_err(PyValueError::new_err)?;
        return Ok((
            starts.into_pyarray(py),
            ends.into_pyarray(py),
            numpy_f64_output(py, values),
        ));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? {
        let latitudes = as_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes = as_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        let (starts, ends, values) = jump_lengths_presorted_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            ends.as_slice()?,
        )
        .map_err(PyValueError::new_err)?;
        return Ok((
            starts.into_pyarray(py),
            ends.into_pyarray(py),
            arrow_f64_output(py, values)?,
        ));
    }

    Err(PyTypeError::new_err(
        "latitudes and longitudes must both be NumPy arrays or both be Arrow arrays",
    ))
}

#[pyfunction]
pub fn jump_lengths_indexed<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyJumpLengths<'py>> {
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;
    if let (Ok(latitudes), Ok(longitudes)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let lats = latitudes.as_slice()?;
        let lngs = longitudes.as_slice()?;
        validate_indexed_ends(lats.len(), indices, ends)?;
        let (starts, ends, values) = jump_lengths_indexed_impl(lats, lngs, indices, ends, None)
            .map_err(PyValueError::new_err)?;
        return Ok((
            starts.into_pyarray(py),
            ends.into_pyarray(py),
            numpy_f64_output(py, values),
        ));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? {
        let latitudes = as_nullable_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes =
            as_nullable_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        let lats = arrow_values(&latitudes);
        let lngs = arrow_values(&longitudes);
        validate_indexed_ends(lats.len(), indices, ends)?;
        let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
        let (starts, ends, values) =
            jump_lengths_indexed_impl(lats, lngs, indices, ends, valid_rows.as_deref())
                .map_err(PyValueError::new_err)?;
        return Ok((
            starts.into_pyarray(py),
            ends.into_pyarray(py),
            arrow_f64_output(py, values)?,
        ));
    }

    Err(PyTypeError::new_err(
        "latitudes and longitudes must both be NumPy arrays or both be Arrow arrays",
    ))
}

#[pyfunction]
#[pyo3(signature = (uids, timestamps, latitudes, longitudes, num_groups = None))]
pub fn jump_lengths_non_ordered<'py>(
    py: Python<'py>,
    uids: &Bound<'py, PyAny>,
    timestamps: &Bound<'py, PyAny>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    num_groups: Option<usize>,
) -> PyResult<PyNonOrderedJumpLengthsLogical<'py>> {
    if let (Ok(timestamps), Ok(latitudes), Ok(longitudes)) = (
        timestamps.extract::<PyReadonlyArray1<f64>>(),
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let (indices, starts, ends, values) =
            super::jump_lengths_numpy::jump_lengths_non_ordered_from_numpy(
                py, uids, timestamps, latitudes, longitudes, num_groups,
            )?;
        return Ok((indices, starts, ends, values.into_any().unbind()));
    }

    if is_arrow_array(timestamps)? && is_arrow_array(latitudes)? && is_arrow_array(longitudes)? {
        let (indices, starts, ends, values) =
            super::jump_lengths_arrow::jump_lengths_non_ordered_from_arrow(
                py,
                uids,
                timestamps.extract::<PyArray>()?,
                latitudes.extract::<PyArray>()?,
                longitudes.extract::<PyArray>()?,
                num_groups,
            )?;
        return Ok((indices, starts, ends, Py::new(py, values)?.into_any()));
    }

    Err(PyTypeError::new_err(
        "timestamps, latitudes, and longitudes must all be NumPy arrays or all be Arrow arrays",
    ))
}
