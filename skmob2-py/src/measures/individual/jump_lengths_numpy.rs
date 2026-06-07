use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use skmob2_core::measures::individual::jump_lengths::{
    jump_lengths_indexed_impl, jump_lengths_presorted_impl, time_ordered_flat_values_impl,
    validate_time_ordered_inputs,
};

use super::jump_lengths::{PyNonOrderedJumpLengths, PyPresortedJumpLengths};
use super::time_ordering::{
    ordered_index_ranges_into_start_end_numpy, time_ordered_indices_from_numpy_uids,
};

#[pyfunction]
#[pyo3(signature = (uids, timestamps, latitudes, longitudes, num_groups = None))]
pub fn jump_lengths_non_ordered_numpy<'py>(
    py: Python<'py>,
    uids: &Bound<'py, PyAny>,
    timestamps: PyReadonlyArray1<'py, f64>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    num_groups: Option<usize>,
) -> PyResult<PyNonOrderedJumpLengths<'py>> {
    let timestamps = timestamps.as_slice()?;
    let latitudes = latitudes.as_slice()?;
    let longitudes = longitudes.as_slice()?;
    validate_time_ordered_inputs(latitudes, longitudes, timestamps)
        .map_err(PyValueError::new_err)?;

    let (indices, ranges) = time_ordered_indices_from_numpy_uids(py, uids, timestamps, num_groups)?;
    let (indices, ranges, values) =
        time_ordered_flat_values_impl(latitudes, longitudes, timestamps, indices, ranges, None)
            .map_err(PyValueError::new_err)?;
    let (indices, starts, ends) = ordered_index_ranges_into_start_end_numpy(py, (indices, ranges));
    Ok((indices, starts, ends, values.into_pyarray(py)))
}

#[pyfunction]
pub fn jump_lengths_presorted_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyPresortedJumpLengths<'py>> {
    let (starts, ends, values) = jump_lengths_presorted_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        ends.as_slice()?,
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        starts.into_pyarray(py),
        ends.into_pyarray(py),
        values.into_pyarray(py),
    ))
}

#[pyfunction]
pub fn jump_lengths_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyPresortedJumpLengths<'py>> {
    let (starts, ends, values) = jump_lengths_indexed_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        indices.as_slice()?,
        ends.as_slice()?,
        None,
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        starts.into_pyarray(py),
        ends.into_pyarray(py),
        values.into_pyarray(py),
    ))
}
