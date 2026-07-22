use fastmob_core::measures::individual::jump_lengths::{
    time_ordered_flat_values_impl, validate_time_ordered_inputs,
};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::jump_lengths::PyNonOrderedJumpLengths;
use super::time_ordering::{
    ordered_index_ranges_into_start_end_arrays, time_ordered_indices_from_ndarray_uids,
};

pub(crate) fn jump_lengths_non_ordered_from_numpy<'py>(
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

    let (indices, ranges) =
        time_ordered_indices_from_ndarray_uids(py, uids, timestamps, num_groups)?;
    let (indices, ranges, values) =
        time_ordered_flat_values_impl(latitudes, longitudes, timestamps, indices, ranges, None)
            .map_err(PyValueError::new_err)?;
    let (indices, starts, ends) = ordered_index_ranges_into_start_end_arrays(py, (indices, ranges));
    Ok((indices, starts, ends, values.into_pyarray(py)))
}
