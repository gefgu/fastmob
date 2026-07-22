use fastmob_core::measures::individual::jump_lengths::{
    jump_lengths_indexed_impl, jump_lengths_presorted_impl, time_ordered_flat_values_impl,
    validate_time_ordered_inputs,
};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, f64_results_into_arrow,
};

use super::jump_lengths::{PyNonOrderedJumpLengthsArrow, PyPresortedJumpLengthsArrow};
use super::time_ordering::{
    ordered_index_ranges_into_start_end_numpy, time_ordered_indices_from_arrow_uids,
};

#[pyfunction]
#[pyo3(signature = (uids, timestamps, latitudes, longitudes, num_groups = None))]
pub fn jump_lengths_non_ordered_arrow<'py>(
    py: Python<'py>,
    uids: &Bound<'py, PyAny>,
    timestamps: PyArray,
    latitudes: PyArray,
    longitudes: PyArray,
    num_groups: Option<usize>,
) -> PyResult<PyNonOrderedJumpLengthsArrow<'py>> {
    let timestamps = as_nullable_f64_array(timestamps, "timestamps")?;
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes, &timestamps]);
    let timestamps_vals = arrow_values(&timestamps);
    let latitudes_vals = arrow_values(&latitudes);
    let longitudes_vals = arrow_values(&longitudes);
    validate_time_ordered_inputs(latitudes_vals, longitudes_vals, timestamps_vals)
        .map_err(PyValueError::new_err)?;

    let (indices, ranges) =
        time_ordered_indices_from_arrow_uids(py, uids, timestamps_vals, num_groups)?;
    let (indices, ranges, values) = time_ordered_flat_values_impl(
        latitudes_vals,
        longitudes_vals,
        timestamps_vals,
        indices,
        ranges,
        valid_rows.as_deref(),
    )
    .map_err(PyValueError::new_err)?;
    let (indices, starts, ends) = ordered_index_ranges_into_start_end_numpy(py, (indices, ranges));
    Ok((indices, starts, ends, f64_results_into_arrow(values)))
}

#[pyfunction]
pub fn jump_lengths_presorted_arrow<'py>(
    py: Python<'py>,
    latitudes: PyArray,
    longitudes: PyArray,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyPresortedJumpLengthsArrow<'py>> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let (starts, ends, values) = jump_lengths_presorted_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        ends.as_slice()?,
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        starts.into_pyarray(py),
        ends.into_pyarray(py),
        f64_results_into_arrow(values),
    ))
}

#[pyfunction]
pub fn jump_lengths_indexed_arrow<'py>(
    py: Python<'py>,
    latitudes: PyArray,
    longitudes: PyArray,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyPresortedJumpLengthsArrow<'py>> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }

    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    let (starts, ends, values) = jump_lengths_indexed_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        indices.as_slice()?,
        ends.as_slice()?,
        valid_rows.as_deref(),
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        starts.into_pyarray(py),
        ends.into_pyarray(py),
        f64_results_into_arrow(values),
    ))
}
