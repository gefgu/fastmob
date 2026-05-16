use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{arrow_values, as_f64_array, f64_results_into_arrow};

use super::jump_lengths::{
    PyNonOrderedJumpLengthsArrow, PyPresortedJumpLengthsArrow, jump_lengths_presorted_impl,
    time_ordered_flat_values_impl, validate_time_ordered_inputs,
};
use crate::time_ordering::{ordered_index_ranges_into_numpy, time_ordered_indices_from_arrow_uids};

#[pyfunction]
pub(crate) fn jump_lengths_non_ordered_arrow<'py>(
    py: Python<'py>,
    uids: &Bound<'py, PyAny>,
    timestamps: PyArray,
    latitudes: PyArray,
    longitudes: PyArray,
) -> PyResult<PyNonOrderedJumpLengthsArrow<'py>> {
    let timestamps = as_f64_array(timestamps, "timestamps")?;
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps = arrow_values(&timestamps);
    let latitudes = arrow_values(&latitudes);
    let longitudes = arrow_values(&longitudes);
    validate_time_ordered_inputs(latitudes, longitudes, timestamps)?;

    let (indices, ranges) = time_ordered_indices_from_arrow_uids(uids, timestamps)?;
    let (indices, ranges, values) =
        time_ordered_flat_values_impl(latitudes, longitudes, timestamps, indices, ranges)?;
    let (indices, starts, ends) = ordered_index_ranges_into_numpy(py, (indices, ranges));
    Ok((indices, starts, ends, f64_results_into_arrow(values)))
}

#[pyfunction]
pub(crate) fn jump_lengths_presorted_arrow<'py>(
    py: Python<'py>,
    latitudes: PyArray,
    longitudes: PyArray,
    ranges: Vec<(usize, usize)>,
) -> PyResult<PyPresortedJumpLengthsArrow<'py>> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let (starts, ends, values) =
        jump_lengths_presorted_impl(arrow_values(&latitudes), arrow_values(&longitudes), &ranges)?;
    Ok((
        starts.into_pyarray(py),
        ends.into_pyarray(py),
        f64_results_into_arrow(values),
    ))
}
