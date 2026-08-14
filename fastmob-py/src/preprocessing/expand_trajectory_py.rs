use fastmob_core::preprocessing::expand_trajectory::{
    expand_5min_trajectory_batch_indexed_impl,
    expand_5min_trajectory_with_imputation_batch_indexed_impl,
};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    ArrowUsizeArrayExt, arrow_i64_values, arrow_u32_values, as_i64_array, as_u32_array,
    i64_results_into_arrow, u32_results_into_arrow, validate_indexed_ends,
};

type ExpandTrajectoryBatchResult<'py> = (
    Bound<'py, PyArray1<usize>>,
    Py<PyAny>,
    Bound<'py, PyArray1<usize>>,
);

/// Expands each user's `[start_timestamps_ms, end_timestamps_ms]` staypoint
/// interval into inclusive 5-minute-aligned slices, parallelized per user via
/// rayon (see `fastmob_core::preprocessing::expand_trajectory` for the
/// algorithm and why the previous pure-Python/narwhals loop in
/// `fastmob/measures/individual/mobility_profiling.py::_expand_to_5min_trajectory`
/// didn't scale). Only timestamps go through this kernel -- `source_row_idx`
/// lets the Python side `pc.take` location/purpose/any other per-row column
/// it needs from the *original* row, so this kernel has no opinion about
/// what a "staypoint" carries beyond a start/end interval.
#[pyfunction]
pub fn expand_5min_trajectory_batch_indexed<'py>(
    py: Python<'py>,
    start_timestamps_ms: ArrowPyArray,
    end_timestamps_ms: ArrowPyArray,
    sorted_indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
) -> PyResult<ExpandTrajectoryBatchResult<'py>> {
    let start = as_i64_array(start_timestamps_ms, "start_timestamps_ms")?;
    let end = as_i64_array(end_timestamps_ms, "end_timestamps_ms")?;
    if start.len() != end.len() {
        return Err(PyValueError::new_err(
            "start_timestamps_ms and end_timestamps_ms must have the same length",
        ));
    }
    let indices = sorted_indices.as_slice()?;
    let ends_slice = ends.as_slice()?;
    validate_indexed_ends(start.len(), indices, ends_slice)?;

    let start_vals = arrow_i64_values(&start);
    let end_vals = arrow_i64_values(&end);
    let (user_range_idx, timestamps_ms, source_row_idx) = py.detach(|| {
        expand_5min_trajectory_batch_indexed_impl(start_vals, end_vals, indices, ends_slice)
    });

    Ok((
        user_range_idx.into_pyarray(py),
        Py::new(py, i64_results_into_arrow(timestamps_ms))?.into_any(),
        source_row_idx.into_pyarray(py),
    ))
}

type ExpandTrajectoryWithImputationBatchResult<'py> =
    (Bound<'py, PyArray1<usize>>, Py<PyAny>, Py<PyAny>, Py<PyAny>);

/// `impute_gaps=True` counterpart of [`expand_5min_trajectory_batch_indexed`]:
/// also fills gaps between a user's first and last observed slice using
/// hour-of-day anchor locations (see
/// `fastmob_core::preprocessing::expand_trajectory`'s module docs for the
/// algorithm, ported from
/// `fastmob.measures.individual.mobility_profiling._impute_5min_gaps`).
/// Needs `location_codes` (dense `u64` codes, one per *input* row -- factorize
/// with `fastmob.utils._common._factorize_arrow_values` on the Python side)
/// since anchor selection depends on location, unlike the plain variant which
/// stays opinion-free about anything beyond timestamps.
///
/// Returns run-length-compressed output -- one `(user_range_idx,
/// run_start_ms, location_code, run_length)` record per maximal run of
/// consecutive same-location slices, not one record per slice. See
/// `ExpandTrajectoryWithImputationBatchResult`'s Rust-side docs for why this
/// is exact (not an approximation): the Python caller sums `run_length`
/// instead of counting rows during block aggregation, so the final numbers
/// are unchanged. Without this, a years-long sparse check-in history could
/// materialize hundreds of thousands of near-identical nightly/workday
/// "home"/"work" rows per user; compressed, it's one row per run.
#[pyfunction]
pub fn expand_5min_trajectory_with_imputation_batch_indexed<'py>(
    py: Python<'py>,
    start_timestamps_ms: ArrowPyArray,
    end_timestamps_ms: ArrowPyArray,
    location_codes: ArrowPyArray,
    sorted_indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
) -> PyResult<ExpandTrajectoryWithImputationBatchResult<'py>> {
    let start = as_i64_array(start_timestamps_ms, "start_timestamps_ms")?;
    let end = as_i64_array(end_timestamps_ms, "end_timestamps_ms")?;
    let location_codes = as_u32_array(location_codes, "location_codes")?;
    if start.len() != end.len() {
        return Err(PyValueError::new_err(
            "start_timestamps_ms and end_timestamps_ms must have the same length",
        ));
    }
    if start.len() != location_codes.len() {
        return Err(PyValueError::new_err(
            "location_codes must have the same length as start_timestamps_ms",
        ));
    }
    let indices = sorted_indices.as_slice()?;
    let ends_slice = ends.as_slice()?;
    validate_indexed_ends(start.len(), indices, ends_slice)?;

    let start_vals = arrow_i64_values(&start);
    let end_vals = arrow_i64_values(&end);
    let location_vals = arrow_u32_values(&location_codes);
    let (user_range_idx, timestamps_ms, location_out, run_length_out) = py.detach(|| {
        expand_5min_trajectory_with_imputation_batch_indexed_impl(
            start_vals,
            end_vals,
            location_vals,
            indices,
            ends_slice,
        )
    });

    Ok((
        user_range_idx.into_pyarray(py),
        Py::new(py, i64_results_into_arrow(timestamps_ms))?.into_any(),
        Py::new(py, u32_results_into_arrow(location_out))?.into_any(),
        Py::new(py, u32_results_into_arrow(run_length_out))?.into_any(),
    ))
}
