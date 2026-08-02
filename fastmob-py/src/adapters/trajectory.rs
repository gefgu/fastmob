use crate::utils::ArrowUsizeArrayExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use fastmob_core::utils::validate_ends;

use crate::utils::{
    arrow_i64_values, arrow_valid_rows, arrow_valid_rows_f64_i64, arrow_values, as_f64_array,
    as_i64_array, as_nullable_f64_array, as_nullable_i64_array, ms_to_seconds, validate_indexed_ends,
};

pub struct CoordinateView<'a> {
    pub latitudes: &'a [f64],
    pub longitudes: &'a [f64],
}

pub struct IndexedCoordinateView<'a> {
    pub coordinates: CoordinateView<'a>,
    pub indices: &'a [usize],
    pub ends: &'a [usize],
    pub valid_rows: Option<&'a [bool]>,
}

pub struct TimedCoordinateView<'a> {
    pub latitudes: &'a [f64],
    pub longitudes: &'a [f64],
    pub times: &'a [f64],
}

pub struct IndexedTimedCoordinateView<'a> {
    pub coordinates: TimedCoordinateView<'a>,
    pub indices: &'a [usize],
    pub ends: &'a [usize],
    pub valid_rows: Option<&'a [bool]>,
}

pub struct GroupCoordinateView<'a> {
    pub group_latitudes: &'a [f64],
    pub group_longitudes: &'a [f64],
    pub row_latitudes: &'a [f64],
    pub row_longitudes: &'a [f64],
    pub valid_rows: Option<&'a [bool]>,
}

pub struct IndexedGroupCoordinateView<'a> {
    pub coordinates: GroupCoordinateView<'a>,
    pub indices: &'a [usize],
    pub ends: &'a [usize],
    pub valid_rows: Option<&'a [bool]>,
}

fn validate_coordinate_lengths(latitudes_len: usize, longitudes_len: usize) -> PyResult<()> {
    if latitudes_len != longitudes_len {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }
    Ok(())
}

fn validate_timed_coordinate_lengths(
    latitudes_len: usize,
    longitudes_len: usize,
    times_len: usize,
) -> PyResult<()> {
    validate_coordinate_lengths(latitudes_len, longitudes_len)?;
    if latitudes_len != times_len {
        return Err(PyValueError::new_err(
            "latitudes, longitudes, and time values must have the same length",
        ));
    }
    Ok(())
}

fn validate_group_coordinate_lengths(
    group_latitudes_len: usize,
    group_longitudes_len: usize,
    row_latitudes_len: usize,
    row_longitudes_len: usize,
    group_len: usize,
) -> PyResult<()> {
    validate_coordinate_lengths(group_latitudes_len, group_longitudes_len)?;
    validate_coordinate_lengths(row_latitudes_len, row_longitudes_len)?;
    if group_latitudes_len != group_len {
        return Err(PyValueError::new_err(
            "group coordinate arrays must match the number of groups",
        ));
    }
    Ok(())
}

/// Extracts and validates a presorted pair of Arrow coordinate arrays, then runs
/// `operation` under `py.detach` with clean coordinate slices. The result is handed back
/// as-is so callers with different output shapes (a plain `Vec<f64>`, a `(Vec<f64>, Vec<bool>)`
/// validity pair, a fallible `Result<_, String>`, ...) can wrap it themselves.
pub fn run_presorted_coordinate_arrow<'py, T, F>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
    operation: F,
) -> PyResult<T>
where
    F: FnOnce(CoordinateView<'_>, &[usize]) -> T + Send,
    T: Send,
{
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    validate_coordinate_lengths(latitudes.len(), longitudes.len())?;
    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    let ends = ends.as_slice()?;
    validate_ends(lats.len(), ends).map_err(PyValueError::new_err)?;
    Ok(py.detach(|| {
        operation(
            CoordinateView {
                latitudes: lats,
                longitudes: lngs,
            },
            ends,
        )
    }))
}

/// Indexed counterpart of [`run_presorted_coordinate_arrow`]: extracts and
/// validates a pair of (possibly nullable) Arrow coordinate arrays alongside
/// row indices/ends, computes `valid_rows`, then runs `operation` under
/// `py.detach`.
pub fn run_indexed_coordinate_arrow<'py, T, F>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
    operation: F,
) -> PyResult<T>
where
    F: FnOnce(IndexedCoordinateView<'_>) -> T + Send,
    T: Send,
{
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    validate_coordinate_lengths(latitudes.len(), longitudes.len())?;
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;
    validate_indexed_ends(latitudes.len(), indices, ends)?;

    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    Ok(py.detach(|| {
        operation(IndexedCoordinateView {
            coordinates: CoordinateView {
                latitudes: lats,
                longitudes: lngs,
            },
            indices,
            ends,
            valid_rows: valid_rows.as_deref(),
        })
    }))
}

/// Timed counterpart of [`run_presorted_coordinate_arrow`]: extracts and validates a
/// presorted lat/lng/time triple of Arrow arrays, then runs `operation` under `py.detach`.
pub fn run_presorted_timed_coordinate_arrow<'py, T, F>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    times: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
    operation: F,
) -> PyResult<T>
where
    F: FnOnce(TimedCoordinateView<'_>, &[usize]) -> T + Send,
    T: Send,
{
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let times = as_f64_array(times, "times")?;
    validate_timed_coordinate_lengths(latitudes.len(), longitudes.len(), times.len())?;
    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    let times = arrow_values(&times);
    let ends = ends.as_slice()?;
    validate_ends(lats.len(), ends).map_err(PyValueError::new_err)?;
    Ok(py.detach(|| {
        operation(
            TimedCoordinateView {
                latitudes: lats,
                longitudes: lngs,
                times,
            },
            ends,
        )
    }))
}

/// Indexed counterpart of [`run_presorted_timed_coordinate_arrow`]: extracts and validates a
/// (possibly nullable) lat/lng/time triple alongside row indices/ends, computes `valid_rows`,
/// then runs `operation` under `py.detach`.
pub fn run_indexed_timed_coordinate_arrow<'py, T, F>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    times: ArrowPyArray,
    indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
    operation: F,
) -> PyResult<T>
where
    F: FnOnce(IndexedTimedCoordinateView<'_>) -> T + Send,
    T: Send,
{
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let times = as_nullable_f64_array(times, "times")?;
    validate_timed_coordinate_lengths(latitudes.len(), longitudes.len(), times.len())?;
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;
    validate_indexed_ends(latitudes.len(), indices, ends)?;

    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes, &times]);
    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    let times = arrow_values(&times);
    Ok(py.detach(|| {
        operation(IndexedTimedCoordinateView {
            coordinates: TimedCoordinateView {
                latitudes: lats,
                longitudes: lngs,
                times,
            },
            indices,
            ends,
            valid_rows: valid_rows.as_deref(),
        })
    }))
}

/// Millisecond-`i64` counterpart of [`run_presorted_timed_coordinate_arrow`]: the
/// timestamp array is extracted as `i64` milliseconds (matching
/// `fastmob.utils._common._extract_timestamps`'s default) and converted to `f64`
/// seconds once, here at the FFI boundary, so `operation` sees exactly the same
/// `TimedCoordinateView` seconds shape the core kernels have always taken --
/// no core kernel needs to change for the ms/i64 migration.
pub fn run_presorted_timed_coordinate_arrow_ms<'py, T, F>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_ms: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
    operation: F,
) -> PyResult<T>
where
    F: FnOnce(TimedCoordinateView<'_>, &[usize]) -> T + Send,
    T: Send,
{
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps_ms = as_i64_array(timestamps_ms, "timestamps_ms")?;
    validate_timed_coordinate_lengths(latitudes.len(), longitudes.len(), timestamps_ms.len())?;
    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    let times_s: Vec<f64> = ms_to_seconds(arrow_i64_values(&timestamps_ms));
    let ends = ends.as_slice()?;
    validate_ends(lats.len(), ends).map_err(PyValueError::new_err)?;
    Ok(py.detach(|| {
        operation(
            TimedCoordinateView {
                latitudes: lats,
                longitudes: lngs,
                times: &times_s,
            },
            ends,
        )
    }))
}

/// Millisecond-`i64` counterpart of [`run_indexed_timed_coordinate_arrow`]. See
/// [`run_presorted_timed_coordinate_arrow_ms`] for why the ms->seconds
/// conversion lives here instead of in the core kernels.
pub fn run_indexed_timed_coordinate_arrow_ms<'py, T, F>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_ms: ArrowPyArray,
    indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
    operation: F,
) -> PyResult<T>
where
    F: FnOnce(IndexedTimedCoordinateView<'_>) -> T + Send,
    T: Send,
{
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let timestamps_ms = as_nullable_i64_array(timestamps_ms, "timestamps_ms")?;
    validate_timed_coordinate_lengths(latitudes.len(), longitudes.len(), timestamps_ms.len())?;
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;
    validate_indexed_ends(latitudes.len(), indices, ends)?;

    let valid_rows = arrow_valid_rows_f64_i64(&[&latitudes, &longitudes], &timestamps_ms);
    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    let times_s: Vec<f64> = ms_to_seconds(arrow_i64_values(&timestamps_ms));
    Ok(py.detach(|| {
        operation(IndexedTimedCoordinateView {
            coordinates: TimedCoordinateView {
                latitudes: lats,
                longitudes: lngs,
                times: &times_s,
            },
            indices,
            ends,
            valid_rows: valid_rows.as_deref(),
        })
    }))
}

/// Group-coordinate counterpart of [`run_presorted_coordinate_arrow`]: extracts and
/// validates a presorted group-center pair (non-nullable) alongside a per-row coordinate
/// pair (nullable, with `valid_rows` computed), then runs `operation` under `py.detach`.
#[allow(clippy::too_many_arguments)]
pub fn run_presorted_group_coordinate_arrow<'py, T, F>(
    py: Python<'py>,
    group_latitudes: ArrowPyArray,
    group_longitudes: ArrowPyArray,
    row_latitudes: ArrowPyArray,
    row_longitudes: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
    operation: F,
) -> PyResult<T>
where
    F: FnOnce(GroupCoordinateView<'_>, &[usize]) -> T + Send,
    T: Send,
{
    let group_latitudes = as_f64_array(group_latitudes, "group_latitudes")?;
    let group_longitudes = as_f64_array(group_longitudes, "group_longitudes")?;
    let row_latitudes = as_nullable_f64_array(row_latitudes, "row_latitudes")?;
    let row_longitudes = as_nullable_f64_array(row_longitudes, "row_longitudes")?;
    let group_lats = arrow_values(&group_latitudes);
    let group_lngs = arrow_values(&group_longitudes);
    let row_lats = arrow_values(&row_latitudes);
    let row_lngs = arrow_values(&row_longitudes);
    let ends = ends.as_slice()?;
    validate_group_coordinate_lengths(
        group_lats.len(),
        group_lngs.len(),
        row_lats.len(),
        row_lngs.len(),
        ends.len(),
    )?;
    validate_ends(row_lats.len(), ends).map_err(PyValueError::new_err)?;
    let valid_rows = arrow_valid_rows(&[&row_latitudes, &row_longitudes]);
    Ok(py.detach(|| {
        operation(
            GroupCoordinateView {
                group_latitudes: group_lats,
                group_longitudes: group_lngs,
                row_latitudes: row_lats,
                row_longitudes: row_lngs,
                valid_rows: valid_rows.as_deref(),
            },
            ends,
        )
    }))
}

/// Indexed counterpart of [`run_presorted_group_coordinate_arrow`].
#[allow(clippy::too_many_arguments)]
pub fn run_indexed_group_coordinate_arrow<'py, T, F>(
    py: Python<'py>,
    group_latitudes: ArrowPyArray,
    group_longitudes: ArrowPyArray,
    row_latitudes: ArrowPyArray,
    row_longitudes: ArrowPyArray,
    indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
    operation: F,
) -> PyResult<T>
where
    F: FnOnce(IndexedGroupCoordinateView<'_>) -> T + Send,
    T: Send,
{
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;

    let group_latitudes = as_f64_array(group_latitudes, "group_latitudes")?;
    let group_longitudes = as_f64_array(group_longitudes, "group_longitudes")?;
    let row_latitudes = as_nullable_f64_array(row_latitudes, "row_latitudes")?;
    let row_longitudes = as_nullable_f64_array(row_longitudes, "row_longitudes")?;
    let group_lats = arrow_values(&group_latitudes);
    let group_lngs = arrow_values(&group_longitudes);
    let row_lats = arrow_values(&row_latitudes);
    let row_lngs = arrow_values(&row_longitudes);
    validate_group_coordinate_lengths(
        group_lats.len(),
        group_lngs.len(),
        row_lats.len(),
        row_lngs.len(),
        ends.len(),
    )?;
    validate_indexed_ends(row_lats.len(), indices, ends)?;
    let valid_rows = arrow_valid_rows(&[&row_latitudes, &row_longitudes]);
    Ok(py.detach(|| {
        operation(IndexedGroupCoordinateView {
            coordinates: GroupCoordinateView {
                group_latitudes: group_lats,
                group_longitudes: group_lngs,
                row_latitudes: row_lats,
                row_longitudes: row_lngs,
                valid_rows: valid_rows.as_deref(),
            },
            indices,
            ends,
            valid_rows: valid_rows.as_deref(),
        })
    }))
}

/// Runs a two-independent-whole-sequence -> scalar operation (the shape
/// shared identically by all 4 `trajectory_distance` methods: DTW, discrete
/// Fréchet, Hausdorff, LCSS). No grouping/indices/ends at all -- each side is
/// exactly one trajectory's full coordinate sequence, already sorted and
/// null-cleaned in Python (see `fastmob/trajectory/_distance.py`'s module
/// docstring for why that Python-side cleaning replaces the usual
/// `valid_rows`/`Option<&[bool]>` nullable-Arrow path here).
pub fn run_two_sequence_arrow<F>(
    py: Python<'_>,
    latitudes_a: ArrowPyArray,
    longitudes_a: ArrowPyArray,
    latitudes_b: ArrowPyArray,
    longitudes_b: ArrowPyArray,
    operation: F,
) -> PyResult<f64>
where
    F: FnOnce(CoordinateView<'_>, CoordinateView<'_>) -> Result<f64, String> + Send,
{
    let latitudes_a = as_f64_array(latitudes_a, "latitudes_a")?;
    let longitudes_a = as_f64_array(longitudes_a, "longitudes_a")?;
    let latitudes_b = as_f64_array(latitudes_b, "latitudes_b")?;
    let longitudes_b = as_f64_array(longitudes_b, "longitudes_b")?;
    validate_coordinate_lengths(latitudes_a.len(), longitudes_a.len())?;
    validate_coordinate_lengths(latitudes_b.len(), longitudes_b.len())?;
    let lats_a = arrow_values(&latitudes_a);
    let lngs_a = arrow_values(&longitudes_a);
    let lats_b = arrow_values(&latitudes_b);
    let lngs_b = arrow_values(&longitudes_b);
    py.detach(|| {
        operation(
            CoordinateView {
                latitudes: lats_a,
                longitudes: lngs_a,
            },
            CoordinateView {
                latitudes: lats_b,
                longitudes: lngs_b,
            },
        )
    })
    .map_err(PyValueError::new_err)
}
