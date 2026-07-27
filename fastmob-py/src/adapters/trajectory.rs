use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use fastmob_core::utils::validate_ends;

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, f64_results_into_arrow,
    u64_results_into_arrow, validate_indexed_ends,
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

pub type PyF64Result = Py<PyAny>;
pub type PyU64Result = Py<PyAny>;
pub type PyF64Pair = (Py<PyAny>, Py<PyAny>);

const COORDINATE_BACKEND_ERROR: &str =
    "latitudes and longitudes must both be NumPy arrays or both be Arrow arrays";

fn is_arrow_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.hasattr("__arrow_c_array__")
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

fn numpy_f64_output<'py>(py: Python<'py>, values: Vec<f64>) -> Py<PyAny> {
    values.into_pyarray(py).into_any().unbind()
}

fn numpy_u64_output<'py>(py: Python<'py>, values: Vec<u64>) -> Py<PyAny> {
    values.into_pyarray(py).into_any().unbind()
}

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

fn arrow_u64_output(py: Python<'_>, values: Vec<u64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, u64_results_into_arrow(values))?.into_any())
}

/// Extracts and validates a presorted pair of Arrow coordinate arrays, then runs
/// `operation` under `py.detach` with clean coordinate slices. Unlike
/// [`run_presorted_coordinate_f64`], the result is handed back as-is so callers
/// with different output shapes (a plain `Vec<f64>`, a `(Vec<f64>, Vec<bool>)`
/// validity pair, a fallible `Result<_, String>`, ...) can wrap it themselves.
pub fn run_presorted_coordinate_arrow<'py, T, F>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    ends: PyReadonlyArray1<'py, usize>,
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
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
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

pub fn run_presorted_coordinate_f64<'py, F>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
    operation: F,
) -> PyResult<PyF64Result>
where
    F: FnOnce(CoordinateView<'_>, &[usize]) -> Result<Vec<f64>, String> + Send,
{
    if let (Ok(latitudes), Ok(longitudes)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let lats = latitudes.as_slice()?;
        let lngs = longitudes.as_slice()?;
        let ends = ends.as_slice()?;
        validate_coordinate_lengths(lats.len(), lngs.len())?;
        validate_ends(lats.len(), ends).map_err(PyValueError::new_err)?;
        let values = py
            .detach(|| {
                operation(
                    CoordinateView {
                        latitudes: lats,
                        longitudes: lngs,
                    },
                    ends,
                )
            })
            .map_err(PyValueError::new_err)?;
        return Ok(numpy_f64_output(py, values));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? {
        let latitudes = as_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes = as_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        validate_coordinate_lengths(latitudes.len(), longitudes.len())?;
        let lats = arrow_values(&latitudes);
        let lngs = arrow_values(&longitudes);
        let ends = ends.as_slice()?;
        validate_ends(lats.len(), ends).map_err(PyValueError::new_err)?;
        let values = py
            .detach(|| {
                operation(
                    CoordinateView {
                        latitudes: lats,
                        longitudes: lngs,
                    },
                    ends,
                )
            })
            .map_err(PyValueError::new_err)?;
        return arrow_f64_output(py, values);
    }

    Err(PyTypeError::new_err(COORDINATE_BACKEND_ERROR))
}

pub fn run_indexed_coordinate_f64<'py, F>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    operation: F,
) -> PyResult<PyF64Result>
where
    F: FnOnce(IndexedCoordinateView<'_>) -> Result<Vec<f64>, String> + Send,
{
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;

    if let (Ok(latitudes), Ok(longitudes)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let lats = latitudes.as_slice()?;
        let lngs = longitudes.as_slice()?;
        validate_coordinate_lengths(lats.len(), lngs.len())?;
        validate_indexed_ends(lats.len(), indices, ends)?;
        let values = py
            .detach(|| {
                operation(IndexedCoordinateView {
                    coordinates: CoordinateView {
                        latitudes: lats,
                        longitudes: lngs,
                    },
                    indices,
                    ends,
                    valid_rows: None,
                })
            })
            .map_err(PyValueError::new_err)?;
        return Ok(numpy_f64_output(py, values));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? {
        let latitudes = as_nullable_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes =
            as_nullable_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        validate_coordinate_lengths(latitudes.len(), longitudes.len())?;
        validate_indexed_ends(latitudes.len(), indices, ends)?;

        let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
        let lats = arrow_values(&latitudes);
        let lngs = arrow_values(&longitudes);
        let values = py
            .detach(|| {
                operation(IndexedCoordinateView {
                    coordinates: CoordinateView {
                        latitudes: lats,
                        longitudes: lngs,
                    },
                    indices,
                    ends,
                    valid_rows: valid_rows.as_deref(),
                })
            })
            .map_err(PyValueError::new_err)?;
        return arrow_f64_output(py, values);
    }

    Err(PyTypeError::new_err(COORDINATE_BACKEND_ERROR))
}

pub fn run_presorted_coordinate_u64<'py, F>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
    operation: F,
) -> PyResult<PyU64Result>
where
    F: FnOnce(CoordinateView<'_>, &[usize]) -> Result<Vec<u64>, String> + Send,
{
    if let (Ok(latitudes), Ok(longitudes)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let lats = latitudes.as_slice()?;
        let lngs = longitudes.as_slice()?;
        let ends = ends.as_slice()?;
        validate_coordinate_lengths(lats.len(), lngs.len())?;
        validate_ends(lats.len(), ends).map_err(PyValueError::new_err)?;
        let values = py
            .detach(|| {
                operation(
                    CoordinateView {
                        latitudes: lats,
                        longitudes: lngs,
                    },
                    ends,
                )
            })
            .map_err(PyValueError::new_err)?;
        return Ok(numpy_u64_output(py, values));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? {
        let latitudes = as_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes = as_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        validate_coordinate_lengths(latitudes.len(), longitudes.len())?;
        let lats = arrow_values(&latitudes);
        let lngs = arrow_values(&longitudes);
        let ends = ends.as_slice()?;
        validate_ends(lats.len(), ends).map_err(PyValueError::new_err)?;
        let values = py
            .detach(|| {
                operation(
                    CoordinateView {
                        latitudes: lats,
                        longitudes: lngs,
                    },
                    ends,
                )
            })
            .map_err(PyValueError::new_err)?;
        return arrow_u64_output(py, values);
    }

    Err(PyTypeError::new_err(COORDINATE_BACKEND_ERROR))
}

pub fn run_indexed_coordinate_u64<'py, F>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    operation: F,
) -> PyResult<PyU64Result>
where
    F: FnOnce(IndexedCoordinateView<'_>) -> Result<Vec<u64>, String> + Send,
{
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;

    if let (Ok(latitudes), Ok(longitudes)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let lats = latitudes.as_slice()?;
        let lngs = longitudes.as_slice()?;
        validate_coordinate_lengths(lats.len(), lngs.len())?;
        validate_indexed_ends(lats.len(), indices, ends)?;
        let values = py
            .detach(|| {
                operation(IndexedCoordinateView {
                    coordinates: CoordinateView {
                        latitudes: lats,
                        longitudes: lngs,
                    },
                    indices,
                    ends,
                    valid_rows: None,
                })
            })
            .map_err(PyValueError::new_err)?;
        return Ok(numpy_u64_output(py, values));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? {
        let latitudes = as_nullable_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes =
            as_nullable_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        validate_coordinate_lengths(latitudes.len(), longitudes.len())?;
        validate_indexed_ends(latitudes.len(), indices, ends)?;

        let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
        let lats = arrow_values(&latitudes);
        let lngs = arrow_values(&longitudes);
        let values = py
            .detach(|| {
                operation(IndexedCoordinateView {
                    coordinates: CoordinateView {
                        latitudes: lats,
                        longitudes: lngs,
                    },
                    indices,
                    ends,
                    valid_rows: valid_rows.as_deref(),
                })
            })
            .map_err(PyValueError::new_err)?;
        return arrow_u64_output(py, values);
    }

    Err(PyTypeError::new_err(COORDINATE_BACKEND_ERROR))
}

pub fn run_presorted_timed_coordinate_f64<'py, F>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    times: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
    operation: F,
) -> PyResult<PyF64Result>
where
    F: FnOnce(TimedCoordinateView<'_>, &[usize]) -> Result<Vec<f64>, String> + Send,
{
    if let (Ok(latitudes), Ok(longitudes), Ok(times)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
        times.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let lats = latitudes.as_slice()?;
        let lngs = longitudes.as_slice()?;
        let times = times.as_slice()?;
        let ends = ends.as_slice()?;
        validate_timed_coordinate_lengths(lats.len(), lngs.len(), times.len())?;
        validate_ends(lats.len(), ends).map_err(PyValueError::new_err)?;
        let values = py
            .detach(|| {
                operation(
                    TimedCoordinateView {
                        latitudes: lats,
                        longitudes: lngs,
                        times,
                    },
                    ends,
                )
            })
            .map_err(PyValueError::new_err)?;
        return Ok(numpy_f64_output(py, values));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? && is_arrow_array(times)? {
        let latitudes = as_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes = as_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        let times = as_f64_array(times.extract::<ArrowPyArray>()?, "times")?;
        validate_timed_coordinate_lengths(latitudes.len(), longitudes.len(), times.len())?;
        let lats = arrow_values(&latitudes);
        let lngs = arrow_values(&longitudes);
        let times = arrow_values(&times);
        let ends = ends.as_slice()?;
        validate_ends(lats.len(), ends).map_err(PyValueError::new_err)?;
        let values = py
            .detach(|| {
                operation(
                    TimedCoordinateView {
                        latitudes: lats,
                        longitudes: lngs,
                        times,
                    },
                    ends,
                )
            })
            .map_err(PyValueError::new_err)?;
        return arrow_f64_output(py, values);
    }

    Err(PyTypeError::new_err(COORDINATE_BACKEND_ERROR))
}

pub fn run_indexed_timed_coordinate_f64<'py, F>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    times: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    operation: F,
) -> PyResult<PyF64Result>
where
    F: FnOnce(IndexedTimedCoordinateView<'_>) -> Result<Vec<f64>, String> + Send,
{
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;

    if let (Ok(latitudes), Ok(longitudes), Ok(times)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
        times.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let lats = latitudes.as_slice()?;
        let lngs = longitudes.as_slice()?;
        let times = times.as_slice()?;
        validate_timed_coordinate_lengths(lats.len(), lngs.len(), times.len())?;
        validate_indexed_ends(lats.len(), indices, ends)?;
        let values = py
            .detach(|| {
                operation(IndexedTimedCoordinateView {
                    coordinates: TimedCoordinateView {
                        latitudes: lats,
                        longitudes: lngs,
                        times,
                    },
                    indices,
                    ends,
                    valid_rows: None,
                })
            })
            .map_err(PyValueError::new_err)?;
        return Ok(numpy_f64_output(py, values));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? && is_arrow_array(times)? {
        let latitudes = as_nullable_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes =
            as_nullable_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        let times = as_nullable_f64_array(times.extract::<ArrowPyArray>()?, "times")?;
        validate_timed_coordinate_lengths(latitudes.len(), longitudes.len(), times.len())?;
        validate_indexed_ends(latitudes.len(), indices, ends)?;

        let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes, &times]);
        let lats = arrow_values(&latitudes);
        let lngs = arrow_values(&longitudes);
        let times = arrow_values(&times);
        let values = py
            .detach(|| {
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
            })
            .map_err(PyValueError::new_err)?;
        return arrow_f64_output(py, values);
    }

    Err(PyTypeError::new_err(COORDINATE_BACKEND_ERROR))
}

pub fn run_presorted_timed_coordinate_f64_pair<'py, F>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    times: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
    operation: F,
) -> PyResult<PyF64Pair>
where
    F: FnOnce(TimedCoordinateView<'_>, &[usize]) -> Result<(Vec<f64>, Vec<f64>), String> + Send,
{
    if let (Ok(latitudes), Ok(longitudes), Ok(times)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
        times.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let lats = latitudes.as_slice()?;
        let lngs = longitudes.as_slice()?;
        let times = times.as_slice()?;
        let ends = ends.as_slice()?;
        validate_timed_coordinate_lengths(lats.len(), lngs.len(), times.len())?;
        validate_ends(lats.len(), ends).map_err(PyValueError::new_err)?;
        let (first, second) = py
            .detach(|| {
                operation(
                    TimedCoordinateView {
                        latitudes: lats,
                        longitudes: lngs,
                        times,
                    },
                    ends,
                )
            })
            .map_err(PyValueError::new_err)?;
        return Ok((numpy_f64_output(py, first), numpy_f64_output(py, second)));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? && is_arrow_array(times)? {
        let latitudes = as_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes = as_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        let times = as_f64_array(times.extract::<ArrowPyArray>()?, "times")?;
        validate_timed_coordinate_lengths(latitudes.len(), longitudes.len(), times.len())?;
        let lats = arrow_values(&latitudes);
        let lngs = arrow_values(&longitudes);
        let times = arrow_values(&times);
        let ends = ends.as_slice()?;
        validate_ends(lats.len(), ends).map_err(PyValueError::new_err)?;
        let (first, second) = py
            .detach(|| {
                operation(
                    TimedCoordinateView {
                        latitudes: lats,
                        longitudes: lngs,
                        times,
                    },
                    ends,
                )
            })
            .map_err(PyValueError::new_err)?;
        return Ok((arrow_f64_output(py, first)?, arrow_f64_output(py, second)?));
    }

    Err(PyTypeError::new_err(COORDINATE_BACKEND_ERROR))
}

pub fn run_indexed_timed_coordinate_f64_pair<'py, F>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    times: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    operation: F,
) -> PyResult<PyF64Pair>
where
    F: FnOnce(IndexedTimedCoordinateView<'_>) -> Result<(Vec<f64>, Vec<f64>), String> + Send,
{
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;

    if let (Ok(latitudes), Ok(longitudes), Ok(times)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
        times.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let lats = latitudes.as_slice()?;
        let lngs = longitudes.as_slice()?;
        let times = times.as_slice()?;
        validate_timed_coordinate_lengths(lats.len(), lngs.len(), times.len())?;
        validate_indexed_ends(lats.len(), indices, ends)?;
        let (first, second) = py
            .detach(|| {
                operation(IndexedTimedCoordinateView {
                    coordinates: TimedCoordinateView {
                        latitudes: lats,
                        longitudes: lngs,
                        times,
                    },
                    indices,
                    ends,
                    valid_rows: None,
                })
            })
            .map_err(PyValueError::new_err)?;
        return Ok((numpy_f64_output(py, first), numpy_f64_output(py, second)));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? && is_arrow_array(times)? {
        let latitudes = as_nullable_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes =
            as_nullable_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        let times = as_nullable_f64_array(times.extract::<ArrowPyArray>()?, "times")?;
        validate_timed_coordinate_lengths(latitudes.len(), longitudes.len(), times.len())?;
        validate_indexed_ends(latitudes.len(), indices, ends)?;

        let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes, &times]);
        let lats = arrow_values(&latitudes);
        let lngs = arrow_values(&longitudes);
        let times = arrow_values(&times);
        let (first, second) = py
            .detach(|| {
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
            })
            .map_err(PyValueError::new_err)?;
        return Ok((arrow_f64_output(py, first)?, arrow_f64_output(py, second)?));
    }

    Err(PyTypeError::new_err(COORDINATE_BACKEND_ERROR))
}

#[allow(clippy::too_many_arguments)]
pub fn run_presorted_group_coordinate_f64<'py, F>(
    py: Python<'py>,
    group_latitudes: &Bound<'py, PyAny>,
    group_longitudes: &Bound<'py, PyAny>,
    row_latitudes: &Bound<'py, PyAny>,
    row_longitudes: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
    operation: F,
) -> PyResult<PyF64Result>
where
    F: FnOnce(GroupCoordinateView<'_>, &[usize]) -> Result<Vec<f64>, String> + Send,
{
    if let (Ok(group_latitudes), Ok(group_longitudes), Ok(row_latitudes), Ok(row_longitudes)) = (
        group_latitudes.extract::<PyReadonlyArray1<f64>>(),
        group_longitudes.extract::<PyReadonlyArray1<f64>>(),
        row_latitudes.extract::<PyReadonlyArray1<f64>>(),
        row_longitudes.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let group_lats = group_latitudes.as_slice()?;
        let group_lngs = group_longitudes.as_slice()?;
        let row_lats = row_latitudes.as_slice()?;
        let row_lngs = row_longitudes.as_slice()?;
        let ends = ends.as_slice()?;
        validate_group_coordinate_lengths(
            group_lats.len(),
            group_lngs.len(),
            row_lats.len(),
            row_lngs.len(),
            ends.len(),
        )?;
        validate_ends(row_lats.len(), ends).map_err(PyValueError::new_err)?;
        let values = py
            .detach(|| {
                operation(
                    GroupCoordinateView {
                        group_latitudes: group_lats,
                        group_longitudes: group_lngs,
                        row_latitudes: row_lats,
                        row_longitudes: row_lngs,
                        valid_rows: None,
                    },
                    ends,
                )
            })
            .map_err(PyValueError::new_err)?;
        return Ok(numpy_f64_output(py, values));
    }

    if is_arrow_array(group_latitudes)?
        && is_arrow_array(group_longitudes)?
        && is_arrow_array(row_latitudes)?
        && is_arrow_array(row_longitudes)?
    {
        let group_latitudes = as_f64_array(
            group_latitudes.extract::<ArrowPyArray>()?,
            "group_latitudes",
        )?;
        let group_longitudes = as_f64_array(
            group_longitudes.extract::<ArrowPyArray>()?,
            "group_longitudes",
        )?;
        let row_latitudes =
            as_nullable_f64_array(row_latitudes.extract::<ArrowPyArray>()?, "row_latitudes")?;
        let row_longitudes =
            as_nullable_f64_array(row_longitudes.extract::<ArrowPyArray>()?, "row_longitudes")?;
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
        let values = py
            .detach(|| {
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
            })
            .map_err(PyValueError::new_err)?;
        return arrow_f64_output(py, values);
    }

    Err(PyTypeError::new_err(COORDINATE_BACKEND_ERROR))
}

#[allow(clippy::too_many_arguments)]
pub fn run_indexed_group_coordinate_f64<'py, F>(
    py: Python<'py>,
    group_latitudes: &Bound<'py, PyAny>,
    group_longitudes: &Bound<'py, PyAny>,
    row_latitudes: &Bound<'py, PyAny>,
    row_longitudes: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    operation: F,
) -> PyResult<PyF64Result>
where
    F: FnOnce(IndexedGroupCoordinateView<'_>) -> Result<Vec<f64>, String> + Send,
{
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;

    if let (Ok(group_latitudes), Ok(group_longitudes), Ok(row_latitudes), Ok(row_longitudes)) = (
        group_latitudes.extract::<PyReadonlyArray1<f64>>(),
        group_longitudes.extract::<PyReadonlyArray1<f64>>(),
        row_latitudes.extract::<PyReadonlyArray1<f64>>(),
        row_longitudes.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let group_lats = group_latitudes.as_slice()?;
        let group_lngs = group_longitudes.as_slice()?;
        let row_lats = row_latitudes.as_slice()?;
        let row_lngs = row_longitudes.as_slice()?;
        validate_group_coordinate_lengths(
            group_lats.len(),
            group_lngs.len(),
            row_lats.len(),
            row_lngs.len(),
            ends.len(),
        )?;
        validate_indexed_ends(row_lats.len(), indices, ends)?;
        let values = py
            .detach(|| {
                operation(IndexedGroupCoordinateView {
                    coordinates: GroupCoordinateView {
                        group_latitudes: group_lats,
                        group_longitudes: group_lngs,
                        row_latitudes: row_lats,
                        row_longitudes: row_lngs,
                        valid_rows: None,
                    },
                    indices,
                    ends,
                    valid_rows: None,
                })
            })
            .map_err(PyValueError::new_err)?;
        return Ok(numpy_f64_output(py, values));
    }

    if is_arrow_array(group_latitudes)?
        && is_arrow_array(group_longitudes)?
        && is_arrow_array(row_latitudes)?
        && is_arrow_array(row_longitudes)?
    {
        let group_latitudes = as_f64_array(
            group_latitudes.extract::<ArrowPyArray>()?,
            "group_latitudes",
        )?;
        let group_longitudes = as_f64_array(
            group_longitudes.extract::<ArrowPyArray>()?,
            "group_longitudes",
        )?;
        let row_latitudes =
            as_nullable_f64_array(row_latitudes.extract::<ArrowPyArray>()?, "row_latitudes")?;
        let row_longitudes =
            as_nullable_f64_array(row_longitudes.extract::<ArrowPyArray>()?, "row_longitudes")?;
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
        let values = py
            .detach(|| {
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
            })
            .map_err(PyValueError::new_err)?;
        return arrow_f64_output(py, values);
    }

    Err(PyTypeError::new_err(COORDINATE_BACKEND_ERROR))
}

/// Runs a two-independent-whole-sequence -> scalar operation (the shape
/// shared identically by all 4 `trajectory_distance` methods: DTW, discrete
/// Fréchet, Hausdorff, LCSS). No grouping/indices/ends at all -- each side is
/// exactly one trajectory's full coordinate sequence, already sorted and
/// null-cleaned in Python (see `fastmob/trajectory/_distance.py`'s module
/// docstring for why that Python-side cleaning replaces the usual
/// `valid_rows`/`Option<&[bool]>` nullable-Arrow path here).
pub fn run_two_sequence_f64<'py, F>(
    py: Python<'py>,
    latitudes_a: &Bound<'py, PyAny>,
    longitudes_a: &Bound<'py, PyAny>,
    latitudes_b: &Bound<'py, PyAny>,
    longitudes_b: &Bound<'py, PyAny>,
    operation: F,
) -> PyResult<f64>
where
    F: FnOnce(CoordinateView<'_>, CoordinateView<'_>) -> Result<f64, String> + Send,
{
    if let (Ok(lat_a), Ok(lng_a), Ok(lat_b), Ok(lng_b)) = (
        latitudes_a.extract::<PyReadonlyArray1<f64>>(),
        longitudes_a.extract::<PyReadonlyArray1<f64>>(),
        latitudes_b.extract::<PyReadonlyArray1<f64>>(),
        longitudes_b.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let lats_a = lat_a.as_slice()?;
        let lngs_a = lng_a.as_slice()?;
        let lats_b = lat_b.as_slice()?;
        let lngs_b = lng_b.as_slice()?;
        validate_coordinate_lengths(lats_a.len(), lngs_a.len())?;
        validate_coordinate_lengths(lats_b.len(), lngs_b.len())?;
        return py
            .detach(|| {
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
            .map_err(PyValueError::new_err);
    }

    if is_arrow_array(latitudes_a)?
        && is_arrow_array(longitudes_a)?
        && is_arrow_array(latitudes_b)?
        && is_arrow_array(longitudes_b)?
    {
        let latitudes_a = as_f64_array(latitudes_a.extract::<ArrowPyArray>()?, "latitudes_a")?;
        let longitudes_a = as_f64_array(longitudes_a.extract::<ArrowPyArray>()?, "longitudes_a")?;
        let latitudes_b = as_f64_array(latitudes_b.extract::<ArrowPyArray>()?, "latitudes_b")?;
        let longitudes_b = as_f64_array(longitudes_b.extract::<ArrowPyArray>()?, "longitudes_b")?;
        validate_coordinate_lengths(latitudes_a.len(), longitudes_a.len())?;
        validate_coordinate_lengths(latitudes_b.len(), longitudes_b.len())?;
        let lats_a = arrow_values(&latitudes_a);
        let lngs_a = arrow_values(&longitudes_a);
        let lats_b = arrow_values(&latitudes_b);
        let lngs_b = arrow_values(&longitudes_b);
        return py
            .detach(|| {
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
            .map_err(PyValueError::new_err);
    }

    Err(PyTypeError::new_err(COORDINATE_BACKEND_ERROR))
}
