use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
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

pub type PyF64WithValidity<'py> = (Py<PyAny>, Bound<'py, PyArray1<bool>>);
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

fn validity_output<'py>(py: Python<'py>, validity: Vec<bool>) -> Bound<'py, PyArray1<bool>> {
    validity.into_pyarray(py)
}

pub fn run_presorted_coordinate_f64_with_validity<'py, F>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
    operation: F,
) -> PyResult<PyF64WithValidity<'py>>
where
    F: FnOnce(CoordinateView<'_>, &[usize]) -> (Vec<f64>, Vec<bool>) + Send,
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
        let (values, validity) = py.detach(|| {
            operation(
                CoordinateView {
                    latitudes: lats,
                    longitudes: lngs,
                },
                ends,
            )
        });
        return Ok((numpy_f64_output(py, values), validity_output(py, validity)));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? {
        let latitudes = as_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes = as_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        validate_coordinate_lengths(latitudes.len(), longitudes.len())?;
        let lats = arrow_values(&latitudes);
        let lngs = arrow_values(&longitudes);
        let ends = ends.as_slice()?;
        validate_ends(lats.len(), ends).map_err(PyValueError::new_err)?;
        let (values, validity) = py.detach(|| {
            operation(
                CoordinateView {
                    latitudes: lats,
                    longitudes: lngs,
                },
                ends,
            )
        });
        return Ok((arrow_f64_output(py, values)?, validity_output(py, validity)));
    }

    Err(PyTypeError::new_err(COORDINATE_BACKEND_ERROR))
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

pub fn run_indexed_coordinate_f64_with_validity<'py, F>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    operation: F,
) -> PyResult<PyF64WithValidity<'py>>
where
    F: FnOnce(IndexedCoordinateView<'_>) -> (Vec<f64>, Vec<bool>) + Send,
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
        let (values, validity) = py.detach(|| {
            operation(IndexedCoordinateView {
                coordinates: CoordinateView {
                    latitudes: lats,
                    longitudes: lngs,
                },
                indices,
                ends,
                valid_rows: None,
            })
        });
        return Ok((numpy_f64_output(py, values), validity_output(py, validity)));
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
        let (values, validity) = py.detach(|| {
            operation(IndexedCoordinateView {
                coordinates: CoordinateView {
                    latitudes: lats,
                    longitudes: lngs,
                },
                indices,
                ends,
                valid_rows: valid_rows.as_deref(),
            })
        });
        return Ok((arrow_f64_output(py, values)?, validity_output(py, validity)));
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
