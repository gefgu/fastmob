use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use fastmob_core::utils::validate_ends;

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, f64_results_into_arrow,
    validate_indexed_ends,
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

pub type PyF64WithValidity<'py> = (Py<PyAny>, Bound<'py, PyArray1<bool>>);
pub type PyF64Result = Py<PyAny>;

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

fn numpy_f64_output<'py>(py: Python<'py>, values: Vec<f64>) -> Py<PyAny> {
    values.into_pyarray(py).into_any().unbind()
}

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
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
