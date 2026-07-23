use std::str::FromStr;

use fastmob_core::trajectory::interpolate_at::{
    PositionQueryMethod, interpolate_at_indexed_impl, interpolate_at_presorted_impl,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, f64_results_into_arrow,
    validate_indexed_ends,
};

fn is_arrow_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.hasattr("__arrow_c_array__")
}

fn numpy_f64_output(py: Python<'_>, values: Vec<f64>) -> Py<PyAny> {
    values.into_pyarray(py).into_any().unbind()
}

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

type InterpolateAtOutput<'py> = (Py<PyAny>, Py<PyAny>, Bound<'py, PyArray1<bool>>);

const BACKEND_ERROR: &str =
    "latitudes, longitudes, and timestamps_s must all be NumPy arrays or all be Arrow arrays";

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn interpolate_at_indexed<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    timestamps_s: &Bound<'py, PyAny>,
    sorted_indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    query_times_s: PyReadonlyArray1<'py, f64>,
    method: &str,
) -> PyResult<InterpolateAtOutput<'py>> {
    let sorted_indices = sorted_indices.as_slice()?;
    let ends = ends.as_slice()?;
    let query_times_s = query_times_s.as_slice()?;
    let method = PositionQueryMethod::from_str(method).map_err(PyValueError::new_err)?;

    if let (Ok(latitudes), Ok(longitudes), Ok(timestamps_s)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
        timestamps_s.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let lats = latitudes.as_slice()?;
        let lngs = longitudes.as_slice()?;
        let times = timestamps_s.as_slice()?;
        validate_indexed_ends(lats.len(), sorted_indices, ends)?;
        let (out_lats, out_lngs, out_valid) = py
            .detach(|| {
                interpolate_at_indexed_impl(
                    lats,
                    lngs,
                    times,
                    sorted_indices,
                    ends,
                    None,
                    query_times_s,
                    method,
                )
            })
            .map_err(PyValueError::new_err)?;
        return Ok((
            numpy_f64_output(py, out_lats),
            numpy_f64_output(py, out_lngs),
            out_valid.into_pyarray(py),
        ));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? && is_arrow_array(timestamps_s)? {
        let latitudes = as_nullable_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes =
            as_nullable_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        let timestamps_s =
            as_nullable_f64_array(timestamps_s.extract::<ArrowPyArray>()?, "timestamps_s")?;
        validate_indexed_ends(latitudes.len(), sorted_indices, ends)?;
        let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes, &timestamps_s]);
        let (out_lats, out_lngs, out_valid) = py
            .detach(|| {
                interpolate_at_indexed_impl(
                    arrow_values(&latitudes),
                    arrow_values(&longitudes),
                    arrow_values(&timestamps_s),
                    sorted_indices,
                    ends,
                    valid_rows.as_deref(),
                    query_times_s,
                    method,
                )
            })
            .map_err(PyValueError::new_err)?;
        return Ok((
            arrow_f64_output(py, out_lats)?,
            arrow_f64_output(py, out_lngs)?,
            out_valid.into_pyarray(py),
        ));
    }

    Err(PyTypeError::new_err(BACKEND_ERROR))
}

#[pyfunction]
pub fn interpolate_at_presorted<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    timestamps_s: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
    query_times_s: PyReadonlyArray1<'py, f64>,
    method: &str,
) -> PyResult<InterpolateAtOutput<'py>> {
    let ends = ends.as_slice()?;
    let query_times_s = query_times_s.as_slice()?;
    let method = PositionQueryMethod::from_str(method).map_err(PyValueError::new_err)?;

    if let (Ok(latitudes), Ok(longitudes), Ok(timestamps_s)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
        timestamps_s.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let lats = latitudes.as_slice()?;
        let lngs = longitudes.as_slice()?;
        let times = timestamps_s.as_slice()?;
        let (out_lats, out_lngs, out_valid) = py
            .detach(|| {
                interpolate_at_presorted_impl(lats, lngs, times, ends, query_times_s, method)
            })
            .map_err(PyValueError::new_err)?;
        return Ok((
            numpy_f64_output(py, out_lats),
            numpy_f64_output(py, out_lngs),
            out_valid.into_pyarray(py),
        ));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? && is_arrow_array(timestamps_s)? {
        let latitudes = as_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes = as_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        let timestamps_s = as_f64_array(timestamps_s.extract::<ArrowPyArray>()?, "timestamps_s")?;
        let (out_lats, out_lngs, out_valid) = py
            .detach(|| {
                interpolate_at_presorted_impl(
                    arrow_values(&latitudes),
                    arrow_values(&longitudes),
                    arrow_values(&timestamps_s),
                    ends,
                    query_times_s,
                    method,
                )
            })
            .map_err(PyValueError::new_err)?;
        return Ok((
            arrow_f64_output(py, out_lats)?,
            arrow_f64_output(py, out_lngs)?,
            out_valid.into_pyarray(py),
        ));
    }

    Err(PyTypeError::new_err(BACKEND_ERROR))
}
