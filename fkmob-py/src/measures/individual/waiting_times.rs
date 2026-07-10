use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use fkmob_core::measures::individual::waiting_times::{
    waiting_times_flat_impl, waiting_times_impl, waiting_times_indexed_flat_impl,
    waiting_times_indexed_impl, waiting_times_seconds as core_waiting_times_seconds,
};

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, f64_results_into_arrow,
};

type PyWaitingTimes<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<f64>>,
);
type PyWaitingTimesArrow<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    PyArray,
);

#[pyfunction]
pub fn waiting_times_seconds<'py>(
    py: Python<'py>,
    timestamps_s: Vec<f64>,
    ends: Vec<usize>,
) -> PyResult<PyWaitingTimes<'py>> {
    let (starts, ends, values) =
        core_waiting_times_seconds(timestamps_s, ends).map_err(PyValueError::new_err)?;
    Ok((
        starts.into_pyarray(py),
        ends.into_pyarray(py),
        values.into_pyarray(py),
    ))
}

#[pyfunction]
pub fn waiting_times_numpy<'py>(
    py: Python<'py>,
    timestamps_s: PyReadonlyArray1<'py, f64>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyWaitingTimes<'py>> {
    let (starts, ends, values) = waiting_times_impl(timestamps_s.as_slice()?, ends.as_slice()?)
        .map_err(PyValueError::new_err)?;
    Ok((
        starts.into_pyarray(py),
        ends.into_pyarray(py),
        values.into_pyarray(py),
    ))
}

#[pyfunction]
pub fn waiting_times_flat_numpy<'py>(
    py: Python<'py>,
    timestamps_s: PyReadonlyArray1<'py, f64>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = waiting_times_flat_impl(timestamps_s.as_slice()?, ends.as_slice()?)
        .map_err(PyValueError::new_err)?;
    Ok(values.into_pyarray(py))
}

#[pyfunction]
pub fn waiting_times_indexed_numpy<'py>(
    py: Python<'py>,
    timestamps_s: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyWaitingTimes<'py>> {
    let (starts, ends, values) = waiting_times_indexed_impl(
        timestamps_s.as_slice()?,
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

#[pyfunction]
pub fn waiting_times_indexed_flat_numpy<'py>(
    py: Python<'py>,
    timestamps_s: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let values = waiting_times_indexed_flat_impl(
        timestamps_s.as_slice()?,
        indices.as_slice()?,
        ends.as_slice()?,
        None,
    )
    .map_err(PyValueError::new_err)?;
    Ok(values.into_pyarray(py))
}

#[pyfunction]
pub fn waiting_times_arrow<'py>(
    py: Python<'py>,
    timestamps_s: PyArray,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyWaitingTimesArrow<'py>> {
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;
    let (starts, ends, values) = waiting_times_impl(arrow_values(&timestamps_s), ends.as_slice()?)
        .map_err(PyValueError::new_err)?;
    Ok((
        starts.into_pyarray(py),
        ends.into_pyarray(py),
        f64_results_into_arrow(values),
    ))
}

#[pyfunction]
pub fn waiting_times_flat_arrow<'py>(
    timestamps_s: PyArray,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyArray> {
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;
    let values = waiting_times_flat_impl(arrow_values(&timestamps_s), ends.as_slice()?)
        .map_err(PyValueError::new_err)?;
    Ok(f64_results_into_arrow(values))
}

#[pyfunction]
pub fn waiting_times_indexed_arrow<'py>(
    py: Python<'py>,
    timestamps_s: PyArray,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyWaitingTimesArrow<'py>> {
    let timestamps_s = as_nullable_f64_array(timestamps_s, "timestamps_s")?;
    let valid_rows = arrow_valid_rows(&[&timestamps_s]);
    let (starts, ends, values) = waiting_times_indexed_impl(
        arrow_values(&timestamps_s),
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

#[pyfunction]
pub fn waiting_times_indexed_flat_arrow<'py>(
    timestamps_s: PyArray,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyArray> {
    let timestamps_s = as_nullable_f64_array(timestamps_s, "timestamps_s")?;
    let valid_rows = arrow_valid_rows(&[&timestamps_s]);
    let values = waiting_times_indexed_flat_impl(
        arrow_values(&timestamps_s),
        indices.as_slice()?,
        ends.as_slice()?,
        valid_rows.as_deref(),
    )
    .map_err(PyValueError::new_err)?;
    Ok(f64_results_into_arrow(values))
}
