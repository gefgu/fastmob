use crate::utils::ArrowUsizeArrayExt;
use fastmob_core::measures::individual::waiting_times::{
    waiting_times_flat_impl, waiting_times_impl, waiting_times_indexed_flat_impl,
    waiting_times_indexed_impl,
};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_i64_values, arrow_valid_rows_f64_i64, as_i64_array, as_nullable_i64_array,
    f64_results_into_arrow, ms_to_seconds, validate_indexed_ends,
};

/// Converts `i64` millisecond timestamps to `f64` seconds once, here at the FFI
/// boundary, so the core kernels keep operating on seconds unchanged.
fn timestamps_ms_to_seconds(timestamps_ms: &arrow_array::Int64Array) -> Vec<f64> {
    ms_to_seconds(arrow_i64_values(timestamps_ms))
}

type PyGroupedF64<'py> = (
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Py<PyAny>,
);
type PyFlatF64 = Py<PyAny>;

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

fn u64_offsets(values: Vec<usize>) -> Vec<u64> {
    values.into_iter().map(|value| value as u64).collect()
}

#[pyfunction]
pub fn waiting_times_presorted<'py>(
    py: Python<'py>,
    timestamps_ms: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
) -> PyResult<PyGroupedF64<'py>> {
    let timestamps_ms = as_i64_array(timestamps_ms, "timestamps_ms")?;
    let timestamps_s = timestamps_ms_to_seconds(&timestamps_ms);
    let (starts, ends, values) =
        waiting_times_impl(&timestamps_s, &ends.as_slice()?).map_err(PyValueError::new_err)?;
    Ok((
        u64_offsets(starts).into_pyarray(py),
        u64_offsets(ends).into_pyarray(py),
        arrow_f64_output(py, values)?,
    ))
}

#[pyfunction]
pub fn waiting_times_presorted_flat<'py>(
    py: Python<'py>,
    timestamps_ms: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
) -> PyResult<PyFlatF64> {
    let timestamps_ms = as_i64_array(timestamps_ms, "timestamps_ms")?;
    let timestamps_s = timestamps_ms_to_seconds(&timestamps_ms);
    let values =
        waiting_times_flat_impl(&timestamps_s, &ends.as_slice()?).map_err(PyValueError::new_err)?;
    arrow_f64_output(py, values)
}

#[pyfunction]
pub fn waiting_times_indexed<'py>(
    py: Python<'py>,
    timestamps_ms: ArrowPyArray,
    indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
) -> PyResult<PyGroupedF64<'py>> {
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;
    let timestamps_ms = as_nullable_i64_array(timestamps_ms, "timestamps_ms")?;
    let valid_rows = arrow_valid_rows_f64_i64(&[], &timestamps_ms);
    let timestamps_s = timestamps_ms_to_seconds(&timestamps_ms);
    validate_indexed_ends(timestamps_s.len(), &indices, &ends)?;
    let (starts, ends, waits) =
        waiting_times_indexed_impl(&timestamps_s, &indices, &ends, valid_rows.as_deref())
            .map_err(PyValueError::new_err)?;
    Ok((
        u64_offsets(starts).into_pyarray(py),
        u64_offsets(ends).into_pyarray(py),
        arrow_f64_output(py, waits)?,
    ))
}

#[pyfunction]
pub fn waiting_times_indexed_flat<'py>(
    py: Python<'py>,
    timestamps_ms: ArrowPyArray,
    indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
) -> PyResult<PyFlatF64> {
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;
    let timestamps_ms = as_nullable_i64_array(timestamps_ms, "timestamps_ms")?;
    let valid_rows = arrow_valid_rows_f64_i64(&[], &timestamps_ms);
    let timestamps_s = timestamps_ms_to_seconds(&timestamps_ms);
    validate_indexed_ends(timestamps_s.len(), &indices, &ends)?;
    let waits =
        waiting_times_indexed_flat_impl(&timestamps_s, &indices, &ends, valid_rows.as_deref())
            .map_err(PyValueError::new_err)?;
    arrow_f64_output(py, waits)
}
