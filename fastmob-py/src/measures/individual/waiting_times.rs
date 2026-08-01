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
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, f64_results_into_arrow,
    validate_indexed_ends,
};

type PyGroupedF64<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Py<PyAny>,
);
type PyFlatF64 = Py<PyAny>;

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

#[pyfunction]
pub fn waiting_times_presorted<'py>(
    py: Python<'py>,
    timestamps_s: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
) -> PyResult<PyGroupedF64<'py>> {
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;
    let (starts, ends, values) = waiting_times_impl(arrow_values(&timestamps_s), ends.as_slice()?)
        .map_err(PyValueError::new_err)?;
    Ok((
        starts.into_pyarray(py),
        ends.into_pyarray(py),
        arrow_f64_output(py, values)?,
    ))
}

#[pyfunction]
pub fn waiting_times_presorted_flat<'py>(
    py: Python<'py>,
    timestamps_s: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
) -> PyResult<PyFlatF64> {
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;
    let values = waiting_times_flat_impl(arrow_values(&timestamps_s), ends.as_slice()?)
        .map_err(PyValueError::new_err)?;
    arrow_f64_output(py, values)
}

#[pyfunction]
pub fn waiting_times_indexed<'py>(
    py: Python<'py>,
    timestamps_s: ArrowPyArray,
    indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
) -> PyResult<PyGroupedF64<'py>> {
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;
    let timestamps_s = as_nullable_f64_array(timestamps_s, "timestamps_s")?;
    let valid_rows = arrow_valid_rows(&[&timestamps_s]);
    let values = arrow_values(&timestamps_s);
    validate_indexed_ends(values.len(), indices, ends)?;
    let (starts, ends, waits) =
        waiting_times_indexed_impl(values, indices, ends, valid_rows.as_deref())
            .map_err(PyValueError::new_err)?;
    Ok((
        starts.into_pyarray(py),
        ends.into_pyarray(py),
        arrow_f64_output(py, waits)?,
    ))
}

#[pyfunction]
pub fn waiting_times_indexed_flat<'py>(
    py: Python<'py>,
    timestamps_s: ArrowPyArray,
    indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
) -> PyResult<PyFlatF64> {
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;
    let timestamps_s = as_nullable_f64_array(timestamps_s, "timestamps_s")?;
    let valid_rows = arrow_valid_rows(&[&timestamps_s]);
    let values = arrow_values(&timestamps_s);
    validate_indexed_ends(values.len(), indices, ends)?;
    let waits = waiting_times_indexed_flat_impl(values, indices, ends, valid_rows.as_deref())
        .map_err(PyValueError::new_err)?;
    arrow_f64_output(py, waits)
}
