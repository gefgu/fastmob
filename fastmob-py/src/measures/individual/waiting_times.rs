use fastmob_core::measures::individual::waiting_times::{
    waiting_times_flat_impl, waiting_times_impl, waiting_times_indexed_flat_impl,
    waiting_times_indexed_impl,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
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

fn numpy_f64_output<'py>(py: Python<'py>, values: Vec<f64>) -> Py<PyAny> {
    values.into_pyarray(py).into_any().unbind()
}

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

fn is_arrow_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.hasattr("__arrow_c_array__")
}

#[pyfunction]
pub fn waiting_times_presorted<'py>(
    py: Python<'py>,
    timestamps_s: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyGroupedF64<'py>> {
    if let Ok(timestamps_s) = timestamps_s.extract::<PyReadonlyArray1<f64>>() {
        let (starts, ends, values) = waiting_times_impl(timestamps_s.as_slice()?, ends.as_slice()?)
            .map_err(PyValueError::new_err)?;
        return Ok((
            starts.into_pyarray(py),
            ends.into_pyarray(py),
            numpy_f64_output(py, values),
        ));
    }

    if is_arrow_array(timestamps_s)? {
        let timestamps_s = as_f64_array(timestamps_s.extract::<ArrowPyArray>()?, "timestamps_s")?;
        let (starts, ends, values) =
            waiting_times_impl(arrow_values(&timestamps_s), ends.as_slice()?)
                .map_err(PyValueError::new_err)?;
        return Ok((
            starts.into_pyarray(py),
            ends.into_pyarray(py),
            arrow_f64_output(py, values)?,
        ));
    }

    Err(PyTypeError::new_err(
        "timestamps_s must be a NumPy array or an Arrow array",
    ))
}

#[pyfunction]
pub fn waiting_times_presorted_flat<'py>(
    py: Python<'py>,
    timestamps_s: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyFlatF64> {
    if let Ok(timestamps_s) = timestamps_s.extract::<PyReadonlyArray1<f64>>() {
        let values = waiting_times_flat_impl(timestamps_s.as_slice()?, ends.as_slice()?)
            .map_err(PyValueError::new_err)?;
        return Ok(numpy_f64_output(py, values));
    }

    if is_arrow_array(timestamps_s)? {
        let timestamps_s = as_f64_array(timestamps_s.extract::<ArrowPyArray>()?, "timestamps_s")?;
        let values = waiting_times_flat_impl(arrow_values(&timestamps_s), ends.as_slice()?)
            .map_err(PyValueError::new_err)?;
        return arrow_f64_output(py, values);
    }

    Err(PyTypeError::new_err(
        "timestamps_s must be a NumPy array or an Arrow array",
    ))
}

#[pyfunction]
pub fn waiting_times_indexed<'py>(
    py: Python<'py>,
    timestamps_s: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyGroupedF64<'py>> {
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;
    if let Ok(timestamps_s) = timestamps_s.extract::<PyReadonlyArray1<f64>>() {
        let values = timestamps_s.as_slice()?;
        validate_indexed_ends(values.len(), indices, ends)?;
        let (starts, ends, waits) = waiting_times_indexed_impl(values, indices, ends, None)
            .map_err(PyValueError::new_err)?;
        return Ok((
            starts.into_pyarray(py),
            ends.into_pyarray(py),
            numpy_f64_output(py, waits),
        ));
    }

    if is_arrow_array(timestamps_s)? {
        let timestamps_s =
            as_nullable_f64_array(timestamps_s.extract::<ArrowPyArray>()?, "timestamps_s")?;
        let valid_rows = arrow_valid_rows(&[&timestamps_s]);
        let values = arrow_values(&timestamps_s);
        validate_indexed_ends(values.len(), indices, ends)?;
        let (starts, ends, waits) =
            waiting_times_indexed_impl(values, indices, ends, valid_rows.as_deref())
                .map_err(PyValueError::new_err)?;
        return Ok((
            starts.into_pyarray(py),
            ends.into_pyarray(py),
            arrow_f64_output(py, waits)?,
        ));
    }

    Err(PyTypeError::new_err(
        "timestamps_s must be a NumPy array or an Arrow array",
    ))
}

#[pyfunction]
pub fn waiting_times_indexed_flat<'py>(
    py: Python<'py>,
    timestamps_s: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyFlatF64> {
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;
    if let Ok(timestamps_s) = timestamps_s.extract::<PyReadonlyArray1<f64>>() {
        let values = timestamps_s.as_slice()?;
        validate_indexed_ends(values.len(), indices, ends)?;
        let waits = waiting_times_indexed_flat_impl(values, indices, ends, None)
            .map_err(PyValueError::new_err)?;
        return Ok(numpy_f64_output(py, waits));
    }

    if is_arrow_array(timestamps_s)? {
        let timestamps_s =
            as_nullable_f64_array(timestamps_s.extract::<ArrowPyArray>()?, "timestamps_s")?;
        let valid_rows = arrow_valid_rows(&[&timestamps_s]);
        let values = arrow_values(&timestamps_s);
        validate_indexed_ends(values.len(), indices, ends)?;
        let waits = waiting_times_indexed_flat_impl(values, indices, ends, valid_rows.as_deref())
            .map_err(PyValueError::new_err)?;
        return arrow_f64_output(py, waits);
    }

    Err(PyTypeError::new_err(
        "timestamps_s must be a NumPy array or an Arrow array",
    ))
}
