use crate::utils::primitive_option_values;
use arrow_array::{
    Array, Int32Array, Int64Array, LargeStringArray, StringArray, UInt32Array, UInt64Array,
    types::{Int32Type, Int64Type, UInt32Type, UInt64Type},
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use skmob2_core::measures::individual::time_ordering::{
    OrderedIndexRanges, split_ordered_index_ranges, time_ordered_indices_for_f64_uid_values,
    time_ordered_indices_for_ord_uid_values, time_ordered_indices_single_user,
};
use skmob2_core::utils::{split_ranges, validate_uid_len};

use crate::utils::{arrow_values, as_f64_array, as_nullable_f64_array};

type OrderedNumpyResult<'py> = (Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>);
type OrderedStartEndNumpyResult<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
);
pub fn ordered_index_ranges_into_numpy<'py>(
    py: Python<'py>,
    ordered: OrderedIndexRanges,
) -> OrderedNumpyResult<'py> {
    let (indices, ends) = split_ordered_index_ranges(ordered);
    (indices.into_pyarray(py), ends.into_pyarray(py))
}

pub fn ordered_index_ranges_into_start_end_numpy<'py>(
    py: Python<'py>,
    (indices, ranges): OrderedIndexRanges,
) -> OrderedStartEndNumpyResult<'py> {
    let (starts, ends) = split_ranges(ranges);
    (
        indices.into_pyarray(py),
        starts.into_pyarray(py),
        ends.into_pyarray(py),
    )
}

pub fn time_ordered_indices_from_numpy_uids(
    py: Python<'_>,
    uids: &Bound<'_, PyAny>,
    timestamps: &[f64],
) -> PyResult<OrderedIndexRanges> {
    if uids.is_none() {
        return Ok(py.detach(|| time_ordered_indices_single_user(timestamps)));
    }

    if let Ok(array) = uids.extract::<PyReadonlyArray1<i64>>() {
        validate_uid_len(timestamps.len(), array.len()?).map_err(PyValueError::new_err)?;
        let slice = array.as_slice()?;
        return Ok(py.detach(|| time_ordered_indices_for_ord_uid_values(slice, timestamps)));
    }
    if let Ok(array) = uids.extract::<PyReadonlyArray1<i32>>() {
        validate_uid_len(timestamps.len(), array.len()?).map_err(PyValueError::new_err)?;
        let slice = array.as_slice()?;
        return Ok(py.detach(|| time_ordered_indices_for_ord_uid_values(slice, timestamps)));
    }
    if let Ok(array) = uids.extract::<PyReadonlyArray1<u64>>() {
        validate_uid_len(timestamps.len(), array.len()?).map_err(PyValueError::new_err)?;
        let slice = array.as_slice()?;
        return Ok(py.detach(|| time_ordered_indices_for_ord_uid_values(slice, timestamps)));
    }
    if let Ok(array) = uids.extract::<PyReadonlyArray1<u32>>() {
        validate_uid_len(timestamps.len(), array.len()?).map_err(PyValueError::new_err)?;
        let slice = array.as_slice()?;
        return Ok(py.detach(|| time_ordered_indices_for_ord_uid_values(slice, timestamps)));
    }
    if let Ok(array) = uids.extract::<PyReadonlyArray1<f64>>() {
        validate_uid_len(timestamps.len(), array.len()?).map_err(PyValueError::new_err)?;
        let slice = array.as_slice()?;
        return Ok(py.detach(|| time_ordered_indices_for_f64_uid_values(slice, timestamps)));
    }

    Err(PyValueError::new_err(
        "unsupported numpy uid dtype for time-ordered indices",
    ))
}

pub fn time_ordered_indices_from_arrow_uids(
    py: Python<'_>,
    uids: &Bound<'_, PyAny>,
    timestamps: &[f64],
) -> PyResult<OrderedIndexRanges> {
    if uids.is_none() {
        return Ok(py.detach(|| time_ordered_indices_single_user(timestamps)));
    }

    let uids = uids.extract::<PyArray>()?;
    let (array_ref, _field) = uids.into_inner();
    let array = array_ref.as_any();

    if let Some(array) = array.downcast_ref::<Int64Array>() {
        validate_uid_len(timestamps.len(), array.len()).map_err(PyValueError::new_err)?;
        let values = primitive_option_values::<Int64Type>(array);
        return Ok(py.detach(|| time_ordered_indices_for_ord_uid_values(&values, timestamps)));
    }
    if let Some(array) = array.downcast_ref::<Int32Array>() {
        validate_uid_len(timestamps.len(), array.len()).map_err(PyValueError::new_err)?;
        let values = primitive_option_values::<Int32Type>(array);
        return Ok(py.detach(|| time_ordered_indices_for_ord_uid_values(&values, timestamps)));
    }
    if let Some(array) = array.downcast_ref::<UInt64Array>() {
        validate_uid_len(timestamps.len(), array.len()).map_err(PyValueError::new_err)?;
        let values = primitive_option_values::<UInt64Type>(array);
        return Ok(py.detach(|| time_ordered_indices_for_ord_uid_values(&values, timestamps)));
    }
    if let Some(array) = array.downcast_ref::<UInt32Array>() {
        validate_uid_len(timestamps.len(), array.len()).map_err(PyValueError::new_err)?;
        let values = primitive_option_values::<UInt32Type>(array);
        return Ok(py.detach(|| time_ordered_indices_for_ord_uid_values(&values, timestamps)));
    }
    if let Some(array) = array.downcast_ref::<StringArray>() {
        validate_uid_len(timestamps.len(), array.len()).map_err(PyValueError::new_err)?;
        let values: Vec<Option<&str>> = (0..array.len())
            .map(|idx| {
                if array.is_null(idx) {
                    None
                } else {
                    Some(array.value(idx))
                }
            })
            .collect();
        return Ok(py.detach(|| time_ordered_indices_for_ord_uid_values(&values, timestamps)));
    }
    if let Some(array) = array.downcast_ref::<LargeStringArray>() {
        validate_uid_len(timestamps.len(), array.len()).map_err(PyValueError::new_err)?;
        let values: Vec<Option<&str>> = (0..array.len())
            .map(|idx| {
                if array.is_null(idx) {
                    None
                } else {
                    Some(array.value(idx))
                }
            })
            .collect();
        return Ok(py.detach(|| time_ordered_indices_for_ord_uid_values(&values, timestamps)));
    }

    Err(PyValueError::new_err(
        "unsupported Arrow uid type for time-ordered indices",
    ))
}

#[pyfunction]
pub fn time_ordered_user_indices_numpy<'py>(
    py: Python<'py>,
    uids: &Bound<'py, PyAny>,
    timestamps: PyReadonlyArray1<'py, f64>,
) -> PyResult<OrderedNumpyResult<'py>> {
    let ordered_indices = time_ordered_indices_from_numpy_uids(py, uids, timestamps.as_slice()?)?;
    Ok(ordered_index_ranges_into_numpy(py, ordered_indices))
}

#[pyfunction]
pub fn time_ordered_user_indices_arrow<'py>(
    py: Python<'py>,
    uids: &Bound<'py, PyAny>,
    timestamps: PyArray,
) -> PyResult<OrderedNumpyResult<'py>> {
    let timestamps = as_nullable_f64_array(timestamps, "timestamps")?;
    let timestamp_values = arrow_values(&timestamps);
    let time_ordered_indices = time_ordered_indices_from_arrow_uids(py, uids, timestamp_values)?;
    Ok(ordered_index_ranges_into_numpy(py, time_ordered_indices))
}
