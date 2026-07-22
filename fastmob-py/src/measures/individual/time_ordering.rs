use arrow_array::{Array, UInt64Array};
use fastmob_core::measures::individual::time_ordering::{
    OrderedIndexRanges, split_ordered_index_ranges, time_ordered_indices_for_u64_codes,
    time_ordered_indices_single_user,
};
use fastmob_core::utils::{split_ranges, validate_uid_len};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{arrow_values, as_nullable_f64_array};

type OrderedNumpyResult<'py> = (Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>);
type OrderedStartEndNumpyResult<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
);
pub fn ordered_index_ranges_into_arrays<'py>(
    py: Python<'py>,
    ordered: OrderedIndexRanges,
) -> OrderedNumpyResult<'py> {
    let (indices, ends) = split_ordered_index_ranges(ordered);
    (indices.into_pyarray(py), ends.into_pyarray(py))
}

pub fn ordered_index_ranges_into_start_end_arrays<'py>(
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

pub fn time_ordered_indices_from_ndarray_uids(
    py: Python<'_>,
    uids: &Bound<'_, PyAny>,
    timestamps: &[f64],
    num_groups: Option<usize>,
) -> PyResult<OrderedIndexRanges> {
    if uids.is_none() {
        return Ok(py.detach(|| time_ordered_indices_single_user(timestamps)));
    }

    if let Ok(array) = uids.extract::<PyReadonlyArray1<u64>>() {
        validate_uid_len(timestamps.len(), array.len()?).map_err(PyValueError::new_err)?;
        let slice = array.as_slice()?;
        let num_groups = num_groups
            .ok_or_else(|| PyValueError::new_err("num_groups is required for uint64 uid codes"))?;
        return py
            .detach(|| time_ordered_indices_for_u64_codes(slice, timestamps, num_groups))
            .map_err(PyValueError::new_err);
    }

    Err(PyValueError::new_err(
        "expected uint64 numpy uid codes for time-ordered indices",
    ))
}

pub fn time_ordered_indices_from_c_array_uids(
    py: Python<'_>,
    uids: &Bound<'_, PyAny>,
    timestamps: &[f64],
    num_groups: Option<usize>,
) -> PyResult<OrderedIndexRanges> {
    if uids.is_none() {
        return Ok(py.detach(|| time_ordered_indices_single_user(timestamps)));
    }

    let uids = uids.extract::<PyArray>()?;
    let (array_ref, _field) = uids.into_inner();
    let array = array_ref.as_any();

    if let Some(array) = array.downcast_ref::<UInt64Array>() {
        if array.null_count() > 0 {
            return Err(PyValueError::new_err(
                "uint64 uid codes for time-ordered indices must not contain nulls",
            ));
        }
        validate_uid_len(timestamps.len(), array.len()).map_err(PyValueError::new_err)?;
        let start = array.offset();
        let end = start + array.len();
        let values = &array.values()[start..end];
        let num_groups = num_groups
            .ok_or_else(|| PyValueError::new_err("num_groups is required for uint64 uid codes"))?;
        return py
            .detach(|| time_ordered_indices_for_u64_codes(values, timestamps, num_groups))
            .map_err(PyValueError::new_err);
    }

    Err(PyValueError::new_err(
        "expected uint64 Arrow uid codes for time-ordered indices",
    ))
}

fn is_arrow_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.hasattr("__arrow_c_array__")
}

#[pyfunction]
#[pyo3(signature = (uids, timestamps, num_groups = None))]
pub fn time_ordered_user_indices<'py>(
    py: Python<'py>,
    uids: &Bound<'py, PyAny>,
    timestamps: &Bound<'py, PyAny>,
    num_groups: Option<usize>,
) -> PyResult<OrderedNumpyResult<'py>> {
    if let Ok(timestamps) = timestamps.extract::<PyReadonlyArray1<f64>>() {
        let ordered_indices =
            time_ordered_indices_from_ndarray_uids(py, uids, timestamps.as_slice()?, num_groups)?;
        return Ok(ordered_index_ranges_into_arrays(py, ordered_indices));
    }

    if is_arrow_array(timestamps)? {
        let timestamps = as_nullable_f64_array(timestamps.extract::<PyArray>()?, "timestamps")?;
        let ordered_indices = time_ordered_indices_from_c_array_uids(
            py,
            uids,
            arrow_values(&timestamps),
            num_groups,
        )?;
        return Ok(ordered_index_ranges_into_arrays(py, ordered_indices));
    }

    Err(PyValueError::new_err(
        "timestamps must be a NumPy array or an Arrow array",
    ))
}
