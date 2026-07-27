use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, PrimitiveArray, UInt8Array,
    UInt32Array, UInt64Array,
    types::{Float64Type, Int64Type, UInt8Type, UInt64Type},
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

pub fn f64_results_into_arrow(results: Vec<f64>) -> PyArray {
    let array: ArrayRef = Arc::new(Float64Array::from(results));
    PyArray::from_array_ref(array)
}

pub fn u64_results_into_arrow(results: Vec<u64>) -> PyArray {
    let array: ArrayRef = Arc::new(UInt64Array::from(results));
    PyArray::from_array_ref(array)
}

pub fn i64_results_into_arrow(results: Vec<i64>) -> PyArray {
    let array: ArrayRef = Arc::new(Int64Array::from(results));
    PyArray::from_array_ref(array)
}

/// Mirrors [`u64_results_into_arrow`] for `u32` outputs.
///
/// @usedBy `fastmob-py/src/preprocessing/segment_traj_py.rs::segment_trajectory_arrow`
/// (segment ids restart at `0` per user, so `u32` is comfortably wide enough).
pub fn u32_results_into_arrow(results: Vec<u32>) -> PyArray {
    let array: ArrayRef = Arc::new(UInt32Array::from(results));
    PyArray::from_array_ref(array)
}

/// Same as [`u64_results_into_arrow`], but maps `sentinel` values to a real
/// Arrow null slot instead of a valid-looking sentinel integer -- for kernels
/// (e.g. H3 cell conversion) that use a reserved value to mean "invalid".
pub fn u64_results_into_arrow_nullable(results: Vec<u64>, sentinel: u64) -> PyArray {
    let array: ArrayRef = Arc::new(UInt64Array::from_iter(
        results
            .into_iter()
            .map(|v| if v == sentinel { None } else { Some(v) }),
    ));
    PyArray::from_array_ref(array)
}

pub fn bool_results_into_arrow(results: Vec<bool>) -> PyArray {
    let array: ArrayRef = Arc::new(BooleanArray::from(results));
    PyArray::from_array_ref(array)
}

pub fn as_f64_array(arr: PyArray, name: &str) -> PyResult<PrimitiveArray<Float64Type>> {
    let (array_ref, _field) = arr.into_inner();
    let array = array_ref
        .as_any()
        .downcast_ref::<Float64Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected float64 Arrow array for {name}")))?;
    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }
    Ok(array)
}

pub fn as_nullable_f64_array(arr: PyArray, name: &str) -> PyResult<Float64Array> {
    let (array_ref, _field) = arr.into_inner();
    array_ref
        .as_any()
        .downcast_ref::<Float64Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected float64 Arrow array for {name}")))
}

pub fn arrow_valid_rows(arrays: &[&Float64Array]) -> Option<Vec<bool>> {
    if arrays.iter().all(|a| a.null_count() == 0) {
        return None;
    }
    let n = arrays[0].len();
    Some(
        (0..n)
            .map(|idx| arrays.iter().all(|a| a.is_valid(idx)))
            .collect(),
    )
}

pub fn arrow_values(array: &PrimitiveArray<Float64Type>) -> &[f64] {
    let start = array.offset();
    let end = start + array.len();
    &array.values()[start..end]
}

pub fn as_u64_array(arr: PyArray, name: &str) -> PyResult<PrimitiveArray<UInt64Type>> {
    let (array_ref, _field) = arr.into_inner();
    let array = array_ref
        .as_any()
        .downcast_ref::<UInt64Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected uint64 Arrow array for {name}")))?;
    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }
    Ok(array)
}

pub fn as_i64_array(arr: PyArray, name: &str) -> PyResult<PrimitiveArray<Int64Type>> {
    let (array_ref, _field) = arr.into_inner();
    let array = array_ref
        .as_any()
        .downcast_ref::<Int64Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected int64 Arrow array for {name}")))?;
    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }
    Ok(array)
}

pub fn as_u8_array(arr: PyArray, name: &str) -> PyResult<PrimitiveArray<UInt8Type>> {
    let (array_ref, _field) = arr.into_inner();
    let array = array_ref
        .as_any()
        .downcast_ref::<UInt8Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected uint8 Arrow array for {name}")))?;
    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }
    Ok(array)
}

pub fn as_bool_array(arr: PyArray, name: &str) -> PyResult<BooleanArray> {
    let (array_ref, _field) = arr.into_inner();
    let array = array_ref
        .as_any()
        .downcast_ref::<BooleanArray>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected bool Arrow array for {name}")))?;
    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }
    Ok(array)
}

pub fn arrow_u64_values(array: &PrimitiveArray<UInt64Type>) -> &[u64] {
    let start = array.offset();
    let end = start + array.len();
    &array.values()[start..end]
}

pub fn arrow_i64_values(array: &PrimitiveArray<Int64Type>) -> &[i64] {
    let start = array.offset();
    let end = start + array.len();
    &array.values()[start..end]
}

pub fn arrow_u8_values(array: &PrimitiveArray<UInt8Type>) -> &[u8] {
    let start = array.offset();
    let end = start + array.len();
    &array.values()[start..end]
}

pub fn arrow_bool_values(array: &BooleanArray) -> Vec<bool> {
    (0..array.len()).map(|i| array.value(i)).collect()
}

pub fn validate_indexed_ends(value_len: usize, indices: &[usize], ends: &[usize]) -> PyResult<()> {
    let mut previous = 0usize;
    for &end in ends {
        if end < previous {
            return Err(PyValueError::new_err(
                "range ends must be monotonically non-decreasing",
            ));
        }
        if end > indices.len() {
            return Err(PyValueError::new_err(
                "range end must be within index array bounds",
            ));
        }
        previous = end;
    }

    for &idx in indices {
        if idx >= value_len {
            return Err(PyValueError::new_err(
                "index must be within coordinate array bounds",
            ));
        }
    }

    Ok(())
}

#[allow(dead_code)]
pub fn split_ranges(ranges: Vec<(usize, usize)>) -> (Vec<usize>, Vec<usize>) {
    ranges.into_iter().unzip()
}
