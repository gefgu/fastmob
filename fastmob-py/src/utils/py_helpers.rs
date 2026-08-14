use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int32Array, Int64Array, PrimitiveArray,
    UInt8Array, UInt32Array, UInt64Array,
    types::{Float64Type, Int64Type, UInt8Type, UInt32Type, UInt64Type},
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

pub fn extract_arrow_array(values: &Bound<'_, PyAny>, name: &str) -> PyResult<PyArray> {
    if !values.hasattr("__arrow_c_array__")? {
        return Err(PyValueError::new_err(format!(
            "expected Arrow array for {name}"
        )));
    }
    values.extract()
}

pub fn f64_results_into_arrow(results: Vec<f64>) -> PyArray {
    let array: ArrayRef = Arc::new(Float64Array::from(results));
    PyArray::from_array_ref(array)
}

pub fn f64_results_into_arrow_nullable(results: Vec<f64>) -> PyArray {
    let array: ArrayRef = Arc::new(Float64Array::from_iter(
        results
            .into_iter()
            .map(|value| value.is_finite().then_some(value)),
    ));
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

pub fn i32_results_into_arrow(results: Vec<i32>) -> PyArray {
    let array: ArrayRef = Arc::new(Int32Array::from(results));
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

pub fn as_nullable_u64_array(arr: PyArray, name: &str) -> PyResult<UInt64Array> {
    let (array_ref, _field) = arr.into_inner();
    array_ref
        .as_any()
        .downcast_ref::<UInt64Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected uint64 Arrow array for {name}")))
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

/// `fastmob.utils._common._extract_timestamps`'s null-datetime-row sentinel, matched
/// exactly: pandas/NumPy int64 has no null representation, so a null datetime row is
/// filled with this value (i64::MIN, never a real Unix-ms timestamp) rather than
/// carrying a genuine Arrow null.
pub const NULL_TIMESTAMP_SENTINEL_MS: i64 = i64::MIN;

/// Converts `i64` millisecond timestamps to `f64` seconds, mapping the null sentinel to
/// `f64::NAN` so downstream `is_finite()` null-row exclusion (already present in every
/// timed core kernel) keeps working unchanged.
pub fn ms_to_seconds(values: &[i64]) -> Vec<f64> {
    values
        .iter()
        .map(|&t| {
            if t == NULL_TIMESTAMP_SENTINEL_MS {
                f64::NAN
            } else {
                t as f64 / 1000.0
            }
        })
        .collect()
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

pub fn as_u32_array(arr: PyArray, name: &str) -> PyResult<PrimitiveArray<UInt32Type>> {
    let (array_ref, _field) = arr.into_inner();
    let array = array_ref
        .as_any()
        .downcast_ref::<UInt32Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected uint32 Arrow array for {name}")))?;
    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }
    Ok(array)
}

pub fn as_nullable_u32_array(arr: PyArray, name: &str) -> PyResult<UInt32Array> {
    let (array_ref, _field) = arr.into_inner();
    array_ref
        .as_any()
        .downcast_ref::<UInt32Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected uint32 Arrow array for {name}")))
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

pub fn as_i32_array(arr: PyArray, name: &str) -> PyResult<Int32Array> {
    let (array_ref, _field) = arr.into_inner();
    let array = array_ref
        .as_any()
        .downcast_ref::<Int32Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected int32 Arrow array for {name}")))?;
    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }
    Ok(array)
}

pub fn as_nullable_i64_array(arr: PyArray, name: &str) -> PyResult<Int64Array> {
    let (array_ref, _field) = arr.into_inner();
    array_ref
        .as_any()
        .downcast_ref::<Int64Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected int64 Arrow array for {name}")))
}

/// Like [`arrow_valid_rows`], but for the common case of `f64` coordinate arrays
/// paired with one `i64` millisecond-timestamp array (the timed-adapter shape).
pub fn arrow_valid_rows_f64_i64(
    f64_arrays: &[&Float64Array],
    i64_array: &Int64Array,
) -> Option<Vec<bool>> {
    if f64_arrays.iter().all(|a| a.null_count() == 0) && i64_array.null_count() == 0 {
        return None;
    }
    let n = i64_array.len();
    Some(
        (0..n)
            .map(|idx| f64_arrays.iter().all(|a| a.is_valid(idx)) && i64_array.is_valid(idx))
            .collect(),
    )
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

pub fn arrow_u32_values(array: &PrimitiveArray<UInt32Type>) -> &[u32] {
    let start = array.offset();
    let end = start + array.len();
    &array.values()[start..end]
}

pub fn arrow_i32_values(array: &Int32Array) -> &[i32] {
    let start = array.offset();
    let end = start + array.len();
    &array.values()[start..end]
}

#[cfg(target_pointer_width = "64")]
pub fn arrow_usize_values(array: &PrimitiveArray<UInt64Type>) -> &[usize] {
    let values = arrow_u64_values(array);
    // UInt64 and usize have identical layouts on supported 64-bit Python targets.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<usize>(), values.len()) }
}

/// Decode Arrow cumulative group ends at the last possible legacy boundary.
///
/// Python callers keep group metadata in Arrow; older core algorithms still
/// accept half-open ranges internally.
pub fn ranges_from_ends(ends: PyArray, value_len: usize) -> PyResult<Vec<(usize, usize)>> {
    let ends = as_u64_array(ends, "ends")?;
    let mut start = 0;
    let mut ranges = Vec::with_capacity(ends.len());
    for &end in arrow_usize_values(&ends) {
        if end < start || end > value_len {
            return Err(PyValueError::new_err(
                "group ends must be monotonic and within input bounds",
            ));
        }
        ranges.push((start, end));
        start = end;
    }
    if start != value_len {
        return Err(PyValueError::new_err(
            "final group end must equal input length",
        ));
    }
    Ok(ranges)
}

pub trait ArrowUsizeArrayExt {
    fn as_slice(&self) -> PyResult<&[usize]>;
}

#[cfg(target_pointer_width = "64")]
impl ArrowUsizeArrayExt for PyArray {
    fn as_slice(&self) -> PyResult<&[usize]> {
        let array = self
            .array()
            .as_any()
            .downcast_ref::<UInt64Array>()
            .ok_or_else(|| PyValueError::new_err("expected uint64 Arrow index array"))?;
        if array.null_count() > 0 {
            return Err(PyValueError::new_err(
                "Arrow index arrays must not contain nulls",
            ));
        }
        Ok(arrow_usize_values(array))
    }
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
