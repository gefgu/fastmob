use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Float64Array, PrimitiveArray, UInt64Array, types::Float64Type,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

/// Extracts each element of an Arrow `PrimitiveArray` as `Option<T::Native>`, mapping nulls to `None`.
pub fn primitive_option_values<T>(array: &PrimitiveArray<T>) -> Vec<Option<T::Native>>
where
    T: arrow_array::types::ArrowPrimitiveType,
    T::Native: Copy,
{
    (0..array.len())
        .map(|idx| {
            if array.is_null(idx) {
                None
            } else {
                Some(array.value(idx))
            }
        })
        .collect()
}

pub fn f64_results_into_arrow(results: Vec<f64>) -> PyArray {
    let array: ArrayRef = Arc::new(Float64Array::from(results));
    PyArray::from_array_ref(array)
}

pub fn u64_results_into_arrow(results: Vec<u64>) -> PyArray {
    let array: ArrayRef = Arc::new(UInt64Array::from(results));
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
        .ok_or_else(|| {
            PyValueError::new_err(format!("expected float64 Arrow array for {name}"))
        })?;
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
        .ok_or_else(|| {
            PyValueError::new_err(format!("expected float64 Arrow array for {name}"))
        })
}

pub fn arrow_values(array: &PrimitiveArray<Float64Type>) -> &[f64] {
    let start = array.offset();
    let end = start + array.len();
    &array.values()[start..end]
}

pub fn ranges_from_starts_ends(
    starts: &[usize],
    ends: &[usize],
) -> PyResult<Vec<(usize, usize)>> {
    if starts.len() != ends.len() {
        return Err(PyValueError::new_err(
            "range starts and ends must have the same length",
        ));
    }

    starts
        .iter()
        .zip(ends)
        .map(|(&start, &end)| {
            if start > end {
                Err(PyValueError::new_err(
                    "range start must be less than or equal to range end",
                ))
            } else {
                Ok((start, end))
            }
        })
        .collect()
}

#[allow(dead_code)]
pub fn split_ranges(ranges: Vec<(usize, usize)>) -> (Vec<usize>, Vec<usize>) {
    ranges.into_iter().unzip()
}
