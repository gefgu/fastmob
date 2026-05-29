use std::sync::Arc;

use arrow_array::{types::Float64Type, Array, ArrayRef, Float64Array, PrimitiveArray, UInt64Array};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

pub(crate) fn validate_coord_ranges(
    latitudes: &[f64],
    longitudes: &[f64],
    ranges: &[(usize, usize)],
) -> PyResult<()> {
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }

    let n = latitudes.len();
    for &(start, end) in ranges {
        if start > end {
            return Err(PyValueError::new_err(
                "range start must be less than or equal to range end",
            ));
        }
        if end > n {
            return Err(PyValueError::new_err(
                "range end must be within coordinate array bounds",
            ));
        }
    }

    Ok(())
}

pub(crate) fn validate_ranges(n: usize, ranges: &[(usize, usize)]) -> PyResult<()> {
    for &(start, end) in ranges {
        if start > end {
            return Err(PyValueError::new_err(
                "range start must be less than or equal to range end",
            ));
        }
        if end > n {
            return Err(PyValueError::new_err(
                "range end must be within array bounds",
            ));
        }
    }
    Ok(())
}

pub(crate) fn validate_indexed_ranges(
    value_len: usize,
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> PyResult<()> {
    let n_indices = indices.len();
    for &(start, end) in ranges {
        if start > end {
            return Err(PyValueError::new_err(
                "range start must be less than or equal to range end",
            ));
        }
        if end > n_indices {
            return Err(PyValueError::new_err(
                "range end must be within index array bounds",
            ));
        }
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

pub(crate) fn validate_indexed_coord_ranges(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> PyResult<()> {
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }
    validate_indexed_ranges(latitudes.len(), indices, ranges)
}

pub(crate) fn ranges_from_starts_ends(
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

pub(crate) fn f64_results_into_arrow(results: Vec<f64>) -> PyArray {
    let array: ArrayRef = Arc::new(Float64Array::from(results));
    PyArray::from_array_ref(array)
}

pub(crate) fn u64_results_into_arrow(results: Vec<u64>) -> PyArray {
    let array: ArrayRef = Arc::new(UInt64Array::from(results));
    PyArray::from_array_ref(array)
}

pub(crate) fn as_f64_array(arr: PyArray, name: &str) -> PyResult<PrimitiveArray<Float64Type>> {
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

pub(crate) fn arrow_values(array: &PrimitiveArray<Float64Type>) -> &[f64] {
    let start = array.offset();
    let end = start + array.len();
    &array.values()[start..end]
}

/// Splits a `Vec<(usize, usize)>` of ranges into two parallel `Vec<usize>` of starts and ends.
///
/// Returns `(starts, ends)` where each element corresponds to one range.
pub(crate) fn split_ranges(ranges: Vec<(usize, usize)>) -> (Vec<usize>, Vec<usize>) {
    ranges.into_iter().unzip()
}

/// Extracts each element of an Arrow `PrimitiveArray` as `Option<T::Native>`, mapping nulls to
/// `None`.
pub(crate) fn primitive_option_values<T>(array: &PrimitiveArray<T>) -> Vec<Option<T::Native>>
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

/// Builds contiguous `(start, end)` ranges from a slice of values that has been sorted by an
/// index permutation, grouping consecutive equal values.
///
/// `values` is the original unsorted array; `indices` is a permutation such that
/// `values[indices[i]]` is non-decreasing (ties broken by index). Each run of equal
/// `values[indices[i]]` becomes one range `(start, end)` in the output.
///
/// Returns an empty `Vec` when `indices` is empty.
pub(crate) fn ranges_from_sorted_values<T: PartialEq>(
    values: &[T],
    indices: &[usize],
) -> Vec<(usize, usize)> {
    if indices.is_empty() {
        return Vec::new();
    }

    let mut ranges = Vec::new();
    let mut start = 0usize;
    for pos in 1..indices.len() {
        if values[indices[pos]] != values[indices[pos - 1]] {
            ranges.push((start, pos));
            start = pos;
        }
    }
    ranges.push((start, indices.len()));
    ranges
}

/// Downcasts an Arrow `PyArray` to a `Float64Array`, allowing nulls.
///
/// Returns a `PyValueError` when the array is not float64.
pub(crate) fn as_nullable_f64_array(arr: PyArray, name: &str) -> PyResult<Float64Array> {
    let (array_ref, _field) = arr.into_inner();
    array_ref
        .as_any()
        .downcast_ref::<Float64Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected float64 Arrow array for {name}")))
}

/// Computes the median of a mutable slice in-place using quickselect (O(n) average).
///
/// Mutates the slice order as a side effect — callers that need the original order
/// must clone before calling. Returns `f64::NAN` for empty slices.
pub(crate) fn median_slice_in_place(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let mid = n / 2;
    v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    if n.is_multiple_of(2) {
        // Left partition v[..mid] is unordered but all ≤ v[mid]; scan for max.
        let left_max = v[..mid].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (left_max + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

/// Validates that a uid array has the same length as the coordinate array.
///
/// Returns a `PyValueError` with a message listing all four affected columns when lengths differ.
pub(crate) fn validate_uid_len(n: usize, uid_len: usize) -> PyResult<()> {
    if n != uid_len {
        return Err(PyValueError::new_err(
            "uids, latitudes, longitudes, and timestamps must have the same length",
        ));
    }
    Ok(())
}
