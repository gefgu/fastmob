use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Float64Array, PrimitiveArray, UInt64Array, types::Float64Type};
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
            return Err(PyValueError::new_err("index must be within array bounds"));
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
