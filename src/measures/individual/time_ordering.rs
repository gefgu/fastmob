use std::time::Instant;

use arrow_array::{
    Array, Int32Array, Int64Array, LargeStringArray, StringArray, UInt32Array, UInt64Array,
    types::{Int32Type, Int64Type, UInt32Type, UInt64Type},
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use rayon::prelude::*;

use crate::utils::{
    arrow_values, as_f64_array, primitive_option_values, split_ranges, validate_uid_len,
};

pub(crate) type IndexRanges = Vec<(usize, usize)>;
pub(crate) type OrderedIndexRanges = (Vec<usize>, IndexRanges);
pub(crate) type PyOrderedIndexRanges<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
);

pub(crate) fn ordered_index_ranges_into_numpy<'py>(
    py: Python<'py>,
    (indices, ranges): OrderedIndexRanges,
) -> PyOrderedIndexRanges<'py> {
    let (starts, ends) = split_ranges(ranges);
    (
        indices.into_pyarray(py),
        starts.into_pyarray(py),
        ends.into_pyarray(py),
    )
}

fn time_ordered_indices_single_user(timestamps: &[f64]) -> OrderedIndexRanges {
    let mut indices: Vec<usize> = (0..timestamps.len()).collect();
    indices.par_sort_by(|&left, &right| {
        timestamps[left]
            .total_cmp(&timestamps[right])
            .then(left.cmp(&right))
    });
    let n = indices.len();
    (indices, vec![(0, n)])
}

fn time_ordered_indices_for_ord_uid_values<T: Ord + Copy + Sync + Send>(
    uids: &[T],
    timestamps: &[f64],
) -> OrderedIndexRanges {
    let n = uids.len();
    let mut grouped: Vec<(T, usize)> = uids
        .par_iter()
        .enumerate()
        .map(|(idx, &uid)| (uid, idx))
        .collect();

    grouped.par_sort_unstable_by(|left, right| left.0.cmp(&right.0));

    let mut ranges = Vec::new();
    if !grouped.is_empty() {
        let mut start = 0;
        for idx in 1..n {
            if grouped[idx].0 != grouped[idx - 1].0 {
                ranges.push((start, idx));
                start = idx;
            }
        }
        ranges.push((start, n));
    }

    let mut indices = vec![0usize; n];
    let indices_ptr = indices.as_mut_ptr() as usize;

    ranges.par_iter().for_each(|&(start, end)| {
        // SAFETY: `ranges` is built from monotonically increasing group boundaries, so each
        // parallel task writes to a unique, non-overlapping slice of `indices`.
        let out_slice = unsafe {
            std::slice::from_raw_parts_mut((indices_ptr as *mut usize).add(start), end - start)
        };

        for offset in 0..(end - start) {
            out_slice[offset] = grouped[start + offset].1;
        }

        out_slice.sort_unstable_by(|&left, &right| {
            timestamps[left]
                .total_cmp(&timestamps[right])
                .then(left.cmp(&right))
        });
    });

    (indices, ranges)
}

fn time_ordered_indices_for_f64_uid_values(uids: &[f64], timestamps: &[f64]) -> OrderedIndexRanges {
    let n = uids.len();
    let mut grouped: Vec<(f64, usize)> = uids
        .par_iter()
        .enumerate()
        .map(|(idx, &uid)| (uid, idx))
        .collect();

    grouped.par_sort_unstable_by(|left, right| left.0.total_cmp(&right.0));

    let mut ranges = Vec::new();
    if !grouped.is_empty() {
        let mut start = 0;
        for idx in 1..n {
            if grouped[idx].0.total_cmp(&grouped[idx - 1].0) != std::cmp::Ordering::Equal {
                ranges.push((start, idx));
                start = idx;
            }
        }
        ranges.push((start, n));
    }

    let mut indices = vec![0usize; n];
    let indices_ptr = indices.as_mut_ptr() as usize;

    ranges.par_iter().for_each(|&(start, end)| {
        // SAFETY: `ranges` is built from monotonically increasing group boundaries, so each
        // parallel task writes to a unique, non-overlapping slice of `indices`.
        let out_slice = unsafe {
            std::slice::from_raw_parts_mut((indices_ptr as *mut usize).add(start), end - start)
        };

        for offset in 0..(end - start) {
            out_slice[offset] = grouped[start + offset].1;
        }

        out_slice.sort_unstable_by(|&left, &right| {
            timestamps[left]
                .total_cmp(&timestamps[right])
                .then(left.cmp(&right))
        });
    });

    (indices, ranges)
}

pub(crate) fn time_ordered_indices_from_numpy_uids(
    py: Python<'_>,
    uids: &Bound<'_, PyAny>,
    timestamps: &[f64],
) -> PyResult<OrderedIndexRanges> {
    if uids.is_none() {
        return Ok(py.detach(|| time_ordered_indices_single_user(timestamps)));
    }

    if let Ok(array) = uids.extract::<PyReadonlyArray1<i64>>() {
        validate_uid_len(timestamps.len(), array.len()?)?;
        let slice = array.as_slice()?;
        return Ok(py.detach(|| time_ordered_indices_for_ord_uid_values(slice, timestamps)));
    }
    if let Ok(array) = uids.extract::<PyReadonlyArray1<i32>>() {
        validate_uid_len(timestamps.len(), array.len()?)?;
        let slice = array.as_slice()?;
        return Ok(py.detach(|| time_ordered_indices_for_ord_uid_values(slice, timestamps)));
    }
    if let Ok(array) = uids.extract::<PyReadonlyArray1<u64>>() {
        validate_uid_len(timestamps.len(), array.len()?)?;
        let slice = array.as_slice()?;
        return Ok(py.detach(|| time_ordered_indices_for_ord_uid_values(slice, timestamps)));
    }
    if let Ok(array) = uids.extract::<PyReadonlyArray1<u32>>() {
        validate_uid_len(timestamps.len(), array.len()?)?;
        let slice = array.as_slice()?;
        return Ok(py.detach(|| time_ordered_indices_for_ord_uid_values(slice, timestamps)));
    }
    if let Ok(array) = uids.extract::<PyReadonlyArray1<f64>>() {
        validate_uid_len(timestamps.len(), array.len()?)?;
        let slice = array.as_slice()?;
        return Ok(py.detach(|| time_ordered_indices_for_f64_uid_values(slice, timestamps)));
    }

    Err(PyValueError::new_err(
        "unsupported numpy uid dtype for time-ordered indices",
    ))
}

pub(crate) fn time_ordered_indices_from_arrow_uids(
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
        validate_uid_len(timestamps.len(), array.len())?;
        let values = primitive_option_values::<Int64Type>(array);
        return Ok(py.detach(|| time_ordered_indices_for_ord_uid_values(&values, timestamps)));
    }
    if let Some(array) = array.downcast_ref::<Int32Array>() {
        validate_uid_len(timestamps.len(), array.len())?;
        let values = primitive_option_values::<Int32Type>(array);
        return Ok(py.detach(|| time_ordered_indices_for_ord_uid_values(&values, timestamps)));
    }
    if let Some(array) = array.downcast_ref::<UInt64Array>() {
        validate_uid_len(timestamps.len(), array.len())?;
        let values = primitive_option_values::<UInt64Type>(array);
        return Ok(py.detach(|| time_ordered_indices_for_ord_uid_values(&values, timestamps)));
    }
    if let Some(array) = array.downcast_ref::<UInt32Array>() {
        validate_uid_len(timestamps.len(), array.len())?;
        let values = primitive_option_values::<UInt32Type>(array);
        return Ok(py.detach(|| time_ordered_indices_for_ord_uid_values(&values, timestamps)));
    }
    if let Some(array) = array.downcast_ref::<StringArray>() {
        validate_uid_len(timestamps.len(), array.len())?;
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
        validate_uid_len(timestamps.len(), array.len())?;
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
pub(crate) fn time_ordered_user_indices_numpy<'py>(
    py: Python<'py>,
    uids: &Bound<'py, PyAny>,
    timestamps: PyReadonlyArray1<'py, f64>,
) -> PyResult<PyOrderedIndexRanges<'py>> {
    let start = Instant::now();

    let ordered_indices = time_ordered_indices_from_numpy_uids(py, uids, timestamps.as_slice()?)?;
    let final_result = ordered_index_ranges_into_numpy(py, ordered_indices);

    println!("Time ordering time: {:.2?}", start.elapsed());

    Ok(final_result)
}

#[pyfunction]
pub(crate) fn time_ordered_user_indices_arrow<'py>(
    py: Python<'py>,
    uids: &Bound<'py, PyAny>,
    timestamps: PyArray,
) -> PyResult<PyOrderedIndexRanges<'py>> {
    let start = Instant::now();
    let timestamps = as_f64_array(timestamps, "timestamps")?;
    let timestamp_values = arrow_values(&timestamps);

    let time_ordered_indices = time_ordered_indices_from_arrow_uids(py, uids, timestamp_values)?;
    let final_result = ordered_index_ranges_into_numpy(py, time_ordered_indices);

    println!("Time ordering time: {:.2?}", start.elapsed());

    Ok(final_result)
}
