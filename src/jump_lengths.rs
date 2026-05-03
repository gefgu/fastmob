use arrow_array::{
    Array, Int32Array, Int64Array, LargeStringArray, PrimitiveArray, StringArray, UInt32Array,
    UInt64Array,
    types::{Int32Type, Int64Type, UInt32Type, UInt64Type},
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use rayon::prelude::*;

use crate::haversine::haversine_km;
use crate::utils::{arrow_values, as_f64_array, validate_coord_ranges};

type IndexRanges = Vec<(usize, usize)>;
type OrderedIndexRanges = (Vec<usize>, IndexRanges);
type PyOrderedIndexRanges<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
);
type PyTimeOrderedJumpLengths<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Vec<Vec<f64>>,
);
type PyTimeOrderedFlatJumpLengths<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Vec<f64>,
);

fn jump_lengths_for_range(
    latitudes: &[f64],
    longitudes: &[f64],
    start: usize,
    end: usize,
) -> Vec<f64> {
    if end - start < 2 {
        return Vec::new();
    }

    (start + 1..end)
        .map(|idx| {
            haversine_km(
                latitudes[idx - 1],
                longitudes[idx - 1],
                latitudes[idx],
                longitudes[idx],
            )
        })
        .collect()
}

fn jump_lengths_for_indexed_range(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    start: usize,
    end: usize,
) -> Vec<f64> {
    if end - start < 2 {
        return Vec::new();
    }

    (start + 1..end)
        .map(|pos| {
            let previous_idx = indices[pos - 1];
            let current_idx = indices[pos];
            haversine_km(
                latitudes[previous_idx],
                longitudes[previous_idx],
                latitudes[current_idx],
                longitudes[current_idx],
            )
        })
        .collect()
}

fn validate_indexed_inputs(
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

    let n_coords = latitudes.len();
    for &idx in indices {
        if idx >= n_coords {
            return Err(PyValueError::new_err(
                "index must be within coordinate array bounds",
            ));
        }
    }

    Ok(())
}

fn validate_time_ordered_inputs(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps: &[f64],
) -> PyResult<()> {
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }
    if latitudes.len() != timestamps.len() {
        return Err(PyValueError::new_err(
            "latitudes, longitudes, and timestamps must have the same length",
        ));
    }
    Ok(())
}

fn validate_uid_len(n: usize, uid_len: usize) -> PyResult<()> {
    if n != uid_len {
        return Err(PyValueError::new_err(
            "uids, latitudes, longitudes, and timestamps must have the same length",
        ));
    }
    Ok(())
}

fn ranges_from_sorted_uid_indices<T: PartialEq>(
    uids: &[T],
    indices: &[usize],
) -> Vec<(usize, usize)> {
    if indices.is_empty() {
        return Vec::new();
    }

    let mut ranges = Vec::new();
    let mut start = 0usize;
    for pos in 1..indices.len() {
        if uids[indices[pos]] != uids[indices[pos - 1]] {
            ranges.push((start, pos));
            start = pos;
        }
    }
    ranges.push((start, indices.len()));
    ranges
}

fn split_ranges(ranges: IndexRanges) -> (Vec<usize>, Vec<usize>) {
    ranges.into_iter().unzip()
}

fn ordered_index_ranges_into_numpy<'py>(
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

fn compare_f64_values(left: f64, right: f64) -> std::cmp::Ordering {
    left.total_cmp(&right)
}

fn time_ordered_indices_single_user(timestamps: &[f64]) -> OrderedIndexRanges {
    let mut indices: Vec<usize> = (0..timestamps.len()).collect();
    indices.par_sort_by(|&left, &right| {
        compare_f64_values(timestamps[left], timestamps[right]).then(left.cmp(&right))
    });
    let n = indices.len();
    (indices, vec![(0, n)])
}

fn time_ordered_indices_for_ord_uid_values<T: Ord + Sync>(
    uids: &[T],
    timestamps: &[f64],
) -> OrderedIndexRanges {
    let mut indices: Vec<usize> = (0..timestamps.len()).collect();
    indices.par_sort_by(|&left, &right| {
        uids[left]
            .cmp(&uids[right])
            .then(compare_f64_values(timestamps[left], timestamps[right]))
            .then(left.cmp(&right))
    });
    let ranges = ranges_from_sorted_uid_indices(uids, &indices);
    (indices, ranges)
}

fn time_ordered_indices_for_f64_uid_values(uids: &[f64], timestamps: &[f64]) -> OrderedIndexRanges {
    let mut indices: Vec<usize> = (0..timestamps.len()).collect();
    indices.par_sort_by(|&left, &right| {
        compare_f64_values(uids[left], uids[right])
            .then(compare_f64_values(timestamps[left], timestamps[right]))
            .then(left.cmp(&right))
    });
    let ranges = ranges_from_sorted_uid_indices(uids, &indices);
    (indices, ranges)
}

fn primitive_option_values<T>(array: &PrimitiveArray<T>) -> Vec<Option<T::Native>>
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

fn time_ordered_indices_from_numpy_uids(
    uids: &Bound<'_, PyAny>,
    timestamps: &[f64],
) -> PyResult<OrderedIndexRanges> {
    if let Ok(array) = uids.extract::<PyReadonlyArray1<i64>>() {
        validate_uid_len(timestamps.len(), array.len()?)?;
        return Ok(time_ordered_indices_for_ord_uid_values(
            array.as_slice()?,
            timestamps,
        ));
    }
    if let Ok(array) = uids.extract::<PyReadonlyArray1<i32>>() {
        validate_uid_len(timestamps.len(), array.len()?)?;
        return Ok(time_ordered_indices_for_ord_uid_values(
            array.as_slice()?,
            timestamps,
        ));
    }
    if let Ok(array) = uids.extract::<PyReadonlyArray1<u64>>() {
        validate_uid_len(timestamps.len(), array.len()?)?;
        return Ok(time_ordered_indices_for_ord_uid_values(
            array.as_slice()?,
            timestamps,
        ));
    }
    if let Ok(array) = uids.extract::<PyReadonlyArray1<u32>>() {
        validate_uid_len(timestamps.len(), array.len()?)?;
        return Ok(time_ordered_indices_for_ord_uid_values(
            array.as_slice()?,
            timestamps,
        ));
    }
    if let Ok(array) = uids.extract::<PyReadonlyArray1<f64>>() {
        validate_uid_len(timestamps.len(), array.len()?)?;
        return Ok(time_ordered_indices_for_f64_uid_values(
            array.as_slice()?,
            timestamps,
        ));
    }

    Err(PyValueError::new_err(
        "unsupported numpy uid dtype for time-ordered jump_lengths",
    ))
}

fn time_ordered_indices_from_arrow_uids(
    uids: PyArray,
    timestamps: &[f64],
) -> PyResult<OrderedIndexRanges> {
    let (array_ref, _field) = uids.into_inner();
    let array = array_ref.as_any();

    if let Some(array) = array.downcast_ref::<Int64Array>() {
        validate_uid_len(timestamps.len(), array.len())?;
        let values = primitive_option_values::<Int64Type>(array);
        return Ok(time_ordered_indices_for_ord_uid_values(&values, timestamps));
    }
    if let Some(array) = array.downcast_ref::<Int32Array>() {
        validate_uid_len(timestamps.len(), array.len())?;
        let values = primitive_option_values::<Int32Type>(array);
        return Ok(time_ordered_indices_for_ord_uid_values(&values, timestamps));
    }
    if let Some(array) = array.downcast_ref::<UInt64Array>() {
        validate_uid_len(timestamps.len(), array.len())?;
        let values = primitive_option_values::<UInt64Type>(array);
        return Ok(time_ordered_indices_for_ord_uid_values(&values, timestamps));
    }
    if let Some(array) = array.downcast_ref::<UInt32Array>() {
        validate_uid_len(timestamps.len(), array.len())?;
        let values = primitive_option_values::<UInt32Type>(array);
        return Ok(time_ordered_indices_for_ord_uid_values(&values, timestamps));
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
        return Ok(time_ordered_indices_for_ord_uid_values(&values, timestamps));
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
        return Ok(time_ordered_indices_for_ord_uid_values(&values, timestamps));
    }

    Err(PyValueError::new_err(
        "unsupported Arrow uid type for time-ordered jump_lengths",
    ))
}

fn time_ordered_values_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps: &[f64],
    indices: Vec<usize>,
    ranges: IndexRanges,
) -> PyResult<(Vec<usize>, IndexRanges, Vec<Vec<f64>>)> {
    validate_time_ordered_inputs(latitudes, longitudes, timestamps)?;
    validate_indexed_inputs(latitudes, longitudes, &indices, &ranges)?;
    let values = jump_lengths_indexed_batch_impl(latitudes, longitudes, &indices, &ranges)?;
    Ok((indices, ranges, values))
}

fn time_ordered_flat_values_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps: &[f64],
    indices: Vec<usize>,
    ranges: IndexRanges,
) -> PyResult<(Vec<usize>, IndexRanges, Vec<f64>)> {
    validate_time_ordered_inputs(latitudes, longitudes, timestamps)?;
    validate_indexed_inputs(latitudes, longitudes, &indices, &ranges)?;
    let values = jump_lengths_indexed_flat_impl(latitudes, longitudes, &indices, &ranges)?;
    Ok((indices, ranges, values))
}

fn jump_lengths_batch_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ranges: &[(usize, usize)],
) -> PyResult<Vec<Vec<f64>>> {
    validate_coord_ranges(latitudes, longitudes, ranges)?;

    Ok(ranges
        .par_iter()
        .map(|&(start, end)| jump_lengths_for_range(latitudes, longitudes, start, end))
        .collect())
}

fn jump_lengths_flat_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ranges: &[(usize, usize)],
) -> PyResult<Vec<f64>> {
    validate_coord_ranges(latitudes, longitudes, ranges)?;

    Ok(ranges
        .iter()
        .flat_map(|&(start, end)| jump_lengths_for_range(latitudes, longitudes, start, end))
        .collect())
}

fn jump_lengths_indexed_batch_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> PyResult<Vec<Vec<f64>>> {
    validate_indexed_inputs(latitudes, longitudes, indices, ranges)?;

    Ok(ranges
        .par_iter()
        .map(|&(start, end)| {
            jump_lengths_for_indexed_range(latitudes, longitudes, indices, start, end)
        })
        .collect())
}

fn jump_lengths_indexed_flat_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> PyResult<Vec<f64>> {
    validate_indexed_inputs(latitudes, longitudes, indices, ranges)?;

    Ok(ranges
        .iter()
        .flat_map(|&(start, end)| {
            jump_lengths_for_indexed_range(latitudes, longitudes, indices, start, end)
        })
        .collect())
}

#[pyfunction]
pub(crate) fn jump_lengths_km(latitudes: Vec<f64>, longitudes: Vec<f64>) -> PyResult<Vec<f64>> {
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }

    if latitudes.is_empty() {
        return Ok(Vec::new());
    }

    let coords: Vec<(f64, f64)> = latitudes.into_iter().zip(longitudes).collect();

    let lengths: Vec<f64> = coords
        .par_windows(2)
        .map(|window| {
            let (lat1, lon1) = window[0];
            let (lat2, lon2) = window[1];
            haversine_km(lat1, lon1, lat2, lon2)
        })
        .collect();

    Ok(lengths)
}

#[pyfunction]
pub(crate) fn jump_lengths_numpy(
    latitudes: PyReadonlyArray1<f64>,
    longitudes: PyReadonlyArray1<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<Vec<f64>>> {
    jump_lengths_batch_impl(latitudes.as_slice()?, longitudes.as_slice()?, &ranges)
}

#[pyfunction]
pub(crate) fn jump_lengths_flat_numpy(
    latitudes: PyReadonlyArray1<f64>,
    longitudes: PyReadonlyArray1<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    jump_lengths_flat_impl(latitudes.as_slice()?, longitudes.as_slice()?, &ranges)
}

#[pyfunction]
pub(crate) fn jump_lengths_indexed_numpy(
    latitudes: PyReadonlyArray1<f64>,
    longitudes: PyReadonlyArray1<f64>,
    indices: PyReadonlyArray1<usize>,
    starts: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<Vec<Vec<f64>>> {
    let ranges = crate::utils::ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    jump_lengths_indexed_batch_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        indices.as_slice()?,
        &ranges,
    )
}

#[pyfunction]
pub(crate) fn jump_lengths_indexed_flat_numpy(
    latitudes: PyReadonlyArray1<f64>,
    longitudes: PyReadonlyArray1<f64>,
    indices: PyReadonlyArray1<usize>,
    starts: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<Vec<f64>> {
    let ranges = crate::utils::ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    jump_lengths_indexed_flat_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        indices.as_slice()?,
        &ranges,
    )
}

#[pyfunction]
pub(crate) fn jump_lengths_time_ordered_single_numpy(
    timestamps: PyReadonlyArray1<f64>,
    latitudes: PyReadonlyArray1<f64>,
    longitudes: PyReadonlyArray1<f64>,
) -> PyResult<Vec<Vec<f64>>> {
    let timestamps = timestamps.as_slice()?;
    let latitudes = latitudes.as_slice()?;
    let longitudes = longitudes.as_slice()?;
    let (indices, ranges) = time_ordered_indices_single_user(timestamps);
    let (_, _, values) =
        time_ordered_values_impl(latitudes, longitudes, timestamps, indices, ranges)?;
    Ok(values)
}

#[pyfunction]
pub(crate) fn jump_lengths_time_ordered_single_flat_numpy(
    timestamps: PyReadonlyArray1<f64>,
    latitudes: PyReadonlyArray1<f64>,
    longitudes: PyReadonlyArray1<f64>,
) -> PyResult<Vec<f64>> {
    let timestamps = timestamps.as_slice()?;
    let latitudes = latitudes.as_slice()?;
    let longitudes = longitudes.as_slice()?;
    let (indices, ranges) = time_ordered_indices_single_user(timestamps);
    let (_, _, values) =
        time_ordered_flat_values_impl(latitudes, longitudes, timestamps, indices, ranges)?;
    Ok(values)
}

#[pyfunction]
pub(crate) fn jump_lengths_time_ordered_numpy<'py>(
    py: Python<'py>,
    uids: &Bound<'py, PyAny>,
    timestamps: PyReadonlyArray1<'py, f64>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
) -> PyResult<PyTimeOrderedJumpLengths<'py>> {
    let timestamps = timestamps.as_slice()?;
    let latitudes = latitudes.as_slice()?;
    let longitudes = longitudes.as_slice()?;
    validate_time_ordered_inputs(latitudes, longitudes, timestamps)?;

    let (indices, ranges) = time_ordered_indices_from_numpy_uids(uids, timestamps)?;

    let (indices, ranges, values) =
        time_ordered_values_impl(latitudes, longitudes, timestamps, indices, ranges)?;
    let (indices, starts, ends) = ordered_index_ranges_into_numpy(py, (indices, ranges));
    Ok((indices, starts, ends, values))
}

#[pyfunction]
pub(crate) fn time_ordered_user_indices_numpy<'py>(
    py: Python<'py>,
    uids: &Bound<'py, PyAny>,
    timestamps: PyReadonlyArray1<'py, f64>,
) -> PyResult<PyOrderedIndexRanges<'py>> {
    Ok(ordered_index_ranges_into_numpy(
        py,
        time_ordered_indices_from_numpy_uids(uids, timestamps.as_slice()?)?,
    ))
}

#[pyfunction]
pub(crate) fn time_ordered_single_user_indices_numpy<'py>(
    py: Python<'py>,
    timestamps: PyReadonlyArray1<'py, f64>,
) -> PyResult<PyOrderedIndexRanges<'py>> {
    Ok(ordered_index_ranges_into_numpy(
        py,
        time_ordered_indices_single_user(timestamps.as_slice()?),
    ))
}

#[pyfunction]
pub(crate) fn jump_lengths_time_ordered_flat_numpy<'py>(
    py: Python<'py>,
    uids: &Bound<'py, PyAny>,
    timestamps: PyReadonlyArray1<'py, f64>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
) -> PyResult<PyTimeOrderedFlatJumpLengths<'py>> {
    let timestamps = timestamps.as_slice()?;
    let latitudes = latitudes.as_slice()?;
    let longitudes = longitudes.as_slice()?;
    validate_time_ordered_inputs(latitudes, longitudes, timestamps)?;

    let (indices, ranges) = time_ordered_indices_from_numpy_uids(uids, timestamps)?;

    let (indices, ranges, values) =
        time_ordered_flat_values_impl(latitudes, longitudes, timestamps, indices, ranges)?;
    let (indices, starts, ends) = ordered_index_ranges_into_numpy(py, (indices, ranges));
    Ok((indices, starts, ends, values))
}

#[pyfunction]
pub(crate) fn jump_lengths_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<Vec<f64>>> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    jump_lengths_batch_impl(arrow_values(&latitudes), arrow_values(&longitudes), &ranges)
}

#[pyfunction]
pub(crate) fn jump_lengths_flat_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    jump_lengths_flat_impl(arrow_values(&latitudes), arrow_values(&longitudes), &ranges)
}

#[pyfunction]
pub(crate) fn jump_lengths_indexed_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    indices: PyReadonlyArray1<usize>,
    starts: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<Vec<Vec<f64>>> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let ranges = crate::utils::ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    jump_lengths_indexed_batch_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        indices.as_slice()?,
        &ranges,
    )
}

#[pyfunction]
pub(crate) fn jump_lengths_indexed_flat_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    indices: PyReadonlyArray1<usize>,
    starts: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<Vec<f64>> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let ranges = crate::utils::ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    jump_lengths_indexed_flat_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        indices.as_slice()?,
        &ranges,
    )
}

#[pyfunction]
pub(crate) fn jump_lengths_time_ordered_single_arrow(
    timestamps: PyArray,
    latitudes: PyArray,
    longitudes: PyArray,
) -> PyResult<Vec<Vec<f64>>> {
    let timestamps = as_f64_array(timestamps, "timestamps")?;
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps = arrow_values(&timestamps);
    let (indices, ranges) = time_ordered_indices_single_user(timestamps);
    let (_, _, values) = time_ordered_values_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        timestamps,
        indices,
        ranges,
    )?;
    Ok(values)
}

#[pyfunction]
pub(crate) fn jump_lengths_time_ordered_single_flat_arrow(
    timestamps: PyArray,
    latitudes: PyArray,
    longitudes: PyArray,
) -> PyResult<Vec<f64>> {
    let timestamps = as_f64_array(timestamps, "timestamps")?;
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps = arrow_values(&timestamps);
    let (indices, ranges) = time_ordered_indices_single_user(timestamps);
    let (_, _, values) = time_ordered_flat_values_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        timestamps,
        indices,
        ranges,
    )?;
    Ok(values)
}

#[pyfunction]
pub(crate) fn jump_lengths_time_ordered_arrow<'py>(
    py: Python<'py>,
    uids: PyArray,
    timestamps: PyArray,
    latitudes: PyArray,
    longitudes: PyArray,
) -> PyResult<PyTimeOrderedJumpLengths<'py>> {
    let timestamps = as_f64_array(timestamps, "timestamps")?;
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps = arrow_values(&timestamps);
    let latitudes = arrow_values(&latitudes);
    let longitudes = arrow_values(&longitudes);
    validate_time_ordered_inputs(latitudes, longitudes, timestamps)?;

    let (indices, ranges) = time_ordered_indices_from_arrow_uids(uids, timestamps)?;

    let (indices, ranges, values) =
        time_ordered_values_impl(latitudes, longitudes, timestamps, indices, ranges)?;
    let (indices, starts, ends) = ordered_index_ranges_into_numpy(py, (indices, ranges));
    Ok((indices, starts, ends, values))
}

#[pyfunction]
pub(crate) fn time_ordered_user_indices_arrow<'py>(
    py: Python<'py>,
    uids: PyArray,
    timestamps: PyArray,
) -> PyResult<PyOrderedIndexRanges<'py>> {
    let timestamps = as_f64_array(timestamps, "timestamps")?;
    Ok(ordered_index_ranges_into_numpy(
        py,
        time_ordered_indices_from_arrow_uids(uids, arrow_values(&timestamps))?,
    ))
}

#[pyfunction]
pub(crate) fn time_ordered_single_user_indices_arrow<'py>(
    py: Python<'py>,
    timestamps: PyArray,
) -> PyResult<PyOrderedIndexRanges<'py>> {
    let timestamps = as_f64_array(timestamps, "timestamps")?;
    Ok(ordered_index_ranges_into_numpy(
        py,
        time_ordered_indices_single_user(arrow_values(&timestamps)),
    ))
}

#[pyfunction]
pub(crate) fn jump_lengths_time_ordered_flat_arrow<'py>(
    py: Python<'py>,
    uids: PyArray,
    timestamps: PyArray,
    latitudes: PyArray,
    longitudes: PyArray,
) -> PyResult<PyTimeOrderedFlatJumpLengths<'py>> {
    let timestamps = as_f64_array(timestamps, "timestamps")?;
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps = arrow_values(&timestamps);
    let latitudes = arrow_values(&latitudes);
    let longitudes = arrow_values(&longitudes);
    validate_time_ordered_inputs(latitudes, longitudes, timestamps)?;

    let (indices, ranges) = time_ordered_indices_from_arrow_uids(uids, timestamps)?;

    let (indices, ranges, values) =
        time_ordered_flat_values_impl(latitudes, longitudes, timestamps, indices, ranges)?;
    let (indices, starts, ends) = ordered_index_ranges_into_numpy(py, (indices, ranges));
    Ok((indices, starts, ends, values))
}
