use arrow_array::{
    Array, Int32Array, Int64Array, LargeStringArray, StringArray, UInt32Array, UInt64Array,
    types::{Int32Type, Int64Type, UInt32Type, UInt64Type},
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use rayon::prelude::*;

use crate::haversine::haversine_km;
use crate::utils::{
    arrow_values, as_f64_array, f64_results_into_arrow, primitive_option_values,
    ranges_from_sorted_values, ranges_from_starts_ends, split_ranges, validate_coord_ranges,
    validate_indexed_coord_ranges, validate_uid_len,
};

type IndexRanges = Vec<(usize, usize)>;
type OrderedIndexRanges = (Vec<usize>, IndexRanges);
type PyOrderedIndexRanges<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
);
type PyGroupedJumpLengths<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<f64>>,
);
type PyGroupedJumpLengthsArrow<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    PyArray,
);
type PyTimeOrderedJumpLengths<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<f64>>,
);
type PyTimeOrderedJumpLengthsArrow<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    PyArray,
);
type PyTimeOrderedFlatJumpLengths<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<f64>>,
);
type PyTimeOrderedFlatJumpLengthsArrow<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    PyArray,
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

fn jump_offsets_for_ranges(ranges: &[(usize, usize)]) -> (Vec<usize>, Vec<usize>) {
    let mut starts = Vec::with_capacity(ranges.len());
    let mut ends = Vec::with_capacity(ranges.len());
    let mut offset = 0usize;
    for &(start, end) in ranges {
        starts.push(offset);
        offset += end.saturating_sub(start).saturating_sub(1);
        ends.push(offset);
    }
    (starts, ends)
}

fn value_offsets_into_numpy<'py>(
    py: Python<'py>,
    ranges: &[(usize, usize)],
) -> (Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>) {
    let (starts, ends) = jump_offsets_for_ranges(ranges);
    (starts.into_pyarray(py), ends.into_pyarray(py))
}

fn time_ordered_indices_single_user(timestamps: &[f64]) -> OrderedIndexRanges {
    let mut indices: Vec<usize> = (0..timestamps.len()).collect();
    indices.par_sort_by(|&left, &right| {
        timestamps[left].total_cmp(&timestamps[right]).then(left.cmp(&right))
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
            .then(timestamps[left].total_cmp(&timestamps[right]))
            .then(left.cmp(&right))
    });
    let ranges = ranges_from_sorted_values(uids, &indices);
    (indices, ranges)
}

fn time_ordered_indices_for_f64_uid_values(uids: &[f64], timestamps: &[f64]) -> OrderedIndexRanges {
    let mut indices: Vec<usize> = (0..timestamps.len()).collect();
    indices.par_sort_by(|&left, &right| {
        uids[left]
            .total_cmp(&uids[right])
            .then(timestamps[left].total_cmp(&timestamps[right]))
            .then(left.cmp(&right))
    });
    let ranges = ranges_from_sorted_values(uids, &indices);
    (indices, ranges)
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

fn time_ordered_flat_values_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps: &[f64],
    indices: Vec<usize>,
    ranges: IndexRanges,
) -> PyResult<(Vec<usize>, IndexRanges, Vec<f64>)> {
    validate_time_ordered_inputs(latitudes, longitudes, timestamps)?;
    validate_indexed_coord_ranges(latitudes, longitudes, &indices, &ranges)?;
    let values = jump_lengths_indexed_flat_impl(latitudes, longitudes, &indices, &ranges)?;
    Ok((indices, ranges, values))
}

fn jump_lengths_flat_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ranges: &[(usize, usize)],
) -> PyResult<Vec<f64>> {
    validate_coord_ranges(latitudes, longitudes, ranges)?;

    Ok(ranges
        .par_iter()
        .flat_map(|&(start, end)| jump_lengths_for_range(latitudes, longitudes, start, end))
        .collect())
}

fn jump_lengths_indexed_flat_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> PyResult<Vec<f64>> {
    validate_indexed_coord_ranges(latitudes, longitudes, indices, ranges)?;

    Ok(ranges
        .par_iter()
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
pub(crate) fn jump_lengths_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<PyGroupedJumpLengths<'py>> {
    let (value_starts, value_ends) = value_offsets_into_numpy(py, &ranges);
    let values = jump_lengths_flat_impl(latitudes.as_slice()?, longitudes.as_slice()?, &ranges)?;
    Ok((value_starts, value_ends, values.into_pyarray(py)))
}

#[pyfunction]
pub(crate) fn jump_lengths_flat_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(
        jump_lengths_flat_impl(latitudes.as_slice()?, longitudes.as_slice()?, &ranges)?
            .into_pyarray(py),
    )
}

#[pyfunction]
pub(crate) fn jump_lengths_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    starts: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyGroupedJumpLengths<'py>> {
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    let (value_starts, value_ends) = value_offsets_into_numpy(py, &ranges);
    let values = jump_lengths_indexed_flat_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        indices.as_slice()?,
        &ranges,
    )?;
    Ok((value_starts, value_ends, values.into_pyarray(py)))
}

#[pyfunction]
pub(crate) fn jump_lengths_indexed_flat_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    starts: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    Ok(jump_lengths_indexed_flat_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        indices.as_slice()?,
        &ranges,
    )?
    .into_pyarray(py))
}

#[pyfunction]
pub(crate) fn jump_lengths_time_ordered_single_numpy<'py>(
    py: Python<'py>,
    timestamps: PyReadonlyArray1<'py, f64>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
) -> PyResult<PyGroupedJumpLengths<'py>> {
    let timestamps = timestamps.as_slice()?;
    let latitudes = latitudes.as_slice()?;
    let longitudes = longitudes.as_slice()?;
    let (indices, ranges) = time_ordered_indices_single_user(timestamps);
    let (_, ranges, values) =
        time_ordered_flat_values_impl(latitudes, longitudes, timestamps, indices, ranges)?;
    let (value_starts, value_ends) = value_offsets_into_numpy(py, &ranges);
    Ok((value_starts, value_ends, values.into_pyarray(py)))
}

#[pyfunction]
pub(crate) fn jump_lengths_time_ordered_single_flat_numpy<'py>(
    py: Python<'py>,
    timestamps: PyReadonlyArray1<'py, f64>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let timestamps = timestamps.as_slice()?;
    let latitudes = latitudes.as_slice()?;
    let longitudes = longitudes.as_slice()?;
    let (indices, ranges) = time_ordered_indices_single_user(timestamps);
    let (_, _, values) =
        time_ordered_flat_values_impl(latitudes, longitudes, timestamps, indices, ranges)?;
    Ok(values.into_pyarray(py))
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
        time_ordered_flat_values_impl(latitudes, longitudes, timestamps, indices, ranges)?;
    let (value_starts, value_ends) = value_offsets_into_numpy(py, &ranges);
    let (indices, starts, ends) = ordered_index_ranges_into_numpy(py, (indices, ranges));
    Ok((
        indices,
        starts,
        ends,
        value_starts,
        value_ends,
        values.into_pyarray(py),
    ))
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
    Ok((indices, starts, ends, values.into_pyarray(py)))
}

#[pyfunction]
pub(crate) fn jump_lengths_arrow<'py>(
    py: Python<'py>,
    latitudes: PyArray,
    longitudes: PyArray,
    ranges: Vec<(usize, usize)>,
) -> PyResult<PyGroupedJumpLengthsArrow<'py>> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let (value_starts, value_ends) = value_offsets_into_numpy(py, &ranges);
    let values =
        jump_lengths_flat_impl(arrow_values(&latitudes), arrow_values(&longitudes), &ranges)?;
    Ok((value_starts, value_ends, f64_results_into_arrow(values)))
}

#[pyfunction]
pub(crate) fn jump_lengths_flat_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    ranges: Vec<(usize, usize)>,
) -> PyResult<PyArray> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    Ok(f64_results_into_arrow(jump_lengths_flat_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        &ranges,
    )?))
}

#[pyfunction]
pub(crate) fn jump_lengths_indexed_arrow<'py>(
    py: Python<'py>,
    latitudes: PyArray,
    longitudes: PyArray,
    indices: PyReadonlyArray1<'py, usize>,
    starts: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyGroupedJumpLengthsArrow<'py>> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    let (value_starts, value_ends) = value_offsets_into_numpy(py, &ranges);
    let values = jump_lengths_indexed_flat_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        indices.as_slice()?,
        &ranges,
    )?;
    Ok((value_starts, value_ends, f64_results_into_arrow(values)))
}

#[pyfunction]
pub(crate) fn jump_lengths_indexed_flat_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    indices: PyReadonlyArray1<usize>,
    starts: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<PyArray> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    Ok(f64_results_into_arrow(jump_lengths_indexed_flat_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        indices.as_slice()?,
        &ranges,
    )?))
}

#[pyfunction]
pub(crate) fn jump_lengths_time_ordered_single_arrow<'py>(
    py: Python<'py>,
    timestamps: PyArray,
    latitudes: PyArray,
    longitudes: PyArray,
) -> PyResult<PyGroupedJumpLengthsArrow<'py>> {
    let timestamps = as_f64_array(timestamps, "timestamps")?;
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps = arrow_values(&timestamps);
    let (indices, ranges) = time_ordered_indices_single_user(timestamps);
    let (_, ranges, values) = time_ordered_flat_values_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        timestamps,
        indices,
        ranges,
    )?;
    let (value_starts, value_ends) = value_offsets_into_numpy(py, &ranges);
    Ok((value_starts, value_ends, f64_results_into_arrow(values)))
}

#[pyfunction]
pub(crate) fn jump_lengths_time_ordered_single_flat_arrow(
    timestamps: PyArray,
    latitudes: PyArray,
    longitudes: PyArray,
) -> PyResult<PyArray> {
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
    Ok(f64_results_into_arrow(values))
}

#[pyfunction]
pub(crate) fn jump_lengths_time_ordered_arrow<'py>(
    py: Python<'py>,
    uids: PyArray,
    timestamps: PyArray,
    latitudes: PyArray,
    longitudes: PyArray,
) -> PyResult<PyTimeOrderedJumpLengthsArrow<'py>> {
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
    let (value_starts, value_ends) = value_offsets_into_numpy(py, &ranges);
    let (indices, starts, ends) = ordered_index_ranges_into_numpy(py, (indices, ranges));
    Ok((
        indices,
        starts,
        ends,
        value_starts,
        value_ends,
        f64_results_into_arrow(values),
    ))
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
) -> PyResult<PyTimeOrderedFlatJumpLengthsArrow<'py>> {
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
    Ok((indices, starts, ends, f64_results_into_arrow(values)))
}
