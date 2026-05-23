use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use rayon::prelude::*;

use crate::haversine::{adjacent_haversine_distances_km, haversine_km};
use crate::time_ordering::IndexRanges;
use crate::utils::{validate_coord_ranges, validate_indexed_coord_ranges};

pub(super) type PyPresortedJumpLengths<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<f64>>,
);
pub(super) type PyPresortedJumpLengthsArrow<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    PyArray,
);
pub(super) type PyNonOrderedJumpLengths<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<f64>>,
);
pub(super) type PyNonOrderedJumpLengthsArrow<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    PyArray,
);

fn jump_lengths_presorted_range(
    latitudes: &[f64],
    longitudes: &[f64],
    start: usize,
    end: usize,
) -> Vec<f64> {
    adjacent_haversine_distances_km(latitudes, longitudes, start, end)
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

pub(super) fn validate_time_ordered_inputs(
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

pub(super) fn time_ordered_flat_values_impl(
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

pub(super) fn jump_lengths_presorted_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ranges: &[(usize, usize)],
) -> PyResult<(Vec<usize>, Vec<usize>, Vec<f64>)> {
    validate_coord_ranges(latitudes, longitudes, ranges)?;
    let (starts, ends) = jump_offsets_for_ranges(ranges);

    let values = ranges
        .par_iter()
        .flat_map(|&(start, end)| jump_lengths_presorted_range(latitudes, longitudes, start, end))
        .collect();
    Ok((starts, ends, values))
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
