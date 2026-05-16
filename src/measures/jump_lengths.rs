use numkong::Haversine as NumKongHaversine;
use numpy::PyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use rayon::prelude::*;

use crate::haversine::haversine_km;
use crate::time_ordering::IndexRanges;
use crate::utils::{validate_coord_ranges, validate_indexed_coord_ranges};

const GEO_HAVERSINE_RADIUS_M: f64 = 6_371_008.8;
const NUMKONG_HAVERSINE_RADIUS_M: f64 = 6_335_439.0;
const NUMKONG_TO_GEO_KM: f64 = GEO_HAVERSINE_RADIUS_M / NUMKONG_HAVERSINE_RADIUS_M / 1000.0;

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
    if end - start < 2 {
        return Vec::new();
    }

    let latitudes_rad: Vec<f64> = latitudes[start..end]
        .iter()
        .map(|lat| lat.to_radians())
        .collect();
    let longitudes_rad: Vec<f64> = longitudes[start..end]
        .iter()
        .map(|lon| lon.to_radians())
        .collect();
    let mut distances = vec![0.0; end - start - 1];

    f64::haversine(
        &latitudes_rad[..latitudes_rad.len() - 1],
        &longitudes_rad[..longitudes_rad.len() - 1],
        &latitudes_rad[1..],
        &longitudes_rad[1..],
        &mut distances,
    )
    .expect("adjacent coordinate slices have matching lengths");

    distances
        .iter_mut()
        .for_each(|distance| *distance *= NUMKONG_TO_GEO_KM);
    distances
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
