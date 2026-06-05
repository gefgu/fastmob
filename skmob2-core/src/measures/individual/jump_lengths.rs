use rayon::prelude::*;

use crate::measures::individual::time_ordering::IndexRanges;
use crate::utils::haversine::{adjacent_haversine_distances_into_km, haversine_km};
use crate::utils::{validate_coord_ranges, validate_indexed_coord_ranges};

type JumpLengthsPresortedResult = Result<(Vec<usize>, Vec<usize>, Vec<f64>), String>;

pub fn write_jump_lengths_presorted_range(
    latitudes: &[f64],
    longitudes: &[f64],
    start: usize,
    end: usize,
    out: &mut [f64],
) {
    if end - start < 2 {
        return;
    }

    adjacent_haversine_distances_into_km(latitudes, longitudes, start, end, out);
}

pub fn jump_lengths_for_indexed_range(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    start: usize,
    end: usize,
    valid_rows: Option<&[bool]>,
) -> Vec<f64> {
    let valid: Vec<usize> = indices[start..end]
        .iter()
        .copied()
        .filter(|&idx| {
            valid_rows.is_none_or(|v| v[idx])
                && latitudes[idx].is_finite()
                && longitudes[idx].is_finite()
        })
        .collect();

    if valid.len() < 2 {
        return Vec::new();
    }

    (1..valid.len())
        .map(|pos| {
            haversine_km(
                latitudes[valid[pos - 1]],
                longitudes[valid[pos - 1]],
                latitudes[valid[pos]],
                longitudes[valid[pos]],
            )
        })
        .collect()
}

pub fn validate_time_ordered_inputs(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps: &[f64],
) -> Result<(), String> {
    if latitudes.len() != longitudes.len() {
        return Err("latitudes and longitudes must have the same length".to_string());
    }
    if latitudes.len() != timestamps.len() {
        return Err("latitudes, longitudes, and timestamps must have the same length".to_string());
    }
    Ok(())
}

pub fn time_ordered_flat_values_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps: &[f64],
    indices: Vec<usize>,
    ranges: IndexRanges,
    valid_rows: Option<&[bool]>,
) -> Result<(Vec<usize>, IndexRanges, Vec<f64>), String> {
    validate_time_ordered_inputs(latitudes, longitudes, timestamps)?;
    validate_indexed_coord_ranges(latitudes, longitudes, &indices, &ranges)?;
    let values = jump_lengths_indexed_flat_impl(latitudes, longitudes, &indices, &ranges, valid_rows)?;
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

pub fn jump_lengths_presorted_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ranges: &[(usize, usize)],
) -> JumpLengthsPresortedResult {
    validate_coord_ranges(latitudes, longitudes, ranges)?;
    let (starts, ends) = jump_offsets_for_ranges(ranges);
    let mut values = vec![0.0; ends.last().copied().unwrap_or(0)];
    let mut rest = values.as_mut_slice();
    let mut chunks = Vec::with_capacity(ranges.len());
    for (&start, &end) in starts.iter().zip(&ends) {
        let len = end - start;
        let (chunk, next) = rest.split_at_mut(len);
        chunks.push(chunk);
        rest = next;
    }

    chunks
        .into_par_iter()
        .zip(ranges.par_iter())
        .for_each(|(chunk, &(start, end))| {
            write_jump_lengths_presorted_range(latitudes, longitudes, start, end, chunk);
        });
    Ok((starts, ends, values))
}

pub fn jump_lengths_indexed_flat_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
    valid_rows: Option<&[bool]>,
) -> Result<Vec<f64>, String> {
    validate_indexed_coord_ranges(latitudes, longitudes, indices, ranges)?;

    Ok(ranges
        .par_iter()
        .flat_map(|&(start, end)| {
            jump_lengths_for_indexed_range(latitudes, longitudes, indices, start, end, valid_rows)
        })
        .collect())
}

pub fn jump_lengths_km(latitudes: Vec<f64>, longitudes: Vec<f64>) -> Result<Vec<f64>, String> {
    if latitudes.len() != longitudes.len() {
        return Err("latitudes and longitudes must have the same length".to_string());
    }

    if latitudes.is_empty() {
        return Ok(Vec::new());
    }

    let coords: Vec<(f64, f64)> = latitudes.into_iter().zip(longitudes).collect();

    let lengths: Vec<f64> = (0..coords.len().saturating_sub(1))
        .into_par_iter()
        .map(|idx| {
            let (lat1, lon1) = coords[idx];
            let (lat2, lon2) = coords[idx + 1];
            haversine_km(lat1, lon1, lat2, lon2)
        })
        .collect();

    Ok(lengths)
}
