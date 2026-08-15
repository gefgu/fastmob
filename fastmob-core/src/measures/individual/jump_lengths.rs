use rayon::prelude::*;
use std::time::Instant;

use crate::utils::haversine::{adjacent_haversine_distances_into_km, haversine_km};
use crate::utils::{ranges_from_ends, validate_coord_ranges, validate_indexed_coord_ends_u64};

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
    indices: &[u64],
    start: u64,
    end: u64,
    valid_rows: Option<&[bool]>,
) -> Vec<f64> {
    let start = usize::try_from(start).expect("jump range start must fit usize");
    let end = usize::try_from(end).expect("jump range end must fit usize");
    let mut valid = Vec::with_capacity(end.saturating_sub(start));
    for &idx in &indices[start..end] {
        let row = usize::try_from(idx).expect("row index must fit usize");
        if valid_rows.is_none_or(|v| v[row])
            && latitudes[row].is_finite()
            && longitudes[row].is_finite()
        {
            valid.push((latitudes[row], longitudes[row]));
        }
    }

    if valid.len() < 2 {
        return Vec::new();
    }

    valid
        .windows(2)
        .map(|pair| haversine_km(pair[0].0, pair[0].1, pair[1].0, pair[1].1))
        .collect()
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
    ends: &[usize],
) -> JumpLengthsPresortedResult {
    let ranges = ranges_from_ends(ends)?;
    validate_coord_ranges(latitudes, longitudes, &ranges)?;
    let (value_starts, value_ends) = jump_offsets_for_ranges(&ranges);
    let mut values = vec![0.0; value_ends.last().copied().unwrap_or(0)];
    let mut rest = values.as_mut_slice();
    let mut chunks = Vec::with_capacity(ranges.len());
    for (&start, &end) in value_starts.iter().zip(&value_ends) {
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
    Ok((value_starts, value_ends, values))
}

pub fn jump_lengths_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[u64],
    ends: &[u64],
    valid_rows: Option<&[bool]>,
) -> JumpLengthsPresortedResult {
    let profile = std::env::var_os("FASTMOB_PROFILE_JUMP_LENGTHS").is_some();
    let total_started = Instant::now();
    let validation_started = Instant::now();
    validate_indexed_coord_ends_u64(latitudes, longitudes, indices, ends)?;
    if let Some(valid_rows) = valid_rows
        && valid_rows.len() != latitudes.len()
    {
        return Err("valid_rows and coordinates must have the same length".to_string());
    }
    if profile {
        eprintln!(
            "[jump_lengths::rust] validation: {:.6}s",
            validation_started.elapsed().as_secs_f64()
        );
    }

    let kernel_started = Instant::now();
    let grouped_values: Vec<Vec<f64>> = (0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            jump_lengths_for_indexed_range(latitudes, longitudes, indices, start, end, valid_rows)
        })
        .collect();
    if profile {
        eprintln!(
            "[jump_lengths::rust] indexed kernel and per-user result allocation: {:.6}s",
            kernel_started.elapsed().as_secs_f64()
        );
    }

    let flatten_started = Instant::now();
    let mut value_starts = Vec::with_capacity(grouped_values.len());
    let mut value_ends = Vec::with_capacity(grouped_values.len());
    let total_len: usize = grouped_values.iter().map(Vec::len).sum();
    let mut values = Vec::with_capacity(total_len);
    for mut group in grouped_values {
        value_starts.push(values.len());
        values.append(&mut group);
        value_ends.push(values.len());
    }

    if profile {
        eprintln!(
            "[jump_lengths::rust] flattening: {:.6}s; total: {:.6}s",
            flatten_started.elapsed().as_secs_f64(),
            total_started.elapsed().as_secs_f64()
        );
    }
    Ok((value_starts, value_ends, values))
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
