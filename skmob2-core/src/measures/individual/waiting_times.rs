use rayon::prelude::*;

use crate::utils::{validate_indexed_ranges, validate_ranges};

pub fn waiting_times_for_range(timestamps_s: &[f64], start: usize, end: usize) -> Vec<f64> {
    if end - start < 2 {
        return Vec::new();
    }
    let mut waits = Vec::with_capacity(end - start - 1);
    for idx in start + 1..end {
        waits.push(timestamps_s[idx] - timestamps_s[idx - 1]);
    }
    waits
}

pub fn waiting_times_for_indexed_range(
    timestamps_s: &[f64],
    indices: &[usize],
    start: usize,
    end: usize,
) -> Vec<f64> {
    if end - start < 2 {
        return Vec::new();
    }
    let mut waits = Vec::with_capacity(end - start - 1);
    for pos in start + 1..end {
        waits.push(timestamps_s[indices[pos]] - timestamps_s[indices[pos - 1]]);
    }
    waits
}

pub fn waiting_times_impl(
    timestamps_s: &[f64],
    ranges: &[(usize, usize)],
) -> Result<Vec<Vec<f64>>, String> {
    validate_ranges(timestamps_s.len(), ranges)?;

    Ok(ranges
        .par_iter()
        .map(|&(start, end)| waiting_times_for_range(timestamps_s, start, end))
        .collect())
}

pub fn waiting_times_flat_impl(
    timestamps_s: &[f64],
    ranges: &[(usize, usize)],
) -> Result<Vec<f64>, String> {
    validate_ranges(timestamps_s.len(), ranges)?;

    let total_len = ranges
        .iter()
        .map(|&(start, end)| end.saturating_sub(start).saturating_sub(1))
        .sum();
    let mut waits = Vec::with_capacity(total_len);
    for &(start, end) in ranges {
        for idx in start + 1..end {
            waits.push(timestamps_s[idx] - timestamps_s[idx - 1]);
        }
    }
    Ok(waits)
}

pub fn waiting_times_indexed_impl(
    timestamps_s: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> Result<Vec<Vec<f64>>, String> {
    validate_indexed_ranges(timestamps_s.len(), indices, ranges)?;

    Ok(ranges
        .par_iter()
        .map(|&(start, end)| waiting_times_for_indexed_range(timestamps_s, indices, start, end))
        .collect())
}

pub fn waiting_times_indexed_flat_impl(
    timestamps_s: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> Result<Vec<f64>, String> {
    validate_indexed_ranges(timestamps_s.len(), indices, ranges)?;

    let total_len = ranges
        .iter()
        .map(|&(start, end)| end.saturating_sub(start).saturating_sub(1))
        .sum();
    let mut waits = Vec::with_capacity(total_len);
    for &(start, end) in ranges {
        for pos in start + 1..end {
            waits.push(timestamps_s[indices[pos]] - timestamps_s[indices[pos - 1]]);
        }
    }
    Ok(waits)
}

pub fn waiting_times_seconds(
    timestamps_s: Vec<f64>,
    ranges: Vec<(usize, usize)>,
) -> Result<Vec<Vec<f64>>, String> {
    waiting_times_impl(&timestamps_s, &ranges)
}
