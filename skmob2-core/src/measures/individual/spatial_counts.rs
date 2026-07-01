use rayon::prelude::*;
use rustc_hash::FxHashSet;

use crate::utils::{
    validate_coord_ends, validate_coord_ranges, validate_ends, validate_indexed_coord_ends,
    validate_indexed_ends, validate_ranges,
};

pub fn number_of_visits_impl(
    n_values: usize,
    ranges: &[(usize, usize)],
) -> Result<Vec<u64>, String> {
    validate_ranges(n_values, ranges)?;
    Ok(ranges
        .iter()
        .map(|&(start, end)| (end - start) as u64)
        .collect())
}

pub fn number_of_visits_from_ends_impl(
    n_values: usize,
    ends: &[usize],
) -> Result<Vec<u64>, String> {
    validate_ends(n_values, ends)?;
    Ok((0..ends.len())
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            (ends[i] - start) as u64
        })
        .collect())
}

pub fn number_of_visits_indexed_impl(
    n_values: usize,
    indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
) -> Result<Vec<u64>, String> {
    validate_indexed_ends(n_values, indices, ends)?;
    Ok((0..ends.len())
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            match valid_rows {
                None => (end - start) as u64,
                Some(valid) => indices[start..end]
                    .iter()
                    .filter(|&&idx| valid[idx])
                    .count() as u64,
            }
        })
        .collect())
}

pub fn number_of_locations_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ranges: &[(usize, usize)],
) -> Result<Vec<u64>, String> {
    validate_coord_ranges(latitudes, longitudes, ranges)?;
    Ok(ranges
        .par_iter()
        .map(|&(start, end)| {
            let mut seen =
                FxHashSet::with_capacity_and_hasher(end.saturating_sub(start), Default::default());
            for idx in start..end {
                seen.insert((latitudes[idx].to_bits(), longitudes[idx].to_bits()));
            }
            seen.len() as u64
        })
        .collect())
}

pub fn number_of_locations_from_ends_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ends: &[usize],
) -> Result<Vec<u64>, String> {
    validate_coord_ends(latitudes, longitudes, ends)?;
    Ok((0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let mut seen =
                FxHashSet::with_capacity_and_hasher(end.saturating_sub(start), Default::default());
            for idx in start..end {
                seen.insert((latitudes[idx].to_bits(), longitudes[idx].to_bits()));
            }
            seen.len() as u64
        })
        .collect())
}

pub fn number_of_locations_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
) -> Result<Vec<u64>, String> {
    validate_indexed_coord_ends(latitudes, longitudes, indices, ends)?;
    Ok((0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let mut seen =
                FxHashSet::with_capacity_and_hasher(end.saturating_sub(start), Default::default());
            for &idx in &indices[start..end] {
                if valid_rows.is_none_or(|v| v[idx])
                    && latitudes[idx].is_finite()
                    && longitudes[idx].is_finite()
                {
                    seen.insert((latitudes[idx].to_bits(), longitudes[idx].to_bits()));
                }
            }
            seen.len() as u64
        })
        .collect())
}
