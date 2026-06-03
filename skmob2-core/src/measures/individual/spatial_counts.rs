use rayon::prelude::*;
use rustc_hash::FxHashSet;

use crate::utils::{
    validate_coord_ranges, validate_indexed_coord_ranges, validate_indexed_ranges, validate_ranges,
};

pub fn number_of_visits_impl(n_values: usize, ranges: &[(usize, usize)]) -> Result<Vec<u64>, String> {
    validate_ranges(n_values, ranges)?;
    Ok(ranges
        .iter()
        .map(|&(start, end)| (end - start) as u64)
        .collect())
}

pub fn number_of_visits_indexed_impl(
    n_values: usize,
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> Result<Vec<u64>, String> {
    validate_indexed_ranges(n_values, indices, ranges)?;
    Ok(ranges
        .iter()
        .map(|&(start, end)| (end - start) as u64)
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

pub fn number_of_locations_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> Result<Vec<u64>, String> {
    validate_indexed_coord_ranges(latitudes, longitudes, indices, ranges)?;
    Ok(ranges
        .par_iter()
        .map(|&(start, end)| {
            let mut seen =
                FxHashSet::with_capacity_and_hasher(end.saturating_sub(start), Default::default());
            for &idx in &indices[start..end] {
                seen.insert((latitudes[idx].to_bits(), longitudes[idx].to_bits()));
            }
            seen.len() as u64
        })
        .collect())
}
