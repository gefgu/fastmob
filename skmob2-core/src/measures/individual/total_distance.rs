use rayon::prelude::*;

use crate::utils::haversine::{adjacent_haversine_sum_km, haversine_km};
use crate::utils::{validate_coord_ranges, validate_indexed_coord_ranges};

pub fn total_distance_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ranges: &[(usize, usize)],
) -> Result<Vec<f64>, String> {
    validate_coord_ranges(latitudes, longitudes, ranges)?;

    let results: Vec<f64> = ranges
        .par_iter()
        .map(|&(start, end)| {
            if end - start < 2 {
                return 0.0;
            }
            adjacent_haversine_sum_km(latitudes, longitudes, start, end)
        })
        .collect();

    Ok(results)
}

pub fn total_distance_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> Result<Vec<f64>, String> {
    validate_indexed_coord_ranges(latitudes, longitudes, indices, ranges)?;

    let results: Vec<f64> = ranges
        .par_iter()
        .map(|&(start, end)| {
            if end - start < 2 {
                return 0.0;
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
                .sum()
        })
        .collect();

    Ok(results)
}
