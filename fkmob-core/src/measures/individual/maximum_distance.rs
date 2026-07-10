use rayon::prelude::*;

use crate::utils::haversine::{adjacent_haversine_max_km, haversine_km};
use crate::utils::{ranges_from_ends, validate_coord_ranges, validate_indexed_coord_ends};

pub fn maximum_distance_ranges_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ranges: &[(usize, usize)],
) -> Result<Vec<f64>, String> {
    validate_coord_ranges(latitudes, longitudes, ranges)?;

    let results: Vec<f64> = ranges
        .par_iter()
        .map(|&(start, end)| {
            if end - start < 2 {
                return f64::NAN;
            }
            adjacent_haversine_max_km(latitudes, longitudes, start, end)
        })
        .collect();

    Ok(results)
}

pub fn maximum_distance_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ends: &[usize],
) -> Result<Vec<f64>, String> {
    let ranges = ranges_from_ends(ends)?;
    maximum_distance_ranges_impl(latitudes, longitudes, &ranges)
}

pub fn maximum_distance_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
) -> Result<Vec<f64>, String> {
    validate_indexed_coord_ends(latitudes, longitudes, indices, ends)?;

    let results: Vec<f64> = (0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
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
                return f64::NAN;
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
                .fold(0.0f64, f64::max)
        })
        .collect();

    Ok(results)
}
