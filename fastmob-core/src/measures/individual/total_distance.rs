use rayon::prelude::*;

use crate::utils::haversine::{adjacent_haversine_sum_km, haversine_km};
use crate::utils::{validate_coord_ends, validate_coord_ranges, validate_indexed_coord_ends_u64};

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

pub fn total_distance_presorted_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ends: &[usize],
) -> Result<Vec<f64>, String> {
    validate_coord_ends(latitudes, longitudes, ends)?;

    let results: Vec<f64> = (0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
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
    indices: &[u64],
    ends: &[u64],
    valid_rows: Option<&[bool]>,
) -> Result<Vec<f64>, String> {
    validate_indexed_coord_ends_u64(latitudes, longitudes, indices, ends)?;

    let results: Vec<f64> = (0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = (if i == 0 { 0 } else { ends[i - 1] }) as usize;
            let end = ends[i] as usize;
            let valid: Vec<usize> = indices[start..end]
                .iter()
                .copied()
                .map(|idx| idx as usize)
                .filter(|&idx| {
                    valid_rows.is_none_or(|v| v[idx])
                        && latitudes[idx].is_finite()
                        && longitudes[idx].is_finite()
                })
                .collect();
            if valid.len() < 2 {
                return 0.0;
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
                .sum()
        })
        .collect();

    Ok(results)
}
