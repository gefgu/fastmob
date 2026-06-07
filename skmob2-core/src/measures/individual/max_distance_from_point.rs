use rayon::prelude::*;

use crate::utils::haversine::haversine_km;
use crate::utils::{validate_coord_ends, validate_coord_ranges, validate_indexed_coord_ends};

pub fn max_distance_from_point_impl(
    home_lats: &[f64],
    home_lngs: &[f64],
    latitudes: &[f64],
    longitudes: &[f64],
    ranges: &[(usize, usize)],
) -> Result<Vec<f64>, String> {
    if home_lats.len() != home_lngs.len() {
        return Err("home_lats and home_lngs must have the same length".to_string());
    }
    if home_lats.len() != ranges.len() {
        return Err("home coordinates and ranges must have the same length".to_string());
    }
    if latitudes.len() != longitudes.len() {
        return Err("latitudes and longitudes must have the same length".to_string());
    }
    validate_coord_ranges(latitudes, longitudes, ranges)?;

    let results: Vec<f64> = ranges
        .par_iter()
        .enumerate()
        .map(|(i, &(start, end))| {
            let home_lat = home_lats[i];
            let home_lng = home_lngs[i];
            (start..end)
                .map(|idx| haversine_km(home_lat, home_lng, latitudes[idx], longitudes[idx]))
                .fold(0.0f64, f64::max)
        })
        .collect();

    Ok(results)
}

pub fn max_distance_from_point_from_ends_impl(
    home_lats: &[f64],
    home_lngs: &[f64],
    latitudes: &[f64],
    longitudes: &[f64],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
) -> Result<Vec<f64>, String> {
    if home_lats.len() != home_lngs.len() {
        return Err("home_lats and home_lngs must have the same length".to_string());
    }
    if home_lats.len() != ends.len() {
        return Err("home coordinates and ranges must have the same length".to_string());
    }
    validate_coord_ends(latitudes, longitudes, ends)?;

    let results: Vec<f64> = (0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let home_lat = home_lats[i];
            let home_lng = home_lngs[i];
            (start..end)
                .filter(|&idx| valid_rows.is_none_or(|v| v[idx]))
                .map(|idx| haversine_km(home_lat, home_lng, latitudes[idx], longitudes[idx]))
                .fold(0.0f64, f64::max)
        })
        .collect();

    Ok(results)
}

pub fn max_distance_from_point_indexed_impl(
    home_lats: &[f64],
    home_lngs: &[f64],
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
) -> Result<Vec<f64>, String> {
    if home_lats.len() != home_lngs.len() || home_lats.len() != ends.len() {
        return Err("home coordinates and ranges must have the same length".to_string());
    }
    validate_indexed_coord_ends(latitudes, longitudes, indices, ends)?;

    Ok((0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let home_lat = home_lats[i];
            let home_lng = home_lngs[i];
            indices[start..end]
                .iter()
                .filter(|&&idx| valid_rows.is_none_or(|v| v[idx]))
                .map(|&idx| haversine_km(home_lat, home_lng, latitudes[idx], longitudes[idx]))
                .fold(0.0f64, f64::max)
        })
        .collect())
}
