use rayon::prelude::*;

use crate::utils::{
    ends_from_ranges, validate_coord_ends, validate_coord_ranges, validate_indexed_coord_ends,
};

pub type UserIndexRanges = (Vec<usize>, Vec<(usize, usize)>);

pub fn rog_for_slice(coords: &[(f64, f64)]) -> f64 {
    let n = coords.len();
    if n == 0 {
        return 0.0;
    }

    let (lat_sum, lng_sum) = coords
        .iter()
        .fold((0.0f64, 0.0f64), |(ls, ns), &(lat, lng)| {
            (ls + lat, ns + lng)
        });
    let cm_lat = lat_sum / n as f64;
    let cm_lng = lng_sum / n as f64;

    let cm_lat_rad = cm_lat.to_radians();
    let cm_lng_rad = cm_lng.to_radians();
    let cos_cm_lat = cm_lat_rad.cos();

    let sum_sq: f64 = coords
        .iter()
        .map(|&(lat, lng)| {
            let lat_rad = lat.to_radians();
            let dlat = lat_rad - cm_lat_rad;
            let dlng = lng.to_radians() - cm_lng_rad;
            let a = (dlat / 2.0).sin().powi(2)
                + cos_cm_lat * lat_rad.cos() * (dlng / 2.0).sin().powi(2);
            let d = 2.0 * a.sqrt().asin() * 6371.0088;
            d * d
        })
        .sum();

    (sum_sq / n as f64).sqrt()
}

pub fn rog_for_parallel_slices(
    latitudes: &[f64],
    longitudes: &[f64],
    start: usize,
    end: usize,
) -> f64 {
    let n = end - start;
    if n == 0 {
        return 0.0;
    }

    let (lat_sum, lng_sum) = (start..end).fold((0.0f64, 0.0f64), |(ls, ns), idx| {
        (ls + latitudes[idx], ns + longitudes[idx])
    });
    let cm_lat = lat_sum / n as f64;
    let cm_lng = lng_sum / n as f64;

    let cm_lat_rad = cm_lat.to_radians();
    let cm_lng_rad = cm_lng.to_radians();
    let cos_cm_lat = cm_lat_rad.cos();

    let sum_sq: f64 = (start..end)
        .map(|idx| {
            let lat_rad = latitudes[idx].to_radians();
            let dlat = lat_rad - cm_lat_rad;
            let dlng = longitudes[idx].to_radians() - cm_lng_rad;
            let a = (dlat / 2.0).sin().powi(2)
                + cos_cm_lat * lat_rad.cos() * (dlng / 2.0).sin().powi(2);
            let d = 2.0 * a.sqrt().asin() * 6371.0088;
            d * d
        })
        .sum();

    (sum_sq / n as f64).sqrt()
}

pub fn rog_for_indexed_slice(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    start: usize,
    end: usize,
) -> f64 {
    let n = end - start;
    if n == 0 {
        return 0.0;
    }

    let (lat_sum, lng_sum) = indices[start..end]
        .iter()
        .fold((0.0f64, 0.0f64), |(ls, ns), &idx| {
            (ls + latitudes[idx], ns + longitudes[idx])
        });
    let cm_lat = lat_sum / n as f64;
    let cm_lng = lng_sum / n as f64;

    let cm_lat_rad = cm_lat.to_radians();
    let cm_lng_rad = cm_lng.to_radians();
    let cos_cm_lat = cm_lat_rad.cos();

    let sum_sq: f64 = indices[start..end]
        .iter()
        .map(|&idx| {
            let lat_rad = latitudes[idx].to_radians();
            let dlat = lat_rad - cm_lat_rad;
            let dlng = longitudes[idx].to_radians() - cm_lng_rad;
            let a = (dlat / 2.0).sin().powi(2)
                + cos_cm_lat * lat_rad.cos() * (dlng / 2.0).sin().powi(2);
            let d = 2.0 * a.sqrt().asin() * 6371.0088;
            d * d
        })
        .sum();

    (sum_sq / n as f64).sqrt()
}

pub fn radius_of_gyration_batch_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ranges: &[(usize, usize)],
) -> Result<Vec<f64>, String> {
    Ok(radius_of_gyration_batch_with_counts_impl(latitudes, longitudes, ranges)?.0)
}

pub fn radius_of_gyration_batch_with_counts_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ranges: &[(usize, usize)],
) -> Result<(Vec<f64>, Vec<usize>), String> {
    validate_coord_ranges(latitudes, longitudes, ranges)?;

    let results: Vec<f64> = ranges
        .par_iter()
        .map(|&(start, end)| rog_for_parallel_slices(latitudes, longitudes, start, end))
        .collect();
    let counts = ranges.iter().map(|&(start, end)| end - start).collect();

    Ok((results, counts))
}

pub fn radius_of_gyration_from_ends_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ends: &[usize],
) -> Result<Vec<f64>, String> {
    Ok(radius_of_gyration_from_ends_with_counts_impl(latitudes, longitudes, ends)?.0)
}

pub fn radius_of_gyration_from_ends_with_counts_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ends: &[usize],
) -> Result<(Vec<f64>, Vec<usize>), String> {
    validate_coord_ends(latitudes, longitudes, ends)?;

    let results: Vec<f64> = (0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            rog_for_parallel_slices(latitudes, longitudes, start, ends[i])
        })
        .collect();
    let counts = (0..ends.len())
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            ends[i] - start
        })
        .collect();

    Ok((results, counts))
}

fn is_valid_indexed_row(
    latitudes: &[f64],
    longitudes: &[f64],
    valid_rows: Option<&[bool]>,
    idx: usize,
) -> bool {
    valid_rows.is_none_or(|rows| rows[idx])
        && latitudes[idx].is_finite()
        && longitudes[idx].is_finite()
}

pub fn radius_of_gyration_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
) -> Result<(Vec<f64>, Vec<usize>), String> {
    radius_of_gyration_indexed_with_valid_rows_impl(latitudes, longitudes, indices, ends, None)
}

pub fn radius_of_gyration_indexed_with_valid_rows_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
) -> Result<(Vec<f64>, Vec<usize>), String> {
    validate_indexed_coord_ends(latitudes, longitudes, indices, ends)?;
    if let Some(valid_rows) = valid_rows {
        if valid_rows.len() != latitudes.len() {
            return Err("valid_rows and coordinates must have the same length".to_string());
        }
    }

    let mut valid_indices = Vec::new();
    let mut valid_ranges = Vec::with_capacity(ends.len());
    let mut valid_counts = Vec::with_capacity(ends.len());
    for i in 0..ends.len() {
        let start = if i == 0 { 0 } else { ends[i - 1] };
        let end = ends[i];
        let valid_start = valid_indices.len();
        for &idx in &indices[start..end] {
            if is_valid_indexed_row(latitudes, longitudes, valid_rows, idx) {
                valid_indices.push(idx);
            }
        }
        let valid_end = valid_indices.len();
        valid_ranges.push((valid_start, valid_end));
        valid_counts.push(valid_end - valid_start);
    }

    let results: Vec<f64> = valid_ranges
        .par_iter()
        .map(|&(start, end)| {
            rog_for_indexed_slice(latitudes, longitudes, &valid_indices, start, end)
        })
        .collect();

    Ok((results, valid_counts))
}

pub fn split_user_index_ranges((indices, ranges): UserIndexRanges) -> (Vec<usize>, Vec<usize>) {
    (indices, ends_from_ranges(&ranges))
}

pub fn user_indices_for_u64_codes(
    codes: &[u64],
    num_groups: usize,
) -> Result<UserIndexRanges, String> {
    if codes.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    if num_groups == 0 {
        return Err(
            "num_groups must be greater than zero when uid codes are not empty".to_string(),
        );
    }

    let mut offsets = vec![0usize; num_groups];
    for &code in codes {
        let group_id =
            usize::try_from(code).map_err(|_| "uid code must fit into usize".to_string())?;
        let count = offsets
            .get_mut(group_id)
            .ok_or_else(|| "uid code must be less than num_groups".to_string())?;
        *count += 1;
    }

    let mut ranges = Vec::with_capacity(num_groups);
    let mut current_offset = 0usize;
    for count in offsets.iter_mut() {
        if *count == 0 {
            continue;
        }
        let start = current_offset;
        let end = current_offset + *count;
        ranges.push((start, end));
        *count = start;
        current_offset = end;
    }

    let mut indices = vec![0usize; codes.len()];
    for (row_idx, &code) in codes.iter().enumerate() {
        let group_id = code as usize;
        let pos = offsets[group_id];
        indices[pos] = row_idx;
        offsets[group_id] += 1;
    }

    Ok((indices, ranges))
}
