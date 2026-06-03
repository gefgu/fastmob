use rayon::prelude::*;

use crate::utils::{
    ranges_from_sorted_values, split_ranges, validate_coord_ranges, validate_indexed_coord_ranges,
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

    let mut valid_latitudes = Vec::new();
    let mut valid_longitudes = Vec::new();
    let mut valid_ranges = Vec::with_capacity(ranges.len());
    let mut valid_counts = Vec::with_capacity(ranges.len());
    for &(start, end) in ranges {
        let valid_start = valid_latitudes.len();
        for idx in start..end {
            let lat = latitudes[idx];
            let lng = longitudes[idx];
            if !lat.is_nan() && !lng.is_nan() {
                valid_latitudes.push(lat);
                valid_longitudes.push(lng);
            }
        }
        let valid_end = valid_latitudes.len();
        valid_ranges.push((valid_start, valid_end));
        valid_counts.push(valid_end - valid_start);
    }

    let results: Vec<f64> = valid_ranges
        .par_iter()
        .map(|&(start, end)| {
            rog_for_parallel_slices(&valid_latitudes, &valid_longitudes, start, end)
        })
        .collect();

    Ok((results, valid_counts))
}

pub fn radius_of_gyration_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> Result<Vec<f64>, String> {
    validate_indexed_coord_ranges(latitudes, longitudes, indices, ranges)?;

    let mut valid_indices = Vec::new();
    let mut valid_ranges = Vec::with_capacity(ranges.len());
    for &(start, end) in ranges {
        let valid_start = valid_indices.len();
        for &idx in &indices[start..end] {
            if !latitudes[idx].is_nan() && !longitudes[idx].is_nan() {
                valid_indices.push(idx);
            }
        }
        valid_ranges.push((valid_start, valid_indices.len()));
    }

    let results: Vec<f64> = valid_ranges
        .par_iter()
        .map(|&(start, end)| {
            rog_for_indexed_slice(latitudes, longitudes, &valid_indices, start, end)
        })
        .collect();

    Ok(results)
}

pub fn split_user_index_ranges(
    (indices, ranges): UserIndexRanges,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let (starts, ends) = split_ranges(ranges);
    (indices, starts, ends)
}

pub fn user_indices_for_ord_values<T: Ord>(values: &[T]) -> UserIndexRanges {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&left, &right| values[left].cmp(&values[right]).then(left.cmp(&right)));
    let ranges = ranges_from_sorted_values(values, &indices);
    (indices, ranges)
}

pub fn user_indices_for_ord_values_at_indices<T: Ord>(
    values: &[T],
    mut indices: Vec<usize>,
) -> UserIndexRanges {
    indices.sort_by(|&left, &right| values[left].cmp(&values[right]).then(left.cmp(&right)));
    let ranges = ranges_from_sorted_values(values, &indices);
    (indices, ranges)
}

pub fn user_indices_for_f64_values(values: &[f64]) -> UserIndexRanges {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&left, &right| {
        values[left]
            .total_cmp(&values[right])
            .then(left.cmp(&right))
    });
    let ranges = ranges_from_sorted_values(values, &indices);
    (indices, ranges)
}

pub fn user_indices_for_f64_values_at_indices(
    values: &[f64],
    mut indices: Vec<usize>,
) -> UserIndexRanges {
    indices.sort_by(|&left, &right| {
        values[left]
            .total_cmp(&values[right])
            .then(left.cmp(&right))
    });
    let ranges = ranges_from_sorted_values(values, &indices);
    (indices, ranges)
}

pub fn valid_coord_indices(
    latitudes: &[f64],
    longitudes: &[f64],
) -> Result<Vec<usize>, String> {
    if latitudes.len() != longitudes.len() {
        return Err("latitudes and longitudes must have the same length".to_string());
    }

    Ok((0..latitudes.len())
        .filter(|&idx| !latitudes[idx].is_nan() && !longitudes[idx].is_nan())
        .collect())
}
