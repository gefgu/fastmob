use rayon::prelude::*;

const EARTH_RADIUS_KM: f64 = 6371.0088;

/// Centre of mass with the trigonometry the squared-distance pass needs
/// precomputed once per user.
#[derive(Clone, Copy, Debug)]
pub struct RogCenter {
    lat_rad: f64,
    lng_rad: f64,
    cos_lat: f64,
}

impl RogCenter {
    pub fn from_sums(lat_sum: f64, lng_sum: f64, n: usize) -> Self {
        let lat_rad = (lat_sum / n as f64).to_radians();
        let lng_rad = (lng_sum / n as f64).to_radians();
        Self {
            lat_rad,
            lng_rad,
            cos_lat: lat_rad.cos(),
        }
    }
}

#[inline]
pub fn sq_haversine_to_center(latitude: f64, longitude: f64, center: &RogCenter) -> f64 {
    let lat_rad = latitude.to_radians();
    let dlat = lat_rad - center.lat_rad;
    let dlng = longitude.to_radians() - center.lng_rad;
    let a =
        (dlat / 2.0).sin().powi(2) + center.cos_lat * lat_rad.cos() * (dlng / 2.0).sin().powi(2);
    let d = 2.0 * a.sqrt().asin() * EARTH_RADIUS_KM;
    d * d
}

/// `(lat_sum, lng_sum)` over the contiguous rows `start..end`.
pub fn coordinate_sums_presorted(
    latitudes: &[f64],
    longitudes: &[f64],
    start: usize,
    end: usize,
) -> (f64, f64) {
    (start..end).fold((0.0f64, 0.0f64), |(ls, ns), idx| {
        (ls + latitudes[idx], ns + longitudes[idx])
    })
}

/// `(lat_sum, lng_sum)` over the rows named by `indices`.
pub fn coordinate_sums_indexed(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
) -> (f64, f64) {
    indices.iter().fold((0.0f64, 0.0f64), |(ls, ns), &idx| {
        (ls + latitudes[idx], ns + longitudes[idx])
    })
}

/// Sum of squared haversine distances (km²) from rows `start..end` to `center`.
pub fn sq_distance_sum_presorted(
    latitudes: &[f64],
    longitudes: &[f64],
    start: usize,
    end: usize,
    center: &RogCenter,
) -> f64 {
    (start..end)
        .map(|idx| sq_haversine_to_center(latitudes[idx], longitudes[idx], center))
        .sum()
}

/// Sum of squared haversine distances (km²) from the rows named by `indices` to `center`.
pub fn sq_distance_sum_indexed(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    center: &RogCenter,
) -> f64 {
    indices
        .iter()
        .map(|&idx| sq_haversine_to_center(latitudes[idx], longitudes[idx], center))
        .sum()
}

pub fn rog_for_presorted_slices(
    latitudes: &[f64],
    longitudes: &[f64],
    start: usize,
    end: usize,
) -> f64 {
    let n = end - start;
    if n == 0 {
        return 0.0;
    }
    let (lat_sum, lng_sum) = coordinate_sums_presorted(latitudes, longitudes, start, end);
    let center = RogCenter::from_sums(lat_sum, lng_sum, n);
    let sum_sq = sq_distance_sum_presorted(latitudes, longitudes, start, end, &center);
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
    let rows = &indices[start..end];
    let (lat_sum, lng_sum) = coordinate_sums_indexed(latitudes, longitudes, rows);
    let center = RogCenter::from_sums(lat_sum, lng_sum, n);
    let sum_sq = sq_distance_sum_indexed(latitudes, longitudes, rows, &center);
    (sum_sq / n as f64).sqrt()
}

pub fn radius_of_gyration_presorted_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ends: &[usize],
) -> (Vec<f64>, Vec<bool>) {
    let results: Vec<f64> = (0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            rog_for_presorted_slices(latitudes, longitudes, start, ends[i])
        })
        .collect();
    let validity = (0..ends.len())
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            ends[i] > start
        })
        .collect();

    (results, validity)
}

pub fn is_valid_indexed_row(
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
    valid_rows: Option<&[bool]>,
) -> (Vec<f64>, Vec<bool>) {
    let mut valid_indices = Vec::new();
    let mut valid_ranges = Vec::with_capacity(ends.len());
    let mut validity = Vec::with_capacity(ends.len());
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
        validity.push(valid_end > valid_start);
    }

    let results: Vec<f64> = valid_ranges
        .par_iter()
        .map(|&(start, end)| {
            rog_for_indexed_slice(latitudes, longitudes, &valid_indices, start, end)
        })
        .collect();

    (results, validity)
}
