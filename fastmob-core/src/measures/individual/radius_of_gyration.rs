use rayon::prelude::*;

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
