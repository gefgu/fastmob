use pyo3::prelude::*;

use crate::haversine::haversine_km;

type CompressRepresentatives = (Vec<usize>, Vec<f64>, Vec<f64>);

fn compress_user_slice(
    lats: &[f64],
    lngs: &[f64],
    spatial_radius_km: f64,
    offset: usize,
) -> Vec<(usize, usize)> {
    let n = lats.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![(offset, offset + 1)];
    }

    let mut groups: Vec<(usize, usize)> = Vec::new();
    let mut lat_0 = lats[0];
    let mut lon_0 = lngs[0];
    let mut group_start = 0usize;

    for i in 0..(n - 1) {
        let lat = lats[i + 1];
        let lon = lngs[i + 1];
        let dist = haversine_km(lat_0, lon_0, lat, lon);

        if dist > spatial_radius_km {
            // Emit current group [group_start..=i]
            groups.push((offset + group_start, offset + i + 1));
            lat_0 = lat;
            lon_0 = lon;
            group_start = i + 1;
        }
    }
    // Always emit the last group
    groups.push((offset + group_start, offset + n));

    groups
}

#[pyfunction]
pub(crate) fn compress_trajectory_batch(
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    ranges: Vec<(usize, usize)>,
    spatial_radius_km: f64,
) -> PyResult<Vec<(usize, usize)>> {
    let mut all_groups: Vec<(usize, usize)> = Vec::new();

    for &(start, end) in &ranges {
        let user_groups = compress_user_slice(
            &latitudes[start..end],
            &longitudes[start..end],
            spatial_radius_km,
            start,
        );
        all_groups.extend(user_groups);
    }

    Ok(all_groups)
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    if values.iter().any(|value| value.is_nan()) {
        return f64::NAN;
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

#[pyfunction]
pub(crate) fn compress_trajectory_representatives(
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    ranges: Vec<(usize, usize)>,
    spatial_radius_km: f64,
) -> PyResult<CompressRepresentatives> {
    let mut representative_indices = Vec::new();
    let mut median_latitudes = Vec::new();
    let mut median_longitudes = Vec::new();

    for &(start, end) in &ranges {
        let user_groups = compress_user_slice(
            &latitudes[start..end],
            &longitudes[start..end],
            spatial_radius_km,
            start,
        );
        for (group_start, group_end) in user_groups {
            representative_indices.push(group_start);
            median_latitudes.push(median(&latitudes[group_start..group_end]));
            median_longitudes.push(median(&longitudes[group_start..group_end]));
        }
    }

    Ok((representative_indices, median_latitudes, median_longitudes))
}
