use rayon::prelude::*;

use crate::haversine::haversine_km;
use crate::utils::median_slice_in_place;

pub(super) type CompressRepresentatives = (Vec<usize>, Vec<f64>, Vec<f64>);
type CompressRepresentativeRow = (usize, f64, f64);

pub(super) fn compress_user_slice(
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
            groups.push((offset + group_start, offset + i + 1));
            lat_0 = lat;
            lon_0 = lon;
            group_start = i + 1;
        }
    }
    groups.push((offset + group_start, offset + n));

    groups
}

fn compress_user_representatives(
    lats: &[f64],
    lngs: &[f64],
    spatial_radius_km: f64,
    offset: usize,
) -> Vec<CompressRepresentativeRow> {
    let groups = compress_user_slice(lats, lngs, spatial_radius_km, offset);
    let mut rows = Vec::with_capacity(groups.len());
    let mut lat_buf = Vec::new();
    let mut lng_buf = Vec::new();

    for (group_start, group_end) in groups {
        lat_buf.clear();
        lng_buf.clear();
        lat_buf.extend_from_slice(&lats[group_start - offset..group_end - offset]);
        lng_buf.extend_from_slice(&lngs[group_start - offset..group_end - offset]);
        rows.push((
            group_start,
            median_slice_in_place(&mut lat_buf),
            median_slice_in_place(&mut lng_buf),
        ));
    }

    rows
}

fn compress_user_representatives_indexed(
    lats: &[f64],
    lngs: &[f64],
    user_indices: &[usize],
    spatial_radius_km: f64,
) -> Vec<CompressRepresentativeRow> {
    let n = user_indices.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        let orig = user_indices[0];
        return vec![(orig, lats[orig], lngs[orig])];
    }

    let mut groups: Vec<(usize, usize)> = Vec::new();
    let mut lat_0 = lats[user_indices[0]];
    let mut lon_0 = lngs[user_indices[0]];
    let mut group_start = 0usize;

    for i in 0..(n - 1) {
        let lat = lats[user_indices[i + 1]];
        let lon = lngs[user_indices[i + 1]];
        let dist = haversine_km(lat_0, lon_0, lat, lon);

        if dist > spatial_radius_km {
            groups.push((group_start, i + 1));
            lat_0 = lat;
            lon_0 = lon;
            group_start = i + 1;
        }
    }
    groups.push((group_start, n));

    let mut rows = Vec::with_capacity(groups.len());
    let mut lat_buf = Vec::new();
    let mut lng_buf = Vec::new();

    for (group_start_local, group_end_local) in groups {
        lat_buf.clear();
        lng_buf.clear();
        for &orig in &user_indices[group_start_local..group_end_local] {
            lat_buf.push(lats[orig]);
            lng_buf.push(lngs[orig]);
        }
        rows.push((
            user_indices[group_start_local],
            median_slice_in_place(&mut lat_buf),
            median_slice_in_place(&mut lng_buf),
        ));
    }

    rows
}

pub(super) fn compress_trajectory_representatives_impl(
    lats: &[f64],
    lngs: &[f64],
    ranges: &[(usize, usize)],
    spatial_radius_km: f64,
) -> CompressRepresentatives {
    let per_user_rows: Vec<Vec<CompressRepresentativeRow>> = ranges
        .par_iter()
        .map(|&(start, end)| {
            compress_user_representatives(
                &lats[start..end],
                &lngs[start..end],
                spatial_radius_km,
                start,
            )
        })
        .collect();

    let total_groups: usize = per_user_rows.iter().map(|g| g.len()).sum();
    let mut representative_indices = Vec::with_capacity(total_groups);
    let mut median_latitudes = Vec::with_capacity(total_groups);
    let mut median_longitudes = Vec::with_capacity(total_groups);

    for user_rows in per_user_rows {
        for (representative_idx, median_lat, median_lng) in user_rows {
            representative_indices.push(representative_idx);
            median_latitudes.push(median_lat);
            median_longitudes.push(median_lng);
        }
    }

    (representative_indices, median_latitudes, median_longitudes)
}

pub(super) fn compress_trajectory_representatives_indexed_impl(
    lats: &[f64],
    lngs: &[f64],
    sorted_indices: &[usize],
    ranges: &[(usize, usize)],
    spatial_radius_km: f64,
) -> CompressRepresentatives {
    let per_user_rows: Vec<Vec<CompressRepresentativeRow>> = ranges
        .par_iter()
        .map(|&(start, end)| {
            compress_user_representatives_indexed(
                lats,
                lngs,
                &sorted_indices[start..end],
                spatial_radius_km,
            )
        })
        .collect();

    let total_groups: usize = per_user_rows.iter().map(|g| g.len()).sum();
    let mut representative_indices = Vec::with_capacity(total_groups);
    let mut median_latitudes = Vec::with_capacity(total_groups);
    let mut median_longitudes = Vec::with_capacity(total_groups);

    for user_rows in per_user_rows {
        for (representative_idx, median_lat, median_lng) in user_rows {
            representative_indices.push(representative_idx);
            median_latitudes.push(median_lat);
            median_longitudes.push(median_lng);
        }
    }

    (representative_indices, median_latitudes, median_longitudes)
}
