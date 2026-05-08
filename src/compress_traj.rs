use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::haversine::haversine_km;
use crate::utils::median_slice_in_place;

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

#[pyfunction]
pub(crate) fn compress_trajectory_representatives<'py>(
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    ranges: Vec<(usize, usize)>,
    spatial_radius_km: f64,
) -> PyResult<CompressRepresentatives> {
    let lats = latitudes.as_slice()?;
    let lngs = longitudes.as_slice()?;

    // Parallel group detection across users (read-only access to lats/lngs).
    let all_groups: Vec<Vec<(usize, usize)>> = ranges
        .par_iter()
        .map(|&(start, end)| {
            compress_user_slice(&lats[start..end], &lngs[start..end], spatial_radius_km, start)
        })
        .collect();

    let total_groups: usize = all_groups.iter().map(|g| g.len()).sum();
    let mut representative_indices = Vec::with_capacity(total_groups);
    let mut median_latitudes = Vec::with_capacity(total_groups);
    let mut median_longitudes = Vec::with_capacity(total_groups);

    for user_groups in all_groups {
        for (group_start, group_end) in user_groups {
            representative_indices.push(group_start);
            // Clone only the group slice (typically very small) for in-place quickselect.
            // This replaces the old O(n log n) clone-and-sort with O(n) quickselect on a
            // group-sized buffer — the dominant win for large groups.
            let mut lat_grp = lats[group_start..group_end].to_vec();
            let mut lng_grp = lngs[group_start..group_end].to_vec();
            median_latitudes.push(median_slice_in_place(&mut lat_grp));
            median_longitudes.push(median_slice_in_place(&mut lng_grp));
        }
    }

    Ok((representative_indices, median_latitudes, median_longitudes))
}
