use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::haversine::haversine_km;
use crate::utils::median_slice_in_place;

type CompressRepresentatives = (Vec<usize>, Vec<f64>, Vec<f64>);
type CompressRepresentativesNumpy<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);
type CompressRepresentativeRow = (usize, f64, f64);

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
    Ok(compress_trajectory_representatives_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        &ranges,
        spatial_radius_km,
    ))
}

#[pyfunction]
pub(crate) fn compress_trajectory_representatives_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    ranges: Vec<(usize, usize)>,
    spatial_radius_km: f64,
) -> PyResult<CompressRepresentativesNumpy<'py>> {
    let (representative_indices, median_latitudes, median_longitudes) =
        compress_trajectory_representatives_impl(
            latitudes.as_slice()?,
            longitudes.as_slice()?,
            &ranges,
            spatial_radius_km,
        );
    Ok((
        PyArray1::from_vec(py, representative_indices),
        PyArray1::from_vec(py, median_latitudes),
        PyArray1::from_vec(py, median_longitudes),
    ))
}

fn compress_trajectory_representatives_impl(
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
