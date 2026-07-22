use fastmob_core::preprocessing::compress_traj::{
    compress_trajectory_representatives_impl, compress_trajectory_representatives_indexed_impl,
    compress_user_slice,
};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, f64_results_into_arrow,
    u64_results_into_arrow, validate_indexed_ends,
};

type CompressRepresentativesNumpy<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);
type CompressRepresentativesArrow = (ArrowPyArray, ArrowPyArray, ArrowPyArray);

#[pyfunction]
pub fn compress_trajectory_batch(
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
pub fn compress_trajectory_representatives_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    ranges: Vec<(usize, usize)>,
    spatial_radius_km: f64,
) -> PyResult<CompressRepresentativesNumpy<'py>> {
    let lats = latitudes.as_slice()?;
    let lngs = longitudes.as_slice()?;
    let (representative_indices, median_latitudes, median_longitudes) = py.detach(|| {
        compress_trajectory_representatives_impl(lats, lngs, &ranges, spatial_radius_km)
    });
    Ok((
        PyArray1::from_vec(py, representative_indices),
        PyArray1::from_vec(py, median_latitudes),
        PyArray1::from_vec(py, median_longitudes),
    ))
}

#[pyfunction]
pub fn compress_trajectory_representatives_arrow(
    py: Python<'_>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    ranges: Vec<(usize, usize)>,
    spatial_radius_km: f64,
) -> PyResult<CompressRepresentativesArrow> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    let (representative_indices, median_latitudes, median_longitudes) = py.detach(|| {
        compress_trajectory_representatives_impl(lats, lngs, &ranges, spatial_radius_km)
    });
    Ok((
        u64_results_into_arrow(
            representative_indices
                .into_iter()
                .map(|i| i as u64)
                .collect(),
        ),
        f64_results_into_arrow(median_latitudes),
        f64_results_into_arrow(median_longitudes),
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn compress_trajectory_representatives_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    sorted_indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    spatial_radius_km: f64,
) -> PyResult<CompressRepresentativesNumpy<'py>> {
    let lats = latitudes.as_slice()?;
    let lngs = longitudes.as_slice()?;
    let idxs = sorted_indices.as_slice()?;
    let ends = ends.as_slice()?;
    validate_indexed_ends(lats.len(), idxs, ends)?;
    let (representative_indices, median_latitudes, median_longitudes) = py.detach(|| {
        compress_trajectory_representatives_indexed_impl(
            lats,
            lngs,
            idxs,
            ends,
            None,
            spatial_radius_km,
        )
    });
    Ok((
        PyArray1::from_vec(py, representative_indices),
        PyArray1::from_vec(py, median_latitudes),
        PyArray1::from_vec(py, median_longitudes),
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn compress_trajectory_representatives_indexed_arrow(
    py: Python<'_>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    sorted_indices: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
    spatial_radius_km: f64,
) -> PyResult<CompressRepresentativesArrow> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    let idxs = sorted_indices.as_slice()?;
    let ends = ends.as_slice()?;
    validate_indexed_ends(lats.len(), idxs, ends)?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    let (representative_indices, median_latitudes, median_longitudes) = py.detach(|| {
        compress_trajectory_representatives_indexed_impl(
            lats,
            lngs,
            idxs,
            ends,
            valid_rows.as_deref(),
            spatial_radius_km,
        )
    });
    Ok((
        u64_results_into_arrow(
            representative_indices
                .into_iter()
                .map(|i| i as u64)
                .collect(),
        ),
        f64_results_into_arrow(median_latitudes),
        f64_results_into_arrow(median_longitudes),
    ))
}
