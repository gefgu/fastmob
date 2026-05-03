use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use rayon::prelude::*;

use crate::haversine::haversine_km;
use crate::utils::{
    arrow_values, as_f64_array, f64_results_into_arrow, ranges_from_starts_ends,
    validate_indexed_coord_ranges,
};

#[pyfunction]
pub(crate) fn max_distance_from_point_batch_km(
    home_lats: Vec<f64>,
    home_lngs: Vec<f64>,
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    if home_lats.len() != home_lngs.len() {
        return Err(PyValueError::new_err(
            "home_lats and home_lngs must have the same length",
        ));
    }
    if home_lats.len() != ranges.len() {
        return Err(PyValueError::new_err(
            "home coordinates and ranges must have the same length",
        ));
    }
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }

    let coords: Vec<(f64, f64)> = latitudes.into_iter().zip(longitudes).collect();

    let results: Vec<f64> = ranges
        .par_iter()
        .enumerate()
        .map(|(i, &(start, end))| {
            let slice = &coords[start..end];
            if slice.is_empty() {
                return 0.0;
            }
            let home_lat = home_lats[i];
            let home_lng = home_lngs[i];
            slice
                .iter()
                .map(|&(lat, lng)| haversine_km(home_lat, home_lng, lat, lng))
                .fold(0.0f64, f64::max)
        })
        .collect();

    Ok(results)
}

fn max_distance_from_point_indexed_impl(
    home_lats: &[f64],
    home_lngs: &[f64],
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> PyResult<Vec<f64>> {
    if home_lats.len() != home_lngs.len() || home_lats.len() != ranges.len() {
        return Err(PyValueError::new_err(
            "home coordinates and ranges must have the same length",
        ));
    }
    validate_indexed_coord_ranges(latitudes, longitudes, indices, ranges)?;

    Ok(ranges
        .par_iter()
        .enumerate()
        .map(|(i, &(start, end))| {
            let home_lat = home_lats[i];
            let home_lng = home_lngs[i];
            indices[start..end]
                .iter()
                .map(|&idx| haversine_km(home_lat, home_lng, latitudes[idx], longitudes[idx]))
                .fold(0.0f64, f64::max)
        })
        .collect())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn max_distance_from_point_indexed_numpy<'py>(
    py: Python<'py>,
    home_lats: PyReadonlyArray1<'py, f64>,
    home_lngs: PyReadonlyArray1<'py, f64>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    starts: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    Ok(max_distance_from_point_indexed_impl(
        home_lats.as_slice()?,
        home_lngs.as_slice()?,
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        indices.as_slice()?,
        &ranges,
    )?
    .into_pyarray(py))
}

#[pyfunction]
pub(crate) fn max_distance_from_point_indexed_arrow(
    home_lats: PyArray,
    home_lngs: PyArray,
    latitudes: PyArray,
    longitudes: PyArray,
    indices: PyReadonlyArray1<usize>,
    starts: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<PyArray> {
    let home_lats = as_f64_array(home_lats, "home_lats")?;
    let home_lngs = as_f64_array(home_lngs, "home_lngs")?;
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    Ok(f64_results_into_arrow(
        max_distance_from_point_indexed_impl(
            arrow_values(&home_lats),
            arrow_values(&home_lngs),
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            indices.as_slice()?,
            &ranges,
        )?,
    ))
}
