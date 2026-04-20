use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::haversine::haversine_km;

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
