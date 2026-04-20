use geo::{Distance, Haversine, Point};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub(crate) fn radius_of_gyration_km(coords: Vec<(f64, f64)>) -> PyResult<f64> {
    Ok(rog_for_slice(&coords))
}

pub(crate) fn rog_for_slice(coords: &[(f64, f64)]) -> f64 {
    let n = coords.len();
    if n == 0 {
        return 0.0;
    }

    let (lat_sum, lng_sum) = coords.iter().fold((0.0f64, 0.0f64), |(ls, ns), &(lat, lng)| {
        (ls + lat, ns + lng)
    });
    let cm_lat = lat_sum / n as f64;
    let cm_lng = lng_sum / n as f64;
    let cm = Point::new(cm_lng, cm_lat);

    let sum_sq: f64 = coords
        .iter()
        .map(|&(lat, lng)| {
            let p = Point::new(lng, lat);
            let d = Haversine.distance(p, cm) / 1000.0;
            d * d
        })
        .sum();

    (sum_sq / n as f64).sqrt()
}

#[pyfunction]
pub(crate) fn radius_of_gyration_batch_km(
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }

    let coords: Vec<(f64, f64)> = latitudes.into_iter().zip(longitudes).collect();

    let results: Vec<f64> = ranges
        .par_iter()
        .map(|&(start, end)| rog_for_slice(&coords[start..end]))
        .collect();

    Ok(results)
}
