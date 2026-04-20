use pyo3::prelude::*;

use crate::haversine::haversine_km;

#[pyfunction]
pub(crate) fn square_displacement_km2(lat0: f64, lng0: f64, lat_t: f64, lng_t: f64) -> f64 {
    let d = haversine_km(lat0, lng0, lat_t, lng_t);
    d * d
}
