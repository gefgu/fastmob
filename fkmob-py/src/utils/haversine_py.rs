use pyo3::prelude::*;
use fkmob_core::utils::haversine::haversine_km as core_haversine_km;

#[pyfunction]
pub fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    core_haversine_km(lat1, lon1, lat2, lon2)
}
