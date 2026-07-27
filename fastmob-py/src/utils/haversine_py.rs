use fastmob_core::utils::haversine::{
    haversine_km as core_haversine_km, haversine_m_batch as core_haversine_m_batch,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    core_haversine_km(lat1, lon1, lat2, lon2)
}

/// Elementwise Haversine distance (metres) between two same-length arrays of
/// points, computed in parallel. See
/// [`fastmob_core::utils::haversine::haversine_m_batch`].
#[pyfunction]
pub fn haversine_m_batch<'py>(
    py: Python<'py>,
    lat1: PyReadonlyArray1<'py, f64>,
    lng1: PyReadonlyArray1<'py, f64>,
    lat2: PyReadonlyArray1<'py, f64>,
    lng2: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let lat1 = lat1.as_slice()?;
    let lng1 = lng1.as_slice()?;
    let lat2 = lat2.as_slice()?;
    let lng2 = lng2.as_slice()?;
    let result = py
        .detach(|| core_haversine_m_batch(lat1, lng1, lat2, lng2))
        .map_err(PyValueError::new_err)?;
    Ok(result.into_pyarray(py))
}
