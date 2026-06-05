use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use skmob2_core::measures::collective::visitation_law::visitation_distances_km as core_visitation_distances_km;

use crate::utils::{arrow_values, as_f64_array};

#[pyfunction]
pub fn visitation_distances_km(
    home_latitudes: Vec<f64>,
    home_longitudes: Vec<f64>,
    location_latitudes: Vec<f64>,
    location_longitudes: Vec<f64>,
) -> PyResult<Vec<f64>> {
    core_visitation_distances_km(
        home_latitudes,
        home_longitudes,
        location_latitudes,
        location_longitudes,
    )
    .map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn visitation_distances_numpy(
    home_latitudes: PyReadonlyArray1<f64>,
    home_longitudes: PyReadonlyArray1<f64>,
    location_latitudes: PyReadonlyArray1<f64>,
    location_longitudes: PyReadonlyArray1<f64>,
) -> PyResult<Vec<f64>> {
    core_visitation_distances_km(
        home_latitudes.as_slice()?.to_vec(),
        home_longitudes.as_slice()?.to_vec(),
        location_latitudes.as_slice()?.to_vec(),
        location_longitudes.as_slice()?.to_vec(),
    )
    .map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn visitation_distances_arrow(
    home_latitudes: PyArray,
    home_longitudes: PyArray,
    location_latitudes: PyArray,
    location_longitudes: PyArray,
) -> PyResult<Vec<f64>> {
    let home_latitudes = as_f64_array(home_latitudes, "home_latitudes")?;
    let home_longitudes = as_f64_array(home_longitudes, "home_longitudes")?;
    let location_latitudes = as_f64_array(location_latitudes, "location_latitudes")?;
    let location_longitudes = as_f64_array(location_longitudes, "location_longitudes")?;
    core_visitation_distances_km(
        arrow_values(&home_latitudes).to_vec(),
        arrow_values(&home_longitudes).to_vec(),
        arrow_values(&location_latitudes).to_vec(),
        arrow_values(&location_longitudes).to_vec(),
    )
    .map_err(PyValueError::new_err)
}
