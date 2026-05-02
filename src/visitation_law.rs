use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use rayon::prelude::*;

use crate::haversine::haversine_km;
use crate::utils::{arrow_values, as_f64_array};

fn visitation_distances_impl(
    home_latitudes: &[f64],
    home_longitudes: &[f64],
    location_latitudes: &[f64],
    location_longitudes: &[f64],
) -> PyResult<Vec<f64>> {
    if home_latitudes.len() != home_longitudes.len() {
        return Err(PyValueError::new_err(
            "home_latitudes and home_longitudes must have the same length",
        ));
    }
    if location_latitudes.len() != location_longitudes.len() {
        return Err(PyValueError::new_err(
            "location_latitudes and location_longitudes must have the same length",
        ));
    }
    if home_latitudes.len() != location_latitudes.len() {
        return Err(PyValueError::new_err(
            "home and location coordinate arrays must have the same length",
        ));
    }

    Ok((0..home_latitudes.len())
        .into_par_iter()
        .map(|idx| {
            haversine_km(
                home_latitudes[idx],
                home_longitudes[idx],
                location_latitudes[idx],
                location_longitudes[idx],
            )
        })
        .collect())
}

#[pyfunction]
pub(crate) fn visitation_distances_km(
    home_latitudes: Vec<f64>,
    home_longitudes: Vec<f64>,
    location_latitudes: Vec<f64>,
    location_longitudes: Vec<f64>,
) -> PyResult<Vec<f64>> {
    visitation_distances_impl(
        &home_latitudes,
        &home_longitudes,
        &location_latitudes,
        &location_longitudes,
    )
}

#[pyfunction]
pub(crate) fn visitation_distances_numpy(
    home_latitudes: PyReadonlyArray1<f64>,
    home_longitudes: PyReadonlyArray1<f64>,
    location_latitudes: PyReadonlyArray1<f64>,
    location_longitudes: PyReadonlyArray1<f64>,
) -> PyResult<Vec<f64>> {
    visitation_distances_impl(
        home_latitudes.as_slice()?,
        home_longitudes.as_slice()?,
        location_latitudes.as_slice()?,
        location_longitudes.as_slice()?,
    )
}

#[pyfunction]
pub(crate) fn visitation_distances_arrow(
    home_latitudes: PyArray,
    home_longitudes: PyArray,
    location_latitudes: PyArray,
    location_longitudes: PyArray,
) -> PyResult<Vec<f64>> {
    let home_latitudes = as_f64_array(home_latitudes, "home_latitudes")?;
    let home_longitudes = as_f64_array(home_longitudes, "home_longitudes")?;
    let location_latitudes = as_f64_array(location_latitudes, "location_latitudes")?;
    let location_longitudes = as_f64_array(location_longitudes, "location_longitudes")?;
    visitation_distances_impl(
        arrow_values(&home_latitudes),
        arrow_values(&home_longitudes),
        arrow_values(&location_latitudes),
        arrow_values(&location_longitudes),
    )
}
