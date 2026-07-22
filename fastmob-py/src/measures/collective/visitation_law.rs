use fastmob_core::measures::collective::visitation_law::visitation_distances_km as core_visitation_distances_km;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{arrow_values, as_f64_array};

fn is_arrow_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.hasattr("__arrow_c_array__")
}

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
pub fn visitation_distances<'py>(
    home_latitudes: &Bound<'py, PyAny>,
    home_longitudes: &Bound<'py, PyAny>,
    location_latitudes: &Bound<'py, PyAny>,
    location_longitudes: &Bound<'py, PyAny>,
) -> PyResult<Vec<f64>> {
    if let (
        Ok(home_latitudes),
        Ok(home_longitudes),
        Ok(location_latitudes),
        Ok(location_longitudes),
    ) = (
        home_latitudes.extract::<PyReadonlyArray1<f64>>(),
        home_longitudes.extract::<PyReadonlyArray1<f64>>(),
        location_latitudes.extract::<PyReadonlyArray1<f64>>(),
        location_longitudes.extract::<PyReadonlyArray1<f64>>(),
    ) {
        return core_visitation_distances_km(
            home_latitudes.as_slice()?.to_vec(),
            home_longitudes.as_slice()?.to_vec(),
            location_latitudes.as_slice()?.to_vec(),
            location_longitudes.as_slice()?.to_vec(),
        )
        .map_err(PyValueError::new_err);
    }

    if is_arrow_array(home_latitudes)?
        && is_arrow_array(home_longitudes)?
        && is_arrow_array(location_latitudes)?
        && is_arrow_array(location_longitudes)?
    {
        let home_latitudes = as_f64_array(home_latitudes.extract::<PyArray>()?, "home_latitudes")?;
        let home_longitudes =
            as_f64_array(home_longitudes.extract::<PyArray>()?, "home_longitudes")?;
        let location_latitudes = as_f64_array(
            location_latitudes.extract::<PyArray>()?,
            "location_latitudes",
        )?;
        let location_longitudes = as_f64_array(
            location_longitudes.extract::<PyArray>()?,
            "location_longitudes",
        )?;
        return core_visitation_distances_km(
            arrow_values(&home_latitudes).to_vec(),
            arrow_values(&home_longitudes).to_vec(),
            arrow_values(&location_latitudes).to_vec(),
            arrow_values(&location_longitudes).to_vec(),
        )
        .map_err(PyValueError::new_err);
    }

    Err(PyTypeError::new_err(
        "coordinate arrays must all be NumPy arrays or all be Arrow arrays",
    ))
}
