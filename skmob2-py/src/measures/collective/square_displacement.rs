use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use skmob2_core::measures::collective::square_displacement::{
    mean_square_displacement_indexed_impl, square_displacement_km2 as core_square_displacement_km2,
};

use crate::utils::{arrow_valid_rows, arrow_values, as_nullable_f64_array};

#[pyfunction]
pub fn square_displacement_km2(lat0: f64, lng0: f64, lat_t: f64, lng_t: f64) -> f64 {
    core_square_displacement_km2(lat0, lng0, lat_t, lng_t)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn mean_square_displacement_indexed_numpy<'py>(
    _py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    timestamps_s: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    delta_s: f64,
) -> PyResult<f64> {
    mean_square_displacement_indexed_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        timestamps_s.as_slice()?,
        indices.as_slice()?,
        ends.as_slice()?,
        delta_s,
        None,
    )
    .map_err(PyValueError::new_err)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn mean_square_displacement_indexed_arrow<'py>(
    _py: Python<'py>,
    latitudes: PyArray,
    longitudes: PyArray,
    timestamps_s: PyArray,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    delta_s: f64,
) -> PyResult<f64> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let timestamps_s = as_nullable_f64_array(timestamps_s, "timestamps_s")?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes, &timestamps_s]);
    mean_square_displacement_indexed_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        arrow_values(&timestamps_s),
        indices.as_slice()?,
        ends.as_slice()?,
        delta_s,
        valid_rows.as_deref(),
    )
    .map_err(PyValueError::new_err)
}
