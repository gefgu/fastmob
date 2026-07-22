use fastmob_core::measures::individual::max_distance_from_point::{
    max_distance_from_point_from_ends_impl, max_distance_from_point_impl,
    max_distance_from_point_indexed_impl,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, f64_results_into_arrow,
};

#[pyfunction]
pub fn max_distance_from_point_batch_km(
    home_lats: Vec<f64>,
    home_lngs: Vec<f64>,
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    max_distance_from_point_impl(&home_lats, &home_lngs, &latitudes, &longitudes, &ranges)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn max_distance_from_point_numpy<'py>(
    py: Python<'py>,
    home_lats: PyReadonlyArray1<'py, f64>,
    home_lngs: PyReadonlyArray1<'py, f64>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(max_distance_from_point_from_ends_impl(
        home_lats.as_slice()?,
        home_lngs.as_slice()?,
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        ends.as_slice()?,
        None,
    )
    .map_err(PyValueError::new_err)?
    .into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn max_distance_from_point_indexed_numpy<'py>(
    py: Python<'py>,
    home_lats: PyReadonlyArray1<'py, f64>,
    home_lngs: PyReadonlyArray1<'py, f64>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(max_distance_from_point_indexed_impl(
        home_lats.as_slice()?,
        home_lngs.as_slice()?,
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        indices.as_slice()?,
        ends.as_slice()?,
        None,
    )
    .map_err(PyValueError::new_err)?
    .into_pyarray(py))
}

#[pyfunction]
pub fn max_distance_from_point_arrow(
    home_lats: PyArray,
    home_lngs: PyArray,
    latitudes: PyArray,
    longitudes: PyArray,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<PyArray> {
    let home_lats = as_f64_array(home_lats, "home_lats")?;
    let home_lngs = as_f64_array(home_lngs, "home_lngs")?;
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    Ok(f64_results_into_arrow(
        max_distance_from_point_from_ends_impl(
            arrow_values(&home_lats),
            arrow_values(&home_lngs),
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            ends.as_slice()?,
            valid_rows.as_deref(),
        )
        .map_err(PyValueError::new_err)?,
    ))
}

#[pyfunction]
pub fn max_distance_from_point_indexed_arrow(
    home_lats: PyArray,
    home_lngs: PyArray,
    latitudes: PyArray,
    longitudes: PyArray,
    indices: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<PyArray> {
    let home_lats = as_f64_array(home_lats, "home_lats")?;
    let home_lngs = as_f64_array(home_lngs, "home_lngs")?;
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    Ok(f64_results_into_arrow(
        max_distance_from_point_indexed_impl(
            arrow_values(&home_lats),
            arrow_values(&home_lngs),
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            indices.as_slice()?,
            ends.as_slice()?,
            valid_rows.as_deref(),
        )
        .map_err(PyValueError::new_err)?,
    ))
}
