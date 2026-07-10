use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use fkmob_core::measures::individual::total_distance::{
    total_distance_from_ends_impl, total_distance_impl, total_distance_indexed_impl,
};

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, f64_results_into_arrow,
};

#[pyfunction]
pub fn total_distance_batch_km(
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    total_distance_impl(&latitudes, &longitudes, &ranges).map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn total_distance_numpy(
    latitudes: PyReadonlyArray1<f64>,
    longitudes: PyReadonlyArray1<f64>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<Vec<f64>> {
    total_distance_from_ends_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        ends.as_slice()?,
    )
    .map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn total_distance_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(total_distance_indexed_impl(
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
pub fn total_distance_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<Vec<f64>> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    total_distance_from_ends_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        ends.as_slice()?,
    )
    .map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn total_distance_indexed_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    indices: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<PyArray> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    Ok(f64_results_into_arrow(
        total_distance_indexed_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            indices.as_slice()?,
            ends.as_slice()?,
            valid_rows.as_deref(),
        )
        .map_err(PyValueError::new_err)?,
    ))
}
