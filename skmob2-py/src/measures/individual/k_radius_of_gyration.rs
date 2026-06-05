use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use skmob2_core::measures::individual::k_radius_of_gyration::{
    k_radius_of_gyration_impl, k_radius_of_gyration_indexed_impl,
    k_radius_of_gyration_km as core_k_rog_km,
};

use crate::utils::{arrow_values, as_f64_array, f64_results_into_arrow};

#[pyfunction]
pub fn k_radius_of_gyration_km(
    coords: Vec<(f64, f64)>,
    visit_counts: Vec<u64>,
    k: usize,
) -> PyResult<f64> {
    core_k_rog_km(coords, visit_counts, k).map_err(PyValueError::new_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn k_radius_of_gyration_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    timestamps: PyReadonlyArray1<'py, f64>,
    ranges: Vec<(usize, usize)>,
    k: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(k_radius_of_gyration_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        timestamps.as_slice()?,
        &ranges,
        k,
    )
    .map_err(PyValueError::new_err)?
    .into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn k_radius_of_gyration_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    timestamps: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    k: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(k_radius_of_gyration_indexed_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        timestamps.as_slice()?,
        indices.as_slice()?,
        ends.as_slice()?,
        k,
    )
    .map_err(PyValueError::new_err)?
    .into_pyarray(py))
}

#[pyfunction]
pub fn k_radius_of_gyration_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    timestamps: PyArray,
    ranges: Vec<(usize, usize)>,
    k: usize,
) -> PyResult<PyArray> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps = as_f64_array(timestamps, "timestamps")?;
    Ok(f64_results_into_arrow(
        k_radius_of_gyration_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            arrow_values(&timestamps),
            &ranges,
            k,
        )
        .map_err(PyValueError::new_err)?,
    ))
}

#[pyfunction]
pub fn k_radius_of_gyration_indexed_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    timestamps: PyArray,
    indices: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
    k: usize,
) -> PyResult<PyArray> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps = as_f64_array(timestamps, "timestamps")?;
    Ok(f64_results_into_arrow(
        k_radius_of_gyration_indexed_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            arrow_values(&timestamps),
            indices.as_slice()?,
            ends.as_slice()?,
            k,
        )
        .map_err(PyValueError::new_err)?,
    ))
}
