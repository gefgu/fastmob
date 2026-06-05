use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use skmob2_core::measures::individual::spatial_counts::{
    number_of_locations_impl, number_of_locations_indexed_impl, number_of_visits_impl,
    number_of_visits_indexed_impl,
};

use crate::utils::{arrow_values, as_f64_array, u64_results_into_arrow};

#[pyfunction]
pub fn number_of_visits_numpy<'py>(
    py: Python<'py>,
    n_values: usize,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    Ok(number_of_visits_impl(n_values, &ranges)
        .map_err(PyValueError::new_err)?
        .into_pyarray(py))
}

#[pyfunction]
pub fn number_of_visits_arrow(n_values: usize, ranges: Vec<(usize, usize)>) -> PyResult<PyArray> {
    Ok(u64_results_into_arrow(
        number_of_visits_impl(n_values, &ranges).map_err(PyValueError::new_err)?,
    ))
}

#[pyfunction]
pub fn number_of_visits_indexed_numpy<'py>(
    py: Python<'py>,
    n_values: usize,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    Ok(
        number_of_visits_indexed_impl(n_values, indices.as_slice()?, ends.as_slice()?)
            .map_err(PyValueError::new_err)?
            .into_pyarray(py),
    )
}

#[pyfunction]
pub fn number_of_visits_indexed_arrow(
    n_values: usize,
    indices: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<PyArray> {
    Ok(u64_results_into_arrow(
        number_of_visits_indexed_impl(n_values, indices.as_slice()?, ends.as_slice()?)
            .map_err(PyValueError::new_err)?,
    ))
}

#[pyfunction]
pub fn number_of_locations_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    Ok(
        number_of_locations_impl(latitudes.as_slice()?, longitudes.as_slice()?, &ranges)
            .map_err(PyValueError::new_err)?
            .into_pyarray(py),
    )
}

#[pyfunction]
pub fn number_of_locations_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    ranges: Vec<(usize, usize)>,
) -> PyResult<PyArray> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    Ok(u64_results_into_arrow(
        number_of_locations_impl(arrow_values(&latitudes), arrow_values(&longitudes), &ranges)
            .map_err(PyValueError::new_err)?,
    ))
}

#[pyfunction]
pub fn number_of_locations_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    Ok(number_of_locations_indexed_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        indices.as_slice()?,
        ends.as_slice()?,
    )
    .map_err(PyValueError::new_err)?
    .into_pyarray(py))
}

#[pyfunction]
pub fn number_of_locations_indexed_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    indices: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<PyArray> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    Ok(u64_results_into_arrow(
        number_of_locations_indexed_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            indices.as_slice()?,
            ends.as_slice()?,
        )
        .map_err(PyValueError::new_err)?,
    ))
}
