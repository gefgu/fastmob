use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use skmob2_core::measures::individual::spatial_counts::{
    number_of_locations_from_ends_impl, number_of_locations_indexed_impl,
    number_of_visits_from_ends_impl, number_of_visits_indexed_impl,
};

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, u64_results_into_arrow,
};

#[pyfunction]
pub fn number_of_visits_numpy<'py>(
    py: Python<'py>,
    n_values: usize,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    Ok(number_of_visits_from_ends_impl(n_values, ends.as_slice()?)
        .map_err(PyValueError::new_err)?
        .into_pyarray(py))
}

#[pyfunction]
pub fn number_of_visits_arrow(n_values: usize, ends: PyReadonlyArray1<usize>) -> PyResult<PyArray> {
    Ok(u64_results_into_arrow(
        number_of_visits_from_ends_impl(n_values, ends.as_slice()?)
            .map_err(PyValueError::new_err)?,
    ))
}

#[pyfunction]
#[pyo3(signature = (n_values, indices, ends, valid_rows = None))]
pub fn number_of_visits_indexed_numpy<'py>(
    py: Python<'py>,
    n_values: usize,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    valid_rows: Option<PyReadonlyArray1<'py, bool>>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let valid_slice: Option<&[bool]> = if let Some(ref v) = valid_rows {
        Some(v.as_slice()?)
    } else {
        None
    };
    Ok(
        number_of_visits_indexed_impl(n_values, indices.as_slice()?, ends.as_slice()?, valid_slice)
            .map_err(PyValueError::new_err)?
            .into_pyarray(py),
    )
}

#[pyfunction]
#[pyo3(signature = (n_values, indices, ends, valid_rows = None))]
pub fn number_of_visits_indexed_arrow(
    n_values: usize,
    indices: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
    valid_rows: Option<PyReadonlyArray1<bool>>,
) -> PyResult<PyArray> {
    let valid_slice: Option<&[bool]> = if let Some(ref v) = valid_rows {
        Some(v.as_slice()?)
    } else {
        None
    };
    Ok(u64_results_into_arrow(
        number_of_visits_indexed_impl(n_values, indices.as_slice()?, ends.as_slice()?, valid_slice)
            .map_err(PyValueError::new_err)?,
    ))
}

#[pyfunction]
pub fn number_of_locations_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    Ok(number_of_locations_from_ends_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        ends.as_slice()?,
    )
    .map_err(PyValueError::new_err)?
    .into_pyarray(py))
}

#[pyfunction]
pub fn number_of_locations_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<PyArray> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    Ok(u64_results_into_arrow(
        number_of_locations_from_ends_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            ends.as_slice()?,
        )
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
        None,
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
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    Ok(u64_results_into_arrow(
        number_of_locations_indexed_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            indices.as_slice()?,
            ends.as_slice()?,
            valid_rows.as_deref(),
        )
        .map_err(PyValueError::new_err)?,
    ))
}
