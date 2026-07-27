use fastmob_core::measures::individual::spatial_counts::{
    number_of_locations_from_ends_impl, number_of_locations_indexed_impl,
    number_of_visits_from_ends_impl, number_of_visits_indexed_impl,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::adapters::trajectory::{
    run_indexed_coordinate_u64, run_presorted_coordinate_u64, PyU64Result,
};

#[pyfunction]
pub fn number_of_visits_presorted<'py>(
    py: Python<'py>,
    n_values: usize,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    Ok(number_of_visits_from_ends_impl(n_values, ends.as_slice()?)
        .map_err(PyValueError::new_err)?
        .into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (n_values, indices, ends, valid_rows = None))]
pub fn number_of_visits_indexed<'py>(
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
pub fn number_of_locations_presorted<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyU64Result> {
    run_presorted_coordinate_u64(py, latitudes, longitudes, ends, |coords, ends| {
        number_of_locations_from_ends_impl(coords.latitudes, coords.longitudes, ends)
    })
}

#[pyfunction]
pub fn number_of_locations_indexed<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyU64Result> {
    run_indexed_coordinate_u64(py, latitudes, longitudes, indices, ends, |view| {
        number_of_locations_indexed_impl(
            view.coordinates.latitudes,
            view.coordinates.longitudes,
            view.indices,
            view.ends,
            view.valid_rows,
        )
    })
}
