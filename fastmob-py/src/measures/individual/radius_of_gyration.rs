use fastmob_core::measures::individual::radius_of_gyration::{
    radius_of_gyration_indexed_impl, radius_of_gyration_presorted_impl,
};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::adapters::trajectory::{
    run_indexed_coordinate_f64_with_validity, run_presorted_coordinate_f64_with_validity,
};

#[pyfunction]
pub fn radius_of_gyration_presorted<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<(Py<PyAny>, Bound<'py, PyArray1<bool>>)> {
    run_presorted_coordinate_f64_with_validity(py, latitudes, longitudes, ends, |coords, ends| {
        radius_of_gyration_presorted_impl(coords.latitudes, coords.longitudes, ends)
    })
}

#[pyfunction]
pub fn radius_of_gyration_indexed<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<(Py<PyAny>, Bound<'py, PyArray1<bool>>)> {
    run_indexed_coordinate_f64_with_validity(py, latitudes, longitudes, indices, ends, |view| {
        radius_of_gyration_indexed_impl(
            view.coordinates.latitudes,
            view.coordinates.longitudes,
            view.indices,
            view.ends,
            view.valid_rows,
        )
    })
}
