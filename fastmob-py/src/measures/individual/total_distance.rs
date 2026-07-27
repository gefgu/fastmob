use fastmob_core::measures::individual::total_distance::{
    total_distance_indexed_impl, total_distance_presorted_impl,
};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::adapters::trajectory::{
    PyF64Result, run_indexed_coordinate_f64, run_presorted_coordinate_f64,
};

#[pyfunction]
pub fn total_distance_presorted<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyF64Result> {
    run_presorted_coordinate_f64(py, latitudes, longitudes, ends, |coords, ends| {
        total_distance_presorted_impl(coords.latitudes, coords.longitudes, ends)
    })
}

#[pyfunction]
pub fn total_distance_indexed<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyF64Result> {
    run_indexed_coordinate_f64(py, latitudes, longitudes, indices, ends, |view| {
        total_distance_indexed_impl(
            view.coordinates.latitudes,
            view.coordinates.longitudes,
            view.indices,
            view.ends,
            view.valid_rows,
        )
    })
}
