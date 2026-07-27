use fastmob_core::measures::individual::maximum_distance::{
    maximum_distance_impl, maximum_distance_indexed_impl,
};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::adapters::trajectory::{
    run_indexed_coordinate_f64, run_presorted_coordinate_f64, PyF64Result,
};

#[pyfunction]
pub fn maximum_distance_presorted<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyF64Result> {
    run_presorted_coordinate_f64(py, latitudes, longitudes, ends, |coords, ends| {
        maximum_distance_impl(coords.latitudes, coords.longitudes, ends)
    })
}

#[pyfunction]
pub fn maximum_distance_indexed<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyF64Result> {
    run_indexed_coordinate_f64(py, latitudes, longitudes, indices, ends, |view| {
        maximum_distance_indexed_impl(
            view.coordinates.latitudes,
            view.coordinates.longitudes,
            view.indices,
            view.ends,
            view.valid_rows,
        )
    })
}
