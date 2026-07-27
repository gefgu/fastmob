use fastmob_core::measures::individual::uncorrelated_entropy::uncorrelated_entropy_indexed_impl;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::adapters::trajectory::{run_indexed_coordinate_f64, PyF64Result};

#[pyfunction]
pub fn uncorrelated_entropy_indexed<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    normalize: bool,
) -> PyResult<PyF64Result> {
    run_indexed_coordinate_f64(py, latitudes, longitudes, indices, ends, |view| {
        uncorrelated_entropy_indexed_impl(
            view.coordinates.latitudes,
            view.coordinates.longitudes,
            view.indices,
            view.ends,
            normalize,
            view.valid_rows,
        )
    })
}
