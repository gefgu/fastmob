use fastmob_core::measures::individual::k_radius_of_gyration::{
    k_radius_of_gyration_from_ends_impl, k_radius_of_gyration_indexed_impl,
    k_radius_of_gyration_km as core_k_rog_km,
};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::adapters::trajectory::{
    PyF64Result, run_indexed_timed_coordinate_f64, run_presorted_timed_coordinate_f64,
};

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
pub fn k_radius_of_gyration_presorted<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    timestamps: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
    k: usize,
) -> PyResult<PyF64Result> {
    run_presorted_timed_coordinate_f64(
        py,
        latitudes,
        longitudes,
        timestamps,
        ends,
        |coords, ends| {
            k_radius_of_gyration_from_ends_impl(
                coords.latitudes,
                coords.longitudes,
                coords.times,
                ends,
                k,
            )
        },
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn k_radius_of_gyration_indexed<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    timestamps: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    k: usize,
) -> PyResult<PyF64Result> {
    run_indexed_timed_coordinate_f64(
        py,
        latitudes,
        longitudes,
        timestamps,
        indices,
        ends,
        |view| {
            k_radius_of_gyration_indexed_impl(
                view.coordinates.latitudes,
                view.coordinates.longitudes,
                view.coordinates.times,
                view.indices,
                view.ends,
                k,
                view.valid_rows,
            )
        },
    )
}
