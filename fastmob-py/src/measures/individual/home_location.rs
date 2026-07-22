use fastmob_core::measures::individual::home_location::{
    home_location_from_ends_impl, home_location_indexed_impl,
};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::adapters::trajectory::{
    PyF64Pair, run_indexed_timed_coordinate_f64_pair, run_presorted_timed_coordinate_f64_pair,
};

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn home_location_presorted<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    hours: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
    start_night: f64,
    end_night: f64,
) -> PyResult<PyF64Pair> {
    run_presorted_timed_coordinate_f64_pair(
        py,
        latitudes,
        longitudes,
        hours,
        ends,
        |coords, ends| {
            home_location_from_ends_impl(
                coords.latitudes,
                coords.longitudes,
                coords.times,
                ends,
                start_night,
                end_night,
            )
        },
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn home_location_indexed<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    hours: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    start_night: f64,
    end_night: f64,
) -> PyResult<PyF64Pair> {
    run_indexed_timed_coordinate_f64_pair(py, latitudes, longitudes, hours, indices, ends, |view| {
        home_location_indexed_impl(
            view.coordinates.latitudes,
            view.coordinates.longitudes,
            view.coordinates.times,
            view.indices,
            view.ends,
            start_night,
            end_night,
            view.valid_rows,
        )
    })
}
