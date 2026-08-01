use fastmob_core::measures::collective::square_displacement::{
    mean_square_displacement_indexed_impl, square_displacement_km2 as core_square_displacement_km2,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::adapters::trajectory::run_indexed_timed_coordinate_arrow;

#[pyfunction]
pub fn square_displacement_km2(lat0: f64, lng0: f64, lat_t: f64, lng_t: f64) -> f64 {
    core_square_displacement_km2(lat0, lng0, lat_t, lng_t)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn mean_square_displacement_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
    delta_s: f64,
) -> PyResult<f64> {
    run_indexed_timed_coordinate_arrow(
        py,
        latitudes,
        longitudes,
        timestamps_s,
        indices,
        ends,
        |view| {
            mean_square_displacement_indexed_impl(
                view.coordinates.latitudes,
                view.coordinates.longitudes,
                view.coordinates.times,
                view.indices,
                view.ends,
                delta_s,
                view.valid_rows,
            )
        },
    )?
    .map_err(PyValueError::new_err)
}
