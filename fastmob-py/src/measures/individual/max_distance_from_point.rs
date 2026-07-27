use fastmob_core::measures::individual::max_distance_from_point::{
    max_distance_from_point_from_ends_impl, max_distance_from_point_indexed_impl,
};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::adapters::trajectory::{
    run_indexed_group_coordinate_f64, run_presorted_group_coordinate_f64, PyF64Result,
};

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn max_distance_from_point_presorted<'py>(
    py: Python<'py>,
    home_lats: &Bound<'py, PyAny>,
    home_lngs: &Bound<'py, PyAny>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyF64Result> {
    run_presorted_group_coordinate_f64(
        py,
        home_lats,
        home_lngs,
        latitudes,
        longitudes,
        ends,
        |coords, ends| {
            max_distance_from_point_from_ends_impl(
                coords.group_latitudes,
                coords.group_longitudes,
                coords.row_latitudes,
                coords.row_longitudes,
                ends,
                coords.valid_rows,
            )
        },
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn max_distance_from_point_indexed<'py>(
    py: Python<'py>,
    home_lats: &Bound<'py, PyAny>,
    home_lngs: &Bound<'py, PyAny>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyF64Result> {
    run_indexed_group_coordinate_f64(
        py,
        home_lats,
        home_lngs,
        latitudes,
        longitudes,
        indices,
        ends,
        |view| {
            max_distance_from_point_indexed_impl(
                view.coordinates.group_latitudes,
                view.coordinates.group_longitudes,
                view.coordinates.row_latitudes,
                view.coordinates.row_longitudes,
                view.indices,
                view.ends,
                view.valid_rows,
            )
        },
    )
}
