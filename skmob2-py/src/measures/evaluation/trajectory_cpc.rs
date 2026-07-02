use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use skmob2_core::measures::evaluation::trajectory_cpc::trajectory_common_part_of_commuters_impl;

use crate::utils::{arrow_values, as_f64_array};

#[pyfunction]
#[pyo3(signature = (latitudes_a, longitudes_a, indices_a, ends_a, latitudes_b, longitudes_b, indices_b, ends_b, resolution))]
#[allow(clippy::too_many_arguments)]
pub fn trajectory_common_part_of_commuters_numpy(
    latitudes_a: PyReadonlyArray1<f64>,
    longitudes_a: PyReadonlyArray1<f64>,
    indices_a: PyReadonlyArray1<usize>,
    ends_a: PyReadonlyArray1<usize>,
    latitudes_b: PyReadonlyArray1<f64>,
    longitudes_b: PyReadonlyArray1<f64>,
    indices_b: PyReadonlyArray1<usize>,
    ends_b: PyReadonlyArray1<usize>,
    resolution: u8,
) -> PyResult<f64> {
    trajectory_common_part_of_commuters_impl(
        latitudes_a.as_slice()?,
        longitudes_a.as_slice()?,
        indices_a.as_slice()?,
        ends_a.as_slice()?,
        latitudes_b.as_slice()?,
        longitudes_b.as_slice()?,
        indices_b.as_slice()?,
        ends_b.as_slice()?,
        resolution,
    )
    .map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (latitudes_a, longitudes_a, indices_a, ends_a, latitudes_b, longitudes_b, indices_b, ends_b, resolution))]
#[allow(clippy::too_many_arguments)]
pub fn trajectory_common_part_of_commuters_arrow(
    latitudes_a: PyArray,
    longitudes_a: PyArray,
    indices_a: PyReadonlyArray1<usize>,
    ends_a: PyReadonlyArray1<usize>,
    latitudes_b: PyArray,
    longitudes_b: PyArray,
    indices_b: PyReadonlyArray1<usize>,
    ends_b: PyReadonlyArray1<usize>,
    resolution: u8,
) -> PyResult<f64> {
    let latitudes_a = as_f64_array(latitudes_a, "latitudes_a")?;
    let longitudes_a = as_f64_array(longitudes_a, "longitudes_a")?;
    let latitudes_b = as_f64_array(latitudes_b, "latitudes_b")?;
    let longitudes_b = as_f64_array(longitudes_b, "longitudes_b")?;

    trajectory_common_part_of_commuters_impl(
        arrow_values(&latitudes_a),
        arrow_values(&longitudes_a),
        indices_a.as_slice()?,
        ends_a.as_slice()?,
        arrow_values(&latitudes_b),
        arrow_values(&longitudes_b),
        indices_b.as_slice()?,
        ends_b.as_slice()?,
        resolution,
    )
    .map_err(PyValueError::new_err)
}
