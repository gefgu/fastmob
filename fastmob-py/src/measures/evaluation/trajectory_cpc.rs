use fastmob_core::measures::evaluation::trajectory_cpc::trajectory_common_part_of_commuters_impl;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{arrow_values, as_f64_array};

#[pyfunction]
#[pyo3(signature = (latitudes_a, longitudes_a, indices_a, ends_a, latitudes_b, longitudes_b, indices_b, ends_b, resolution))]
#[allow(clippy::too_many_arguments)]
pub fn trajectory_common_part_of_commuters<'py>(
    latitudes_a: ArrowPyArray,
    longitudes_a: ArrowPyArray,
    indices_a: PyReadonlyArray1<'py, usize>,
    ends_a: PyReadonlyArray1<'py, usize>,
    latitudes_b: ArrowPyArray,
    longitudes_b: ArrowPyArray,
    indices_b: PyReadonlyArray1<'py, usize>,
    ends_b: PyReadonlyArray1<'py, usize>,
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
