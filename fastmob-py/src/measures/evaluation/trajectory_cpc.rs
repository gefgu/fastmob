use fastmob_core::measures::evaluation::trajectory_cpc::trajectory_common_part_of_commuters_impl;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{arrow_values, as_f64_array};

fn is_arrow_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.hasattr("__arrow_c_array__")
}

#[pyfunction]
#[pyo3(signature = (latitudes_a, longitudes_a, indices_a, ends_a, latitudes_b, longitudes_b, indices_b, ends_b, resolution))]
#[allow(clippy::too_many_arguments)]
pub fn trajectory_common_part_of_commuters<'py>(
    latitudes_a: &Bound<'py, PyAny>,
    longitudes_a: &Bound<'py, PyAny>,
    indices_a: PyReadonlyArray1<'py, usize>,
    ends_a: PyReadonlyArray1<'py, usize>,
    latitudes_b: &Bound<'py, PyAny>,
    longitudes_b: &Bound<'py, PyAny>,
    indices_b: PyReadonlyArray1<'py, usize>,
    ends_b: PyReadonlyArray1<'py, usize>,
    resolution: u8,
) -> PyResult<f64> {
    if let (Ok(latitudes_a), Ok(longitudes_a), Ok(latitudes_b), Ok(longitudes_b)) = (
        latitudes_a.extract::<PyReadonlyArray1<f64>>(),
        longitudes_a.extract::<PyReadonlyArray1<f64>>(),
        latitudes_b.extract::<PyReadonlyArray1<f64>>(),
        longitudes_b.extract::<PyReadonlyArray1<f64>>(),
    ) {
        return trajectory_common_part_of_commuters_impl(
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
        .map_err(PyValueError::new_err);
    }

    if is_arrow_array(latitudes_a)?
        && is_arrow_array(longitudes_a)?
        && is_arrow_array(latitudes_b)?
        && is_arrow_array(longitudes_b)?
    {
        let latitudes_a = as_f64_array(latitudes_a.extract::<PyArray>()?, "latitudes_a")?;
        let longitudes_a = as_f64_array(longitudes_a.extract::<PyArray>()?, "longitudes_a")?;
        let latitudes_b = as_f64_array(latitudes_b.extract::<PyArray>()?, "latitudes_b")?;
        let longitudes_b = as_f64_array(longitudes_b.extract::<PyArray>()?, "longitudes_b")?;
        return trajectory_common_part_of_commuters_impl(
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
        .map_err(PyValueError::new_err);
    }

    Err(PyTypeError::new_err(
        "coordinate arrays must all be NumPy arrays or all be Arrow arrays",
    ))
}
