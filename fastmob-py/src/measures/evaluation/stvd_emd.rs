use fastmob_core::measures::evaluation::stvd_emd::stvd_emd_impl;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{arrow_values, as_f64_array};

fn is_arrow_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.hasattr("__arrow_c_array__")
}

#[pyfunction]
#[pyo3(signature = (xs_a, ys_a, times_a, weights_a, xs_b, ys_b, times_b, weights_b, alpha=10.0, cyclical_period=1440.0, num_projections=50))]
#[allow(clippy::too_many_arguments)]
pub fn stvd_emd<'py>(
    xs_a: &Bound<'py, PyAny>,
    ys_a: &Bound<'py, PyAny>,
    times_a: &Bound<'py, PyAny>,
    weights_a: &Bound<'py, PyAny>,
    xs_b: &Bound<'py, PyAny>,
    ys_b: &Bound<'py, PyAny>,
    times_b: &Bound<'py, PyAny>,
    weights_b: &Bound<'py, PyAny>,
    alpha: f64,
    cyclical_period: f64,
    num_projections: usize,
) -> PyResult<f64> {
    if let (
        Ok(xs_a),
        Ok(ys_a),
        Ok(times_a),
        Ok(weights_a),
        Ok(xs_b),
        Ok(ys_b),
        Ok(times_b),
        Ok(weights_b),
    ) = (
        xs_a.extract::<PyReadonlyArray1<f64>>(),
        ys_a.extract::<PyReadonlyArray1<f64>>(),
        times_a.extract::<PyReadonlyArray1<f64>>(),
        weights_a.extract::<PyReadonlyArray1<f64>>(),
        xs_b.extract::<PyReadonlyArray1<f64>>(),
        ys_b.extract::<PyReadonlyArray1<f64>>(),
        times_b.extract::<PyReadonlyArray1<f64>>(),
        weights_b.extract::<PyReadonlyArray1<f64>>(),
    ) {
        return stvd_emd_impl(
            xs_a.as_slice()?,
            ys_a.as_slice()?,
            times_a.as_slice()?,
            weights_a.as_slice()?,
            xs_b.as_slice()?,
            ys_b.as_slice()?,
            times_b.as_slice()?,
            weights_b.as_slice()?,
            alpha,
            cyclical_period,
            num_projections,
        )
        .map_err(PyValueError::new_err);
    }

    if is_arrow_array(xs_a)?
        && is_arrow_array(ys_a)?
        && is_arrow_array(times_a)?
        && is_arrow_array(weights_a)?
        && is_arrow_array(xs_b)?
        && is_arrow_array(ys_b)?
        && is_arrow_array(times_b)?
        && is_arrow_array(weights_b)?
    {
        let xs_a = as_f64_array(xs_a.extract::<PyArray>()?, "xs_a")?;
        let ys_a = as_f64_array(ys_a.extract::<PyArray>()?, "ys_a")?;
        let times_a = as_f64_array(times_a.extract::<PyArray>()?, "times_a")?;
        let weights_a = as_f64_array(weights_a.extract::<PyArray>()?, "weights_a")?;
        let xs_b = as_f64_array(xs_b.extract::<PyArray>()?, "xs_b")?;
        let ys_b = as_f64_array(ys_b.extract::<PyArray>()?, "ys_b")?;
        let times_b = as_f64_array(times_b.extract::<PyArray>()?, "times_b")?;
        let weights_b = as_f64_array(weights_b.extract::<PyArray>()?, "weights_b")?;
        return stvd_emd_impl(
            arrow_values(&xs_a),
            arrow_values(&ys_a),
            arrow_values(&times_a),
            arrow_values(&weights_a),
            arrow_values(&xs_b),
            arrow_values(&ys_b),
            arrow_values(&times_b),
            arrow_values(&weights_b),
            alpha,
            cyclical_period,
            num_projections,
        )
        .map_err(PyValueError::new_err);
    }

    Err(PyTypeError::new_err(
        "STVD arrays must all be NumPy arrays or all be Arrow arrays",
    ))
}
