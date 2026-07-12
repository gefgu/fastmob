use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use fastmob_core::measures::evaluation::stvd_emd::stvd_emd_impl;

use crate::utils::{arrow_values, as_f64_array};

#[pyfunction]
#[pyo3(signature = (xs_a, ys_a, times_a, weights_a, xs_b, ys_b, times_b, weights_b, alpha=10.0, cyclical_period=1440.0, num_projections=50))]
#[allow(clippy::too_many_arguments)]
pub fn stvd_emd_numpy(
    xs_a: PyReadonlyArray1<f64>,
    ys_a: PyReadonlyArray1<f64>,
    times_a: PyReadonlyArray1<f64>,
    weights_a: PyReadonlyArray1<f64>,
    xs_b: PyReadonlyArray1<f64>,
    ys_b: PyReadonlyArray1<f64>,
    times_b: PyReadonlyArray1<f64>,
    weights_b: PyReadonlyArray1<f64>,
    alpha: f64,
    cyclical_period: f64,
    num_projections: usize,
) -> PyResult<f64> {
    stvd_emd_impl(
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
    .map_err(PyValueError::new_err)
}

#[pyfunction]
#[pyo3(signature = (xs_a, ys_a, times_a, weights_a, xs_b, ys_b, times_b, weights_b, alpha=10.0, cyclical_period=1440.0, num_projections=50))]
#[allow(clippy::too_many_arguments)]
pub fn stvd_emd_arrow(
    xs_a: PyArray,
    ys_a: PyArray,
    times_a: PyArray,
    weights_a: PyArray,
    xs_b: PyArray,
    ys_b: PyArray,
    times_b: PyArray,
    weights_b: PyArray,
    alpha: f64,
    cyclical_period: f64,
    num_projections: usize,
) -> PyResult<f64> {
    let xs_a = as_f64_array(xs_a, "xs_a")?;
    let ys_a = as_f64_array(ys_a, "ys_a")?;
    let ts_a = as_f64_array(times_a, "times_a")?;
    let ws_a = as_f64_array(weights_a, "weights_a")?;
    let xs_b = as_f64_array(xs_b, "xs_b")?;
    let ys_b = as_f64_array(ys_b, "ys_b")?;
    let ts_b = as_f64_array(times_b, "times_b")?;
    let ws_b = as_f64_array(weights_b, "weights_b")?;

    stvd_emd_impl(
        arrow_values(&xs_a),
        arrow_values(&ys_a),
        arrow_values(&ts_a),
        arrow_values(&ws_a),
        arrow_values(&xs_b),
        arrow_values(&ys_b),
        arrow_values(&ts_b),
        arrow_values(&ws_b),
        alpha,
        cyclical_period,
        num_projections,
    )
    .map_err(PyValueError::new_err)
}
