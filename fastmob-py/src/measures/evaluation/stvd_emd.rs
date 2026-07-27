use fastmob_core::measures::evaluation::stvd_emd::stvd_emd_impl;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{arrow_values, as_f64_array};

#[pyfunction]
#[pyo3(signature = (xs_a, ys_a, times_a, weights_a, xs_b, ys_b, times_b, weights_b, alpha=10.0, cyclical_period=1440.0, num_projections=50))]
#[allow(clippy::too_many_arguments)]
pub fn stvd_emd(
    xs_a: ArrowPyArray,
    ys_a: ArrowPyArray,
    times_a: ArrowPyArray,
    weights_a: ArrowPyArray,
    xs_b: ArrowPyArray,
    ys_b: ArrowPyArray,
    times_b: ArrowPyArray,
    weights_b: ArrowPyArray,
    alpha: f64,
    cyclical_period: f64,
    num_projections: usize,
) -> PyResult<f64> {
    let xs_a = as_f64_array(xs_a, "xs_a")?;
    let ys_a = as_f64_array(ys_a, "ys_a")?;
    let times_a = as_f64_array(times_a, "times_a")?;
    let weights_a = as_f64_array(weights_a, "weights_a")?;
    let xs_b = as_f64_array(xs_b, "xs_b")?;
    let ys_b = as_f64_array(ys_b, "ys_b")?;
    let times_b = as_f64_array(times_b, "times_b")?;
    let weights_b = as_f64_array(weights_b, "weights_b")?;
    stvd_emd_impl(
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
    .map_err(PyValueError::new_err)
}
