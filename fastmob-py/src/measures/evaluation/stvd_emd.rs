use fastmob_core::measures::evaluation::stvd_emd::stvd_emd_impl;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{arrow_values, as_f64_array};

#[pyfunction]
#[pyo3(signature = (lats_a, lngs_a, times_a, weights_a, lats_b, lngs_b, times_b, weights_b, alpha=10.0, cyclical_period=1440.0))]
#[allow(clippy::too_many_arguments)]
pub fn stvd_emd(
    lats_a: ArrowPyArray,
    lngs_a: ArrowPyArray,
    times_a: ArrowPyArray,
    weights_a: ArrowPyArray,
    lats_b: ArrowPyArray,
    lngs_b: ArrowPyArray,
    times_b: ArrowPyArray,
    weights_b: ArrowPyArray,
    alpha: f64,
    cyclical_period: f64,
) -> PyResult<f64> {
    let lats_a = as_f64_array(lats_a, "lats_a")?;
    let lngs_a = as_f64_array(lngs_a, "lngs_a")?;
    let times_a = as_f64_array(times_a, "times_a")?;
    let weights_a = as_f64_array(weights_a, "weights_a")?;
    let lats_b = as_f64_array(lats_b, "lats_b")?;
    let lngs_b = as_f64_array(lngs_b, "lngs_b")?;
    let times_b = as_f64_array(times_b, "times_b")?;
    let weights_b = as_f64_array(weights_b, "weights_b")?;
    stvd_emd_impl(
        arrow_values(&lats_a),
        arrow_values(&lngs_a),
        arrow_values(&times_a),
        arrow_values(&weights_a),
        arrow_values(&lats_b),
        arrow_values(&lngs_b),
        arrow_values(&times_b),
        arrow_values(&weights_b),
        alpha,
        cyclical_period,
    )
    .map_err(PyValueError::new_err)
}
