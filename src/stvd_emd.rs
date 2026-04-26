use arrow_array::{types::Float64Type, Float64Array, PrimitiveArray};
use ndarray::Array2;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use std::f64::consts::PI;
use wass::sliced_wasserstein;

fn _stvd_emd_impl(
    xs_a: &[f64],
    ys_a: &[f64],
    ts_a: &[f64],
    ws_a: &[f64],
    xs_b: &[f64],
    ys_b: &[f64],
    ts_b: &[f64],
    ws_b: &[f64],
    alpha: f64,
    cyclical_period: f64,
    num_projections: usize,
) -> PyResult<f64> {
    let n = xs_a.len();
    let m = xs_b.len();

    if n == 0 || m == 0 {
        return Err(PyValueError::new_err("distributions must be non-empty"));
    }
    if ys_a.len() != n || ts_a.len() != n || ws_a.len() != n {
        return Err(PyValueError::new_err(
            "all arrays for distribution A must have the same length",
        ));
    }
    if xs_b.len() != m || ys_b.len() != m || ts_b.len() != m || ws_b.len() != m {
        return Err(PyValueError::new_err(
            "all arrays for distribution B must have the same length",
        ));
    }
    if cyclical_period <= 0.0 {
        return Err(PyValueError::new_err("cyclical_period must be positive"));
    }
    if num_projections == 0 {
        return Err(PyValueError::new_err("num_projections must be positive"));
    }

    let sum_a: f64 = ws_a.iter().sum();
    let sum_b: f64 = ws_b.iter().sum();
    if sum_a <= 0.0 || sum_b <= 0.0 {
        return Err(PyValueError::new_err(
            "weights must sum to a positive value",
        ));
    }

    // wass::sliced_wasserstein in v0.2.0 is unweighted; we validate input weights
    // above to keep the same input contract for this function.

    let r = (alpha * cyclical_period) / (2.0 * PI);

    let mut cloud_a = Array2::<f32>::zeros((n, 4));
    for i in 0..n {
        cloud_a[[i, 0]] = xs_a[i] as f32;
        cloud_a[[i, 1]] = ys_a[i] as f32;

        let t = ts_a[i].rem_euclid(cyclical_period);
        let theta = 2.0 * PI * (t / cyclical_period);
        cloud_a[[i, 2]] = (r * theta.cos()) as f32;
        cloud_a[[i, 3]] = (r * theta.sin()) as f32;
    }

    let mut cloud_b = Array2::<f32>::zeros((m, 4));
    for j in 0..m {
        cloud_b[[j, 0]] = xs_b[j] as f32;
        cloud_b[[j, 1]] = ys_b[j] as f32;

        let t = ts_b[j].rem_euclid(cyclical_period);
        let theta = 2.0 * PI * (t / cyclical_period);
        cloud_b[[j, 2]] = (r * theta.cos()) as f32;
        cloud_b[[j, 3]] = (r * theta.sin()) as f32;
    }

    let seed = 42_u64;
    let p = 1.0_f32;
    let distance = sliced_wasserstein(&cloud_a, &cloud_b, num_projections, seed, p);

    Ok(distance as f64)
}

fn as_f64_array(arr: PyArray) -> PyResult<PrimitiveArray<Float64Type>> {
    let (array_ref, _field) = arr.into_inner();
    array_ref
        .as_any()
        .downcast_ref::<Float64Array>()
        .map(|a| a.clone())
        .ok_or_else(|| PyValueError::new_err("expected float64 Arrow array"))
}

#[pyfunction]
#[pyo3(signature = (xs_a, ys_a, times_a, weights_a, xs_b, ys_b, times_b, weights_b, alpha=10.0, cyclical_period=1440.0, num_projections=50))]
pub(crate) fn stvd_emd_numpy(
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
    _stvd_emd_impl(
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
}

#[pyfunction]
#[pyo3(signature = (xs_a, ys_a, times_a, weights_a, xs_b, ys_b, times_b, weights_b, alpha=10.0, cyclical_period=1440.0, num_projections=50))]
pub(crate) fn stvd_emd_arrow(
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
    // Convert each PyArray to Float64Array (zero-copy: borrows the PyArrow buffer)
    let xs_a = as_f64_array(xs_a)?;
    let ys_a = as_f64_array(ys_a)?;
    let ts_a = as_f64_array(times_a)?;
    let ws_a = as_f64_array(weights_a)?;
    let xs_b = as_f64_array(xs_b)?;
    let ys_b = as_f64_array(ys_b)?;
    let ts_b = as_f64_array(times_b)?;
    let ws_b = as_f64_array(weights_b)?;

    _stvd_emd_impl(
        xs_a.values(),
        ys_a.values(),
        ts_a.values(),
        ws_a.values(),
        xs_b.values(),
        ys_b.values(),
        ts_b.values(),
        ws_b.values(),
        alpha,
        cyclical_period,
        num_projections,
    )
}
