use ndarray::{Array1, Array2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use wass::sinkhorn_log;

#[pyfunction]
#[pyo3(signature = (xs_a, ys_a, times_a, weights_a, xs_b, ys_b, times_b, weights_b, alpha=10.0, cyclical_period=1440.0, reg=0.01, max_iter=1000))]
pub(crate) fn wasserstein_emd(
    xs_a: Vec<f64>,
    ys_a: Vec<f64>,
    times_a: Vec<f64>,
    weights_a: Vec<f64>,
    xs_b: Vec<f64>,
    ys_b: Vec<f64>,
    times_b: Vec<f64>,
    weights_b: Vec<f64>,
    alpha: f64,
    cyclical_period: f64,
    reg: f64,
    max_iter: usize,
) -> PyResult<f64> {
    let n = xs_a.len();
    let m = xs_b.len();

    if n == 0 || m == 0 {
        return Err(PyValueError::new_err("distributions must be non-empty"));
    }
    if ys_a.len() != n || times_a.len() != n || weights_a.len() != n {
        return Err(PyValueError::new_err(
            "all arrays for distribution A must have the same length",
        ));
    }
    if xs_b.len() != m || ys_b.len() != m || times_b.len() != m || weights_b.len() != m {
        return Err(PyValueError::new_err(
            "all arrays for distribution B must have the same length",
        ));
    }
    if reg <= 0.0 {
        return Err(PyValueError::new_err("reg must be positive"));
    }

    let sum_a: f64 = weights_a.iter().sum();
    let sum_b: f64 = weights_b.iter().sum();
    if sum_a <= 0.0 || sum_b <= 0.0 {
        return Err(PyValueError::new_err("weights must sum to a positive value"));
    }

    let a = Array1::<f32>::from_vec(weights_a.iter().map(|w| (w / sum_a) as f32).collect());
    let b = Array1::<f32>::from_vec(weights_b.iter().map(|w| (w / sum_b) as f32).collect());

    let mut cost = Array2::<f32>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let dx = xs_a[i] - xs_b[j];
            let dy = ys_a[i] - ys_b[j];
            let spatial_sq = dx * dx + dy * dy;

            let dt = (times_a[i] - times_b[j]).abs();
            let dt_cyclic = dt.min(cyclical_period - dt);
            let temporal = alpha * dt_cyclic;

            cost[[i, j]] = (spatial_sq + temporal * temporal).sqrt() as f32;
        }
    }

    let (_plan, distance) = sinkhorn_log(&a, &b, &cost, reg as f32, max_iter);
    Ok(distance as f64)
}
