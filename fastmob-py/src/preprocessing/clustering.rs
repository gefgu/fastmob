use fastmob_core::preprocessing::clustering::{
    cluster_gmm_impl, cluster_kmeans_impl, cluster_standardized_kmeans_impl,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{arrow_values, as_f64_array, i32_results_into_arrow};

/// Cluster scalar values into `n_clusters` groups via K-Means.
/// Returns a label per input value in `0..n_clusters`.
#[pyfunction]
#[pyo3(signature = (values, n_clusters=3, max_iter=300, tolerance=1e-4, seed=42))]
pub fn cluster_kmeans(
    values: Vec<f64>,
    n_clusters: usize,
    max_iter: u64,
    tolerance: f64,
    seed: u64,
) -> PyResult<Vec<usize>> {
    cluster_kmeans_impl(values, n_clusters, max_iter, tolerance, seed)
        .map_err(PyValueError::new_err)
}

/// Standardize two Arrow float64 feature columns and cluster their rows via
/// multi-start K-Means.  The output is an Arrow int32 array aligned to inputs.
#[pyfunction]
#[pyo3(signature = (first, second, n_clusters=3, max_iter=300, tolerance=1e-4, seed=0, n_init=1))]
pub fn cluster_standardized_kmeans_arrow(
    py: Python<'_>,
    first: PyArray,
    second: PyArray,
    n_clusters: usize,
    max_iter: u64,
    tolerance: f64,
    seed: u64,
    n_init: usize,
) -> PyResult<PyArray> {
    let first = as_f64_array(first, "first")?;
    let second = as_f64_array(second, "second")?;
    if first.len() != second.len() {
        return Err(PyValueError::new_err(
            "first and second must have the same length",
        ));
    }
    let mut features = Vec::with_capacity(first.len() * 2);
    for (&left, &right) in arrow_values(&first).iter().zip(arrow_values(&second)) {
        features.extend([left, right]);
    }
    let labels = py.detach(|| {
        cluster_standardized_kmeans_impl(
            &features, 2, n_clusters, max_iter, tolerance, seed, n_init,
        )
    });
    let labels = labels.map_err(PyValueError::new_err)?;
    Ok(i32_results_into_arrow(
        labels.into_iter().map(|label| label as i32).collect(),
    ))
}

/// Cluster scalar values into `n_clusters` groups via Gaussian Mixture Model.
/// Returns a label per input value in `0..n_clusters`.
#[pyfunction]
#[pyo3(signature = (values, n_clusters=3, max_iter=100, tolerance=1e-4, seed=42))]
pub fn cluster_gmm(
    values: Vec<f64>,
    n_clusters: usize,
    max_iter: u64,
    tolerance: f64,
    seed: u64,
) -> PyResult<Vec<usize>> {
    cluster_gmm_impl(values, n_clusters, max_iter, tolerance, seed).map_err(PyValueError::new_err)
}
