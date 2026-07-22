use fastmob_core::preprocessing::clustering::{cluster_gmm_impl, cluster_kmeans_impl};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

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
