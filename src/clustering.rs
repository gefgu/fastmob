use linfa::prelude::*;
use linfa_clustering::{GaussianMixtureModel, KMeans};
use ndarray015::Array2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn to_f32_array2(values: Vec<f64>) -> Result<Array2<f32>, String> {
    let n = values.len();
    let f32_values: Vec<f32> = values.into_iter().map(|v| v as f32).collect();
    Array2::from_shape_vec((n, 1), f32_values).map_err(|e| e.to_string())
}

/// Cluster scalar values into `n_clusters` groups via K-Means.
/// Returns a label per input value in `0..n_clusters`.
#[pyfunction]
#[pyo3(signature = (values, n_clusters=3, max_iter=300, tolerance=1e-4, seed=42))]
pub(crate) fn cluster_kmeans(
    values: Vec<f64>,
    n_clusters: usize,
    max_iter: u64,
    tolerance: f64,
    seed: u64,
) -> PyResult<Vec<usize>> {
    let n = values.len();
    if n < n_clusters {
        return Err(PyValueError::new_err(format!(
            "need at least {n_clusters} samples, got {n}"
        )));
    }
    let arr = to_f32_array2(values).map_err(|e| PyValueError::new_err(e))?;
    let dataset = DatasetBase::from(arr);
    let rng = StdRng::seed_from_u64(seed);
    let model = KMeans::params_with_rng(n_clusters, rng)
        .max_n_iterations(max_iter)
        .tolerance(tolerance as f32)
        .fit(&dataset)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(model.predict(&dataset).into_raw_vec())
}

/// Cluster scalar values into `n_clusters` groups via Gaussian Mixture Model.
/// Returns a label per input value in `0..n_clusters`.
#[pyfunction]
#[pyo3(signature = (values, n_clusters=3, max_iter=100, tolerance=1e-4, seed=42))]
pub(crate) fn cluster_gmm(
    values: Vec<f64>,
    n_clusters: usize,
    max_iter: u64,
    tolerance: f64,
    seed: u64,
) -> PyResult<Vec<usize>> {
    let n = values.len();
    if n < n_clusters {
        return Err(PyValueError::new_err(format!(
            "need at least {n_clusters} samples, got {n}"
        )));
    }
    let arr = to_f32_array2(values).map_err(|e| PyValueError::new_err(e))?;
    let dataset = DatasetBase::from(arr);
    let rng = StdRng::seed_from_u64(seed);
    let model = GaussianMixtureModel::params(n_clusters)
        .max_n_iterations(max_iter)
        .tolerance(tolerance as f32)
        .with_rng(rng)
        .fit(&dataset)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(model.predict(&dataset).into_raw_vec())
}
