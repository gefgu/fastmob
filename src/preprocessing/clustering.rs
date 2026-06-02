use std::collections::HashMap;

use linfa::prelude::*;
use linfa_clustering::{Dbscan, GaussianMixtureModel, KMeans};
use linfa_nn::{distance::Distance, BallTree};
use ndarray015::{Array2, ArrayView, Dimension};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

const EARTH_RADIUS_KM: f64 = 6371.0088;

#[derive(Clone)]
struct HaversineRad;

impl Distance<f64> for HaversineRad {
    fn distance<D: Dimension>(&self, a: ArrayView<f64, D>, b: ArrayView<f64, D>) -> f64 {
        let mut ai = a.iter().copied();
        let mut bi = b.iter().copied();
        let lat1 = ai.next().unwrap_or(0.0);
        let lng1 = ai.next().unwrap_or(0.0);
        let lat2 = bi.next().unwrap_or(0.0);
        let lng2 = bi.next().unwrap_or(0.0);
        let dlat = (lat2 - lat1) / 2.0;
        let dlng = (lng2 - lng1) / 2.0;
        let h = dlat.sin().powi(2) + lat1.cos() * lat2.cos() * dlng.sin().powi(2);
        2.0 * h.sqrt().asin()
    }
}

fn dbscan_user_slice(
    lats: &[f64],
    lngs: &[f64],
    eps_rad: f64,
    min_points: usize,
) -> Result<Vec<i64>, String> {
    let n = lats.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let mut data = Array2::zeros((n, 2));
    for (i, (&lat, &lng)) in lats.iter().zip(lngs.iter()).enumerate() {
        data[(i, 0)] = lat.to_radians();
        data[(i, 1)] = lng.to_radians();
    }
    let raw_labels = Dbscan::params_with(min_points, HaversineRad, BallTree)
        .tolerance(eps_rad)
        .check()
        .map_err(|e| e.to_string())?
        .transform(&data);

    let mut counts: HashMap<usize, usize> = HashMap::new();
    for lbl in raw_labels.iter().flatten() {
        *counts.entry(*lbl).or_insert(0) += 1;
    }
    let mut sorted_clusters: Vec<usize> = counts.keys().copied().collect();
    sorted_clusters.sort_by(|&a, &b| counts[&b].cmp(&counts[&a]).then(a.cmp(&b)));
    let remap: HashMap<usize, i64> = sorted_clusters
        .into_iter()
        .enumerate()
        .map(|(new_id, old_id)| (old_id, new_id as i64))
        .collect();

    Ok(raw_labels
        .iter()
        .map(|lbl| lbl.map(|id| remap[&id]).unwrap_or(-1))
        .collect())
}

/// Cluster stop locations by user using DBSCAN with Haversine distance.
/// Returns one label per row; noise points get label -1.
/// Clusters are ranked by visit count (most visited = 0).
#[pyfunction]
pub(crate) fn cluster_dbscan_haversine(
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    ranges: Vec<(usize, usize)>,
    eps_km: f64,
    min_points: usize,
) -> PyResult<Vec<i64>> {
    let eps_rad = eps_km / EARTH_RADIUS_KM;
    let n = latitudes.len();
    let mut all_labels = vec![-1i64; n];
    for (start, end) in ranges {
        if start >= end {
            continue;
        }
        let user_labels =
            dbscan_user_slice(&latitudes[start..end], &longitudes[start..end], eps_rad, min_points)
                .map_err(PyValueError::new_err)?;
        all_labels[start..end].copy_from_slice(&user_labels);
    }
    Ok(all_labels)
}

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
    let arr = to_f32_array2(values).map_err(PyValueError::new_err)?;
    let dataset = DatasetBase::from(arr);
    let rng = Xoshiro256PlusPlus::seed_from_u64(seed);
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
    let arr = to_f32_array2(values).map_err(PyValueError::new_err)?;
    let dataset = DatasetBase::from(arr);
    let rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let model = GaussianMixtureModel::params(n_clusters)
        .max_n_iterations(max_iter)
        .tolerance(tolerance as f32)
        .with_rng(rng)
        .fit(&dataset)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(model.predict(&dataset).into_raw_vec())
}
