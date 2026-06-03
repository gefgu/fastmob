use linfa::prelude::*;
use linfa_clustering::{GaussianMixtureModel, KMeans};
use ndarray015::Array2;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

fn to_f32_array2(values: Vec<f64>) -> Result<Array2<f32>, String> {
    let n = values.len();
    let f32_values: Vec<f32> = values.into_iter().map(|v| v as f32).collect();
    Array2::from_shape_vec((n, 1), f32_values).map_err(|e| e.to_string())
}

pub fn cluster_kmeans_impl(
    values: Vec<f64>,
    n_clusters: usize,
    max_iter: u64,
    tolerance: f64,
    seed: u64,
) -> Result<Vec<usize>, String> {
    let n = values.len();
    if n < n_clusters {
        return Err(format!("need at least {n_clusters} samples, got {n}"));
    }
    let arr = to_f32_array2(values)?;
    let dataset = DatasetBase::from(arr);
    let rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let model = KMeans::params_with_rng(n_clusters, rng)
        .max_n_iterations(max_iter)
        .tolerance(tolerance as f32)
        .fit(&dataset)
        .map_err(|e| e.to_string())?;
    Ok(model.predict(&dataset).into_raw_vec())
}

pub fn cluster_gmm_impl(
    values: Vec<f64>,
    n_clusters: usize,
    max_iter: u64,
    tolerance: f64,
    seed: u64,
) -> Result<Vec<usize>, String> {
    let n = values.len();
    if n < n_clusters {
        return Err(format!("need at least {n_clusters} samples, got {n}"));
    }
    let arr = to_f32_array2(values)?;
    let dataset = DatasetBase::from(arr);
    let rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let model = GaussianMixtureModel::params(n_clusters)
        .max_n_iterations(max_iter)
        .tolerance(tolerance as f32)
        .with_rng(rng)
        .fit(&dataset)
        .map_err(|e| e.to_string())?;
    Ok(model.predict(&dataset).into_raw_vec())
}
