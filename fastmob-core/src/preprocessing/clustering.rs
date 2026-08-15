use linfa::prelude::*;
use linfa_clustering::{GaussianMixtureModel, KMeans};
use ndarray015::Array2;
use rand::{Rng, SeedableRng};
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

/// Cluster rows of a dense feature matrix after population-standardizing each
/// column.  This mirrors ``sklearn.preprocessing.StandardScaler`` followed by
/// K-Means, without materialising a Python/NumPy matrix at the binding edge.
///
/// ``features`` is row-major and must contain exactly ``n_rows * n_features``
/// values.  Callers can opt into multiple seeded initialisations; profile
/// classification uses one deterministic K-Means++ start because its three
/// deliberately coarse mobility groups are well-separated.
pub fn cluster_standardized_kmeans_impl(
    features: &[f64],
    n_features: usize,
    n_clusters: usize,
    max_iter: u64,
    tolerance: f64,
    seed: u64,
    n_init: usize,
) -> Result<Vec<usize>, String> {
    if n_features == 0 || !features.len().is_multiple_of(n_features) {
        return Err("features must be a non-empty row-major matrix".to_string());
    }
    let n_rows = features.len() / n_features;
    if n_rows < n_clusters {
        return Err(format!("need at least {n_clusters} samples, got {n_rows}"));
    }
    if !features.iter().all(|value| value.is_finite()) {
        return Err("features must be finite".to_string());
    }

    let mut means = vec![0.0; n_features];
    for row in features.chunks_exact(n_features) {
        for (feature, value) in row.iter().enumerate() {
            means[feature] += value;
        }
    }
    for mean in &mut means {
        *mean /= n_rows as f64;
    }
    let mut scales = vec![0.0; n_features];
    for row in features.chunks_exact(n_features) {
        for (feature, value) in row.iter().enumerate() {
            scales[feature] += (value - means[feature]).powi(2);
        }
    }
    for scale in &mut scales {
        *scale = (*scale / n_rows as f64).sqrt();
    }
    let standardized: Vec<f64> = features
        .chunks_exact(n_features)
        .flat_map(|row| {
            row.iter().enumerate().map(|(feature, value)| {
                if scales[feature] == 0.0 {
                    0.0
                } else {
                    (value - means[feature]) / scales[feature]
                }
            })
        })
        .collect();
    let attempts = n_init.max(1);
    let mut best: Option<(f64, Vec<usize>)> = None;
    for init in 0..attempts {
        let (inertia, labels) = kmeans_plus_plus(
            &standardized,
            n_rows,
            n_features,
            n_clusters,
            max_iter,
            tolerance,
            seed.wrapping_add(init as u64),
        );
        if best
            .as_ref()
            .is_none_or(|(best_inertia, _)| inertia < *best_inertia)
        {
            best = Some((inertia, labels));
        }
    }
    Ok(best.expect("at least one K-Means initialisation").1)
}

fn squared_distance(row: &[f64], centroid: &[f64]) -> f64 {
    row.iter()
        .zip(centroid)
        .map(|(value, center)| (value - center).powi(2))
        .sum()
}

/// K-Means++ specialised for FastMOB's small, dense profile matrix. Keeping
/// its loops contiguous avoids generic ML-dataset allocation overhead.
fn kmeans_plus_plus(
    values: &[f64],
    n_rows: usize,
    n_features: usize,
    n_clusters: usize,
    max_iter: u64,
    tolerance: f64,
    seed: u64,
) -> (f64, Vec<usize>) {
    if n_features == 2 {
        return kmeans_two_dimensions(values, n_rows, n_clusters, max_iter, tolerance, seed);
    }
    kmeans_plus_plus_generic(
        values, n_rows, n_features, n_clusters, max_iter, tolerance, seed,
    )
}

/// The profile classifier always has exactly two features.  This compact path
/// avoids iterator and slice construction in the innermost distance loop.
fn kmeans_two_dimensions(
    values: &[f64],
    n_rows: usize,
    n_clusters: usize,
    max_iter: u64,
    tolerance: f64,
    seed: u64,
) -> (f64, Vec<usize>) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut centroids = Vec::with_capacity(n_clusters * 2);
    let first = rng.gen_range(0..n_rows);
    centroids.extend_from_slice(&values[first * 2..first * 2 + 2]);
    let mut nearest = vec![f64::INFINITY; n_rows];
    for _ in 1..n_clusters {
        let cx = centroids[centroids.len() - 2];
        let cy = centroids[centroids.len() - 1];
        let mut total = 0.0;
        for index in 0..n_rows {
            let dx = values[index * 2] - cx;
            let dy = values[index * 2 + 1] - cy;
            nearest[index] = nearest[index].min(dx * dx + dy * dy);
            total += nearest[index];
        }
        let selected = if total == 0.0 {
            rng.gen_range(0..n_rows)
        } else {
            let target = rng.r#gen::<f64>() * total;
            let mut cumulative = 0.0;
            nearest
                .iter()
                .position(|distance| {
                    cumulative += distance;
                    cumulative >= target
                })
                .unwrap_or(n_rows - 1)
        };
        centroids.extend_from_slice(&values[selected * 2..selected * 2 + 2]);
    }

    let mut labels = vec![0; n_rows];
    for _ in 0..max_iter {
        let mut sums_x = vec![0.0; n_clusters];
        let mut sums_y = vec![0.0; n_clusters];
        let mut counts = vec![0usize; n_clusters];
        for index in 0..n_rows {
            let x = values[index * 2];
            let y = values[index * 2 + 1];
            let mut label = 0;
            let mut best_distance = f64::INFINITY;
            for cluster in 0..n_clusters {
                let dx = x - centroids[cluster * 2];
                let dy = y - centroids[cluster * 2 + 1];
                let distance = dx * dx + dy * dy;
                if distance < best_distance {
                    best_distance = distance;
                    label = cluster;
                }
            }
            labels[index] = label;
            counts[label] += 1;
            sums_x[label] += x;
            sums_y[label] += y;
        }
        let mut max_shift: f64 = 0.0;
        for (cluster, &count) in counts.iter().enumerate().take(n_clusters) {
            if count == 0 {
                continue;
            }
            let start = cluster * 2;
            let next_x = sums_x[cluster] / counts[cluster] as f64;
            let next_y = sums_y[cluster] / counts[cluster] as f64;
            let dx = centroids[start] - next_x;
            let dy = centroids[start + 1] - next_y;
            max_shift = max_shift.max(dx * dx + dy * dy);
            centroids[start] = next_x;
            centroids[start + 1] = next_y;
        }
        if max_shift <= tolerance * tolerance {
            break;
        }
    }
    let inertia = (0..n_rows)
        .map(|index| {
            let cluster = labels[index];
            let dx = values[index * 2] - centroids[cluster * 2];
            let dy = values[index * 2 + 1] - centroids[cluster * 2 + 1];
            dx * dx + dy * dy
        })
        .sum();
    (inertia, labels)
}

fn kmeans_plus_plus_generic(
    values: &[f64],
    n_rows: usize,
    n_features: usize,
    n_clusters: usize,
    max_iter: u64,
    tolerance: f64,
    seed: u64,
) -> (f64, Vec<usize>) {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut centroids = Vec::with_capacity(n_clusters * n_features);
    let first = rng.gen_range(0..n_rows);
    centroids.extend_from_slice(&values[first * n_features..(first + 1) * n_features]);

    let mut nearest = vec![f64::INFINITY; n_rows];
    for _ in 1..n_clusters {
        let latest = &centroids[centroids.len() - n_features..];
        let mut total = 0.0;
        for (index, row) in values.chunks_exact(n_features).enumerate() {
            nearest[index] = nearest[index].min(squared_distance(row, latest));
            total += nearest[index];
        }
        let selected = if total == 0.0 {
            rng.gen_range(0..n_rows)
        } else {
            let target = rng.r#gen::<f64>() * total;
            let mut cumulative = 0.0;
            nearest
                .iter()
                .position(|distance| {
                    cumulative += distance;
                    cumulative >= target
                })
                .unwrap_or(n_rows - 1)
        };
        centroids.extend_from_slice(&values[selected * n_features..(selected + 1) * n_features]);
    }

    let mut labels = vec![0; n_rows];
    for _ in 0..max_iter {
        let mut sums = vec![0.0; n_clusters * n_features];
        let mut counts = vec![0usize; n_clusters];
        for (index, row) in values.chunks_exact(n_features).enumerate() {
            let label = (0..n_clusters)
                .min_by(|&left, &right| {
                    squared_distance(row, &centroids[left * n_features..(left + 1) * n_features])
                        .total_cmp(&squared_distance(
                            row,
                            &centroids[right * n_features..(right + 1) * n_features],
                        ))
                })
                .expect("at least one cluster");
            labels[index] = label;
            counts[label] += 1;
            for (feature, value) in row.iter().enumerate() {
                sums[label * n_features + feature] += value;
            }
        }
        let mut max_shift: f64 = 0.0;
        for (cluster, &count) in counts.iter().enumerate().take(n_clusters) {
            if count == 0 {
                continue;
            }
            let start = cluster * n_features;
            let previous = centroids[start..start + n_features].to_vec();
            for feature in 0..n_features {
                centroids[start + feature] = sums[start + feature] / count as f64;
            }
            max_shift = max_shift.max(squared_distance(
                &previous,
                &centroids[start..start + n_features],
            ));
        }
        if max_shift <= tolerance * tolerance {
            break;
        }
    }
    let inertia = values
        .chunks_exact(n_features)
        .zip(&labels)
        .map(|(row, &label)| {
            squared_distance(
                row,
                &centroids[label * n_features..(label + 1) * n_features],
            )
        })
        .sum();
    (inertia, labels)
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
