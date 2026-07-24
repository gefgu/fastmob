//! Euclidean-eps DBSCAN over already-normalized shape-signature vectors
//! (`shape_signature::distance_geometry_signature`), using the
//! `linfa`/`linfa-clustering` dependency already available in this crate --
//! deliberately *not* tracktable's own box/Chebyshev-metric DBSCAN, since
//! signature values are already normalized to `[0,1]` and a plain Euclidean
//! ball is the natural (and simpler) neighbourhood shape at that point.

use linfa::DatasetBase;
use linfa::traits::Transformer;
use linfa_clustering::Dbscan;
use ndarray015::Array2;

/// Cluster a batch of already-computed shape signatures (one row per
/// trajectory, same length for every row). Returns one label per row:
/// a non-negative cluster id, or `-1` for DBSCAN noise (unclustered) --
/// same convention as `fastmob.preprocessing.cluster`.
pub fn cluster_shape_signatures(
    signatures: &[Vec<f64>],
    epsilon: f64,
    min_cluster_size: usize,
) -> Result<Vec<i64>, String> {
    let n = signatures.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    if min_cluster_size < 2 {
        return Err("min_cluster_size must be >= 2".to_string());
    }
    if epsilon <= 0.0 {
        return Err("epsilon must be > 0".to_string());
    }

    let dim = signatures[0].len();
    let mut flat = Vec::with_capacity(n * dim);
    for sig in signatures {
        if sig.len() != dim {
            return Err("all signatures must have the same length".to_string());
        }
        flat.extend(sig.iter().map(|&v| v as f32));
    }

    let arr = Array2::from_shape_vec((n, dim), flat).map_err(|e| e.to_string())?;
    let dataset = DatasetBase::from(arr);
    let memberships = Dbscan::params(min_cluster_size)
        .tolerance(epsilon as f32)
        .transform(dataset)
        .map_err(|e| e.to_string())?;

    Ok(memberships
        .targets
        .iter()
        .map(|opt| opt.map_or(-1, |c| c as i64))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distinct_blobs_cluster_separately() {
        let signatures: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![0.01, 0.01],
            vec![0.02, 0.0],
            vec![1.0, 1.0],
            vec![1.01, 0.99],
            vec![1.02, 1.0],
        ];
        let labels = cluster_shape_signatures(&signatures, 0.1, 2).unwrap();
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn isolated_point_is_noise() {
        let signatures: Vec<Vec<f64>> = vec![
            vec![0.0, 0.0],
            vec![0.01, 0.01],
            vec![0.02, 0.0],
            vec![5.0, 5.0], // far away, alone -> noise
        ];
        let labels = cluster_shape_signatures(&signatures, 0.1, 2).unwrap();
        assert_eq!(labels[3], -1);
    }

    #[test]
    fn rejects_invalid_params() {
        let signatures: Vec<Vec<f64>> = vec![vec![0.0, 0.0], vec![0.1, 0.1]];
        assert!(cluster_shape_signatures(&signatures, 0.1, 1).is_err());
        assert!(cluster_shape_signatures(&signatures, 0.0, 2).is_err());
    }

    #[test]
    fn empty_input_returns_empty_labels() {
        let signatures: Vec<Vec<f64>> = vec![];
        assert_eq!(
            cluster_shape_signatures(&signatures, 0.1, 2).unwrap(),
            Vec::<i64>::new()
        );
    }
}
