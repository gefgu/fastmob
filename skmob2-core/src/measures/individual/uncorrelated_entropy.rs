use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::utils::validate_indexed_coord_ends;

pub fn uncorrelated_entropy_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
    normalize: bool,
) -> Result<Vec<f64>, String> {
    validate_indexed_coord_ends(latitudes, longitudes, indices, ends)?;

    Ok((0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let n = end - start;
            if n == 0 {
                return 0.0;
            }
            let mut counts: FxHashMap<(u64, u64), u64> =
                FxHashMap::with_capacity_and_hasher(n, Default::default());
            for &idx in &indices[start..end] {
                let key = (latitudes[idx].to_bits(), longitudes[idx].to_bits());
                *counts.entry(key).or_insert(0) += 1;
            }
            let total = n as f64;
            let entropy = counts.values().fold(0.0f64, |acc, &c| {
                let p = c as f64 / total;
                acc - p * p.log2()
            });
            if normalize {
                let n_unique = counts.len();
                if n_unique > 1 {
                    entropy / (n_unique as f64).log2()
                } else {
                    0.0
                }
            } else {
                entropy
            }
        })
        .collect())
}
