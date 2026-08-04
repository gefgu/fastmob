//! Sparse common-part-of-commuters comparison.

use rustc_hash::FxHashMap;

type Edge = (u64, u64);
pub type EdgeCounts = FxHashMap<Edge, f64>;

pub fn sparse_edge_counts(
    edges: impl IntoIterator<Item = (Option<u64>, Option<u64>, f64)>,
) -> Result<(EdgeCounts, f64), String> {
    let mut counts = EdgeCounts::default();
    let mut total = 0.0;
    for (origin, destination, weight) in edges {
        if !weight.is_finite() || weight < 0.0 {
            return Err("flow weights must be finite and non-negative".to_string());
        }
        let (Some(origin), Some(destination)) = (origin, destination) else {
            continue;
        };
        if origin == destination || weight == 0.0 {
            continue;
        }
        *counts.entry((origin, destination)).or_insert(0.0) += weight;
        total += weight;
    }
    Ok((counts, total))
}

pub fn common_part_of_commuters_from_counts(
    counts_a: EdgeCounts,
    total_a: f64,
    counts_b: EdgeCounts,
    total_b: f64,
) -> f64 {
    let total = total_a + total_b;
    if total == 0.0 {
        return 0.0;
    }
    let (smaller, larger) = if counts_a.len() <= counts_b.len() {
        (&counts_a, &counts_b)
    } else {
        (&counts_b, &counts_a)
    };
    let common: f64 = smaller
        .iter()
        .map(|(edge, &weight)| weight.min(*larger.get(edge).unwrap_or(&0.0)))
        .sum();
    2.0 * common / total
}

pub fn common_part_of_links_from_counts(counts_a: &EdgeCounts, counts_b: &EdgeCounts) -> f64 {
    let total = counts_a.len() + counts_b.len();
    if total == 0 {
        return 0.0;
    }
    let common = counts_a
        .keys()
        .filter(|edge| counts_b.contains_key(edge))
        .count();
    2.0 * common as f64 / total as f64
}

pub fn common_part_of_commuters_distance(values_a: &[f64], values_b: &[f64]) -> f64 {
    let denominator: f64 = values_a.iter().sum();
    if denominator == 0.0 || values_a.is_empty() || values_b.is_empty() {
        return 0.0;
    }
    let max = values_a
        .iter()
        .chain(values_b)
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() || max <= 0.0 {
        return 0.0;
    }
    let bins = (max / 2.0).ceil() as usize;
    let mut a = vec![0u64; bins];
    let mut b = vec![0u64; bins];
    for (values, counts) in [(values_a, &mut a), (values_b, &mut b)] {
        for &value in values {
            if value.is_finite() && value >= 0.0 {
                counts[((value / 2.0).floor() as usize).min(bins - 1)] += 1;
            }
        }
    }
    a.iter().zip(b).map(|(&x, y)| x.min(y) as f64).sum::<f64>() / denominator
}

/// Compute CPC from sparse weighted OD edge streams.
///
/// Null endpoints and self-loops are ignored. Edge weights must be finite and
/// non-negative; zero-weight edges are ignored. The kernel stores only edges
/// that occur in either input, never a dense location-by-location matrix.
pub fn common_part_of_commuters(
    edges_a: impl IntoIterator<Item = (Option<u64>, Option<u64>, f64)>,
    edges_b: impl IntoIterator<Item = (Option<u64>, Option<u64>, f64)>,
) -> Result<f64, String> {
    let (counts_a, total_a) = sparse_edge_counts(edges_a)?;
    let (counts_b, total_b) = sparse_edge_counts(edges_b)?;
    Ok(common_part_of_commuters_from_counts(
        counts_a, total_a, counts_b, total_b,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uses_only_shared_sparse_edges() {
        let value = common_part_of_commuters(
            [(Some(1), Some(2), 3.0), (Some(1), Some(2), 2.0)],
            [(Some(1), Some(2), 4.0), (Some(2), Some(3), 1.0)],
        )
        .unwrap();
        assert_eq!(value, 0.8);
    }

    #[test]
    fn ignores_nulls_self_loops_and_zero_weights() {
        let value = common_part_of_commuters(
            [
                (None, Some(2), 2.0),
                (Some(1), Some(1), 2.0),
                (Some(1), Some(2), 0.0),
            ],
            std::iter::empty(),
        )
        .unwrap();
        assert_eq!(value, 0.0);
    }
}
