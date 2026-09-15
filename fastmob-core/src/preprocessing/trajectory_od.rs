//! Fused H3-cell pairing and Origin-Destination edge counting.
//!
//! `trajectory_to_od` used to pair each user's consecutive H3 cells in Python/NumPy,
//! materialize the full (non-deduplicated) origin/destination arrays, then hand them to
//! a separate Narwhals `group_by` for counting. This does the pairing and the counting
//! in one pass over data already sorted and grouped by user, following the same sparse
//! `FxHashMap` pattern already used by `measures::evaluation::cpc::sparse_edge_counts`
//! and `measures::individual::location_frequency`.

use rustc_hash::FxHashMap;

pub type OdEdgeCounts = (Vec<u64>, Vec<u64>, Vec<u64>);

/// `ends` are cumulative per-user end offsets into `cells`, matching the contract of
/// `_build_presorted_user_ends`: `cells` is already sorted/grouped by user, and each
/// user's own consecutive-cell pairs never cross into another user's range.
pub fn od_edge_counts_impl(cells: &[u64], ends: &[usize], drop_self_loops: bool) -> OdEdgeCounts {
    let mut counts: FxHashMap<(u64, u64), u64> = FxHashMap::default();
    let mut start = 0usize;
    for &end in ends {
        if end > start + 1 {
            for idx in start..end - 1 {
                let (origin, destination) = (cells[idx], cells[idx + 1]);
                if !(drop_self_loops && origin == destination) {
                    *counts.entry((origin, destination)).or_insert(0) += 1;
                }
            }
        }
        start = end;
    }

    let mut edges: Vec<((u64, u64), u64)> = counts.into_iter().collect();
    edges.sort_unstable_by_key(|&((origin, destination), _)| (origin, destination));

    let mut origins = Vec::with_capacity(edges.len());
    let mut destinations = Vec::with_capacity(edges.len());
    let mut edge_counts = Vec::with_capacity(edges.len());
    for ((origin, destination), count) in edges {
        origins.push(origin);
        destinations.push(destination);
        edge_counts.push(count);
    }
    (origins, destinations, edge_counts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairs_and_counts_within_each_user_only() {
        // user 0: 1 -> 2 -> 1 (two edges, (1,2) and (2,1)); user 1: 3 -> 3 (self-loop)
        let cells = [1u64, 2, 1, 3, 3];
        let ends = [3usize, 5];
        let (origins, destinations, counts) = od_edge_counts_impl(&cells, &ends, true);
        assert_eq!(origins, vec![1, 2]);
        assert_eq!(destinations, vec![2, 1]);
        assert_eq!(counts, vec![1, 1]);
    }

    #[test]
    fn aggregates_duplicate_edges_across_users() {
        // both users go 1 -> 2, so the edge count should be 2, not two separate rows
        let cells = [1u64, 2, 1, 2];
        let ends = [2usize, 4];
        let (origins, destinations, counts) = od_edge_counts_impl(&cells, &ends, true);
        assert_eq!(origins, vec![1]);
        assert_eq!(destinations, vec![2]);
        assert_eq!(counts, vec![2]);
    }

    #[test]
    fn keeps_self_loops_when_requested() {
        let cells = [5u64, 5];
        let ends = [2usize];
        let (origins, destinations, counts) = od_edge_counts_impl(&cells, &ends, false);
        assert_eq!(origins, vec![5]);
        assert_eq!(destinations, vec![5]);
        assert_eq!(counts, vec![1]);
    }

    #[test]
    fn empty_input_returns_empty_output() {
        let (origins, destinations, counts) = od_edge_counts_impl(&[], &[], true);
        assert!(origins.is_empty());
        assert!(destinations.is_empty());
        assert!(counts.is_empty());
    }

    #[test]
    fn single_ping_user_contributes_no_edges() {
        let cells = [1u64];
        let ends = [1usize];
        let (origins, _, _) = od_edge_counts_impl(&cells, &ends, true);
        assert!(origins.is_empty());
    }
}
