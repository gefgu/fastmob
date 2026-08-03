//! Co-presence graph construction and graph-metric computation for
//! contact-network analysis of mobility data.
//!
//! Two pieces exist here because they don't scale past toy graphs in pure
//! Python:
//!
//! - `build_co_presence_edges`: turns (day, location, agent) rows into an
//!   edge list + per-edge day-persistence, replacing an `itertools.combinations`
//!   loop keyed into a `dict[edge, set[day]]` (measured: 150s on a
//!   58,502-user real-world dataset, ~65M raw pair-instances).
//! - `compute_graph_metrics`: per-node clustering coefficient + per-edge
//!   topological overlap, replacing pure-Python `O(sum of degree^2)` nested
//!   loops over `set`-based adjacency (measured: ~51 minutes extrapolated on
//!   the same data, whose co-presence graph is unusually dense -- average
//!   degree ~1,070 because popular venues see the same users repeatedly
//!   across many days).

use rayon::prelude::*;
use rustc_hash::FxHashMap;
use std::cmp::Ordering;

/// Groups `(day, location, node)` rows by `(day, location)`, emits every
/// pair within each group up to `max_group_size` (groups larger than that
/// are skipped, treating an oversized daily crowd at one location as
/// uninformative rather than combinatorially exploding), then collapses
/// `(u, v, day)` instances into one edge per unique pair with
/// `persistence = distinct_day_count / time_steps`.
///
/// Inputs are one row per `(day, location, node)` presence (the caller is
/// expected to already deduplicate repeated presences before calling this).
pub fn build_co_presence_edges(
    day_codes: &[i64],
    location_codes: &[i64],
    nodes: &[i64],
    max_group_size: usize,
    time_steps: usize,
) -> (Vec<u32>, Vec<u32>, Vec<f64>, u64, u64) {
    let mut groups: FxHashMap<(i64, i64), Vec<u32>> = FxHashMap::default();
    for i in 0..nodes.len() {
        groups
            .entry((day_codes[i], location_codes[i]))
            .or_default()
            .push(nodes[i] as u32);
    }

    let mut skipped_groups = 0u64;
    let mut skipped_rows = 0u64;
    // (u, v, day) instances, deduped later via sort -- a pair seen at two
    // different locations on the same day must still count as one
    // day-of-persistence, not two.
    let mut triples: Vec<(u32, u32, i64)> = Vec::new();
    for ((day, _location), mut members) in groups {
        let n = members.len();
        if n < 2 {
            continue;
        }
        if n > max_group_size {
            skipped_groups += 1;
            skipped_rows += n as u64;
            continue;
        }
        members.sort_unstable();
        members.dedup();
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                triples.push((members[i], members[j], day));
            }
        }
    }

    triples.par_sort_unstable();
    triples.dedup();

    let mut edge_from = Vec::new();
    let mut edge_to = Vec::new();
    let mut persistence = Vec::new();
    let mut idx = 0;
    while idx < triples.len() {
        let (u, v, _) = triples[idx];
        let start = idx;
        while idx < triples.len() && triples[idx].0 == u && triples[idx].1 == v {
            idx += 1;
        }
        edge_from.push(u);
        edge_to.push(v);
        persistence.push((idx - start) as f64 / time_steps.max(1) as f64);
    }

    (
        edge_from,
        edge_to,
        persistence,
        skipped_groups,
        skipped_rows,
    )
}

/// Sorted, deduplicated adjacency list per node, built from an edge list
/// (each edge contributes to both endpoints' neighbor lists).
fn build_adjacency(node_count: usize, edge_from: &[u32], edge_to: &[u32]) -> Vec<Vec<u32>> {
    let mut degree = vec![0u32; node_count];
    for i in 0..edge_from.len() {
        degree[edge_from[i] as usize] += 1;
        degree[edge_to[i] as usize] += 1;
    }
    let mut adjacency: Vec<Vec<u32>> = degree
        .iter()
        .map(|&d| Vec::with_capacity(d as usize))
        .collect();
    for i in 0..edge_from.len() {
        let (u, v) = (edge_from[i], edge_to[i]);
        adjacency[u as usize].push(v);
        adjacency[v as usize].push(u);
    }
    adjacency.par_iter_mut().for_each(|neighbors| {
        neighbors.sort_unstable();
        neighbors.dedup();
    });
    adjacency
}

/// Count of elements present in both sorted slices, restricted to values
/// `> threshold`. Used to count, for a node `i` and one neighbor `a` (with
/// `a` itself drawn from `i`'s sorted neighbor list), how many of `i`'s
/// other neighbors greater than `a` are also neighbors of `a` -- i.e. how
/// many triangles through `i` include the edge `(a, b)` for `b > a`,
/// counting each unordered neighbor pair `{a, b}` exactly once.
fn count_common_greater_than(sorted_a: &[u32], sorted_b: &[u32], threshold: u32) -> u64 {
    let start_a = sorted_a.partition_point(|&x| x <= threshold);
    let start_b = sorted_b.partition_point(|&x| x <= threshold);
    let (mut i, mut j) = (start_a, start_b);
    let mut count = 0u64;
    while i < sorted_a.len() && j < sorted_b.len() {
        match sorted_a[i].cmp(&sorted_b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

/// Size of the intersection of two sorted slices (no threshold).
fn count_common(sorted_a: &[u32], sorted_b: &[u32]) -> u64 {
    let (mut i, mut j) = (0, 0);
    let mut count = 0u64;
    while i < sorted_a.len() && j < sorted_b.len() {
        match sorted_a[i].cmp(&sorted_b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                count += 1;
                i += 1;
                j += 1;
            }
        }
    }
    count
}

/// Topological (Jaccard) overlap of two nodes' neighbor sets, for arbitrary
/// node ids -- not just ones already connected by an edge. Used by social-tie
/// inference (`random_baseline_overlap_threshold`) to evaluate overlap for
/// randomly sampled pairs that are usually not edges themselves.
pub fn pairwise_topological_overlap(adjacency: &[Vec<u32>], a: usize, b: usize) -> f64 {
    if a == b {
        return 1.0;
    }
    let (na, nb) = (&adjacency[a], &adjacency[b]);
    let inter = count_common(na, nb);
    let union = na.len() + nb.len() - inter as usize;
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

pub struct GraphMetrics {
    pub clustering_coefficient: Vec<f64>,
    pub topological_overlap: Vec<f64>,
}

/// Per-node clustering coefficient and per-edge topological overlap
/// (Jaccard similarity of endpoint neighborhoods), computed in parallel
/// across nodes/edges using sorted-adjacency intersection instead of
/// Python's per-pair `set` membership checks -- the same asymptotic
/// complexity as the naive approach for a graph this dense, but with a
/// vastly better constant factor (no per-check Python/hash overhead) and
/// rayon parallelism across cores.
pub fn compute_graph_metrics(
    node_count: usize,
    edge_from: &[u32],
    edge_to: &[u32],
) -> GraphMetrics {
    let adjacency = build_adjacency(node_count, edge_from, edge_to);

    let clustering_coefficient: Vec<f64> = (0..node_count)
        .into_par_iter()
        .map(|i| {
            let neighbors = &adjacency[i];
            let degree = neighbors.len();
            if degree < 2 {
                return 0.0;
            }
            let links: u64 = neighbors
                .iter()
                .map(|&a| count_common_greater_than(neighbors, &adjacency[a as usize], a))
                .sum();
            (2.0 * links as f64) / (degree as f64 * (degree as f64 - 1.0))
        })
        .collect();

    let topological_overlap: Vec<f64> = (0..edge_from.len())
        .into_par_iter()
        .map(|i| {
            let (u, v) = (edge_from[i] as usize, edge_to[i] as usize);
            pairwise_topological_overlap(&adjacency, u, v)
        })
        .collect();

    GraphMetrics {
        clustering_coefficient,
        topological_overlap,
    }
}

/// Samples `samples` random node pairs and returns their topological
/// overlaps' `(1 - p_rnd)`-quantile -- the null-model significance threshold
/// used by social-tie inference to decide whether an observed pair's overlap
/// is higher than random chance would produce. Deterministic given `seed`.
pub fn random_baseline_overlap_threshold(
    node_count: usize,
    edge_from: &[u32],
    edge_to: &[u32],
    samples: usize,
    p_rnd: f64,
    seed: u64,
) -> f64 {
    use rand::Rng;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    if node_count < 2 || samples == 0 {
        return 0.0;
    }
    let adjacency = build_adjacency(node_count, edge_from, edge_to);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);

    let mut overlaps: Vec<f64> = (0..samples)
        .map(|_| {
            let a = rng.gen_range(0..node_count);
            let mut b = rng.gen_range(0..node_count);
            while b == a {
                b = rng.gen_range(0..node_count);
            }
            pairwise_topological_overlap(&adjacency, a, b)
        })
        .collect();

    overlaps.sort_by(|x, y| x.total_cmp(y));
    let p_rnd = p_rnd.clamp(0.0, 1.0);
    let quantile = (1.0 - p_rnd).clamp(0.0, 1.0);
    let idx = ((overlaps.len() as f64 - 1.0) * quantile).round() as usize;
    overlaps[idx.min(overlaps.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn co_presence_edges_from_two_day_groups() {
        // Day 0 at location A: users 1,2,3 co-present -> edges (1,2),(1,3),(2,3).
        // Day 1 at location A: users 1,2 co-present again -> edge (1,2) persists.
        // Day 0 at location B: users 4,5 -> edge (4,5), independent component.
        let day = vec![0, 0, 0, 1, 1, 0, 0];
        let loc = vec![0, 0, 0, 0, 0, 1, 1];
        let node = vec![1, 2, 3, 1, 2, 4, 5];
        let (from, to, persistence, skipped_groups, skipped_rows) =
            build_co_presence_edges(&day, &loc, &node, 200, 2);
        assert_eq!(skipped_groups, 0);
        assert_eq!(skipped_rows, 0);

        let mut edges: Vec<(u32, u32, f64)> = from
            .into_iter()
            .zip(to)
            .zip(persistence)
            .map(|((u, v), p)| (u, v, p))
            .collect();
        edges.sort_by_key(|&(u, v, _)| (u, v));

        assert_eq!(
            edges,
            vec![(1, 2, 1.0), (1, 3, 0.5), (2, 3, 0.5), (4, 5, 0.5)]
        );
    }

    #[test]
    fn oversized_group_is_skipped_not_paired() {
        let day = vec![0, 0, 0];
        let loc = vec![0, 0, 0];
        let node = vec![1, 2, 3];
        let (from, to, persistence, skipped_groups, skipped_rows) =
            build_co_presence_edges(&day, &loc, &node, 2, 1);
        assert!(from.is_empty());
        assert!(to.is_empty());
        assert!(persistence.is_empty());
        assert_eq!(skipped_groups, 1);
        assert_eq!(skipped_rows, 3);
    }

    #[test]
    fn duplicate_presence_in_same_group_counted_once() {
        // Same (day, location, node) row repeated should not produce a
        // self-loop or double an edge -- dedup happens inside the group.
        let day = vec![0, 0, 0];
        let loc = vec![0, 0, 0];
        let node = vec![1, 1, 2];
        let (from, to, persistence, _, _) = build_co_presence_edges(&day, &loc, &node, 200, 1);
        assert_eq!(from, vec![1]);
        assert_eq!(to, vec![2]);
        assert_eq!(persistence, vec![1.0]);
    }

    #[test]
    fn clustering_coefficient_of_a_triangle_is_one() {
        // 0-1-2 triangle plus a pendant 3 attached to 0. Node 0 has 3
        // neighbors (1,2,3) but only the pair (1,2) is connected, so its
        // own coefficient is 1/3; nodes 1 and 2 have exactly 2 neighbors
        // each, both connected, so theirs is 1.0; node 3 has degree 1 (< 2)
        // so its coefficient is 0 by definition.
        let edge_from = vec![0u32, 1, 0, 0];
        let edge_to = vec![1u32, 2, 2, 3];
        let metrics = compute_graph_metrics(4, &edge_from, &edge_to);
        assert!((metrics.clustering_coefficient[0] - 1.0 / 3.0).abs() < 1e-9);
        assert!((metrics.clustering_coefficient[1] - 1.0).abs() < 1e-9);
        assert!((metrics.clustering_coefficient[2] - 1.0).abs() < 1e-9);
        assert_eq!(metrics.clustering_coefficient[3], 0.0);
    }

    #[test]
    fn clustering_coefficient_of_a_star_is_zero() {
        // Center 0 connected to 1,2,3 with no edges among the leaves.
        let edge_from = vec![0u32, 0, 0];
        let edge_to = vec![1u32, 2, 3];
        let metrics = compute_graph_metrics(4, &edge_from, &edge_to);
        assert_eq!(metrics.clustering_coefficient[0], 0.0);
    }

    #[test]
    fn topological_overlap_matches_hand_computed_jaccard() {
        // Path 0-1-2-3 plus edge 0-2.
        let edge_from = vec![0u32, 1, 2, 0];
        let edge_to = vec![1u32, 2, 3, 2];
        let metrics = compute_graph_metrics(4, &edge_from, &edge_to);
        // N(0) = {1,2}, N(1) = {0,2}, N(2) = {0,1,3}, N(3) = {2}
        // edge (0,1): inter({1,2},{0,2}) = {2} -> 1; union = 2+2-1 = 3 -> 1/3
        // edge (1,2): inter({0,2},{0,1,3}) = {0} -> 1; union = 2+3-1 = 4 -> 1/4
        // edge (2,3): inter({0,1,3},{2}) = {} -> 0; union = 3+1-0 = 4 -> 0
        // edge (0,2): inter({1,2},{0,1,3}) = {1} -> 1; union = 2+3-1 = 4 -> 1/4
        let mut by_edge: FxHashMap<(u32, u32), f64> = FxHashMap::default();
        for i in 0..edge_from.len() {
            by_edge.insert((edge_from[i], edge_to[i]), metrics.topological_overlap[i]);
        }
        assert!((by_edge[&(0, 1)] - 1.0 / 3.0).abs() < 1e-9);
        assert!((by_edge[&(1, 2)] - 1.0 / 4.0).abs() < 1e-9);
        assert!((by_edge[&(2, 3)] - 0.0).abs() < 1e-9);
        assert!((by_edge[&(0, 2)] - 1.0 / 4.0).abs() < 1e-9);
    }

    #[test]
    fn empty_graph_produces_empty_metrics() {
        let metrics = compute_graph_metrics(0, &[], &[]);
        assert!(metrics.clustering_coefficient.is_empty());
        assert!(metrics.topological_overlap.is_empty());
    }

    #[test]
    fn pairwise_topological_overlap_works_for_non_edge_pairs() {
        // 0-1, 2-1: node 0 and node 2 are not connected to each other, but
        // both connect to 1 -- overlap must still be computable directly.
        let edge_from = vec![0u32, 2];
        let edge_to = vec![1u32, 1];
        let adjacency = build_adjacency(3, &edge_from, &edge_to);
        // N(0) = {1}, N(2) = {1} -> intersection {1}, union {1} -> overlap 1.0
        assert!((pairwise_topological_overlap(&adjacency, 0, 2) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn random_baseline_overlap_threshold_is_deterministic() {
        let edge_from = vec![0u32, 1, 2, 0];
        let edge_to = vec![1u32, 2, 3, 2];
        let a = random_baseline_overlap_threshold(4, &edge_from, &edge_to, 100, 0.1, 42);
        let b = random_baseline_overlap_threshold(4, &edge_from, &edge_to, 100, 0.1, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn random_baseline_overlap_threshold_handles_tiny_graph() {
        let result = random_baseline_overlap_threshold(1, &[], &[], 10, 0.1, 1);
        assert_eq!(result, 0.0);
    }
}
