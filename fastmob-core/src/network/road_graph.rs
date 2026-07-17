//! Road/rail routing over a graph using contraction hierarchies
//! (`fast_paths`): the graph is prepared once per session, then every
//! shortest-path query is a fast bidirectional search instead of a fresh
//! Dijkstra.

use rayon::prelude::*;
use rustc_hash::FxHashMap;

pub struct RoadGraph {
    fast_graph: fast_paths::FastGraph,
    edge_weight_ds: FxHashMap<(usize, usize), i64>,
    edge_length_m: FxHashMap<(usize, usize), f64>,
}

impl RoadGraph {
    /// Builds and prepares the contraction hierarchy from directed edges.
    /// Parallel edges between the same (from, to) pair are deduped, keeping
    /// the smallest weight; self-loops are dropped (`fast_paths` disallows them).
    pub fn build(edge_from: &[usize], edge_to: &[usize], edge_weight_ds: &[usize]) -> Self {
        Self::build_with_length(edge_from, edge_to, edge_weight_ds, &[])
    }

    /// Same as `build`, but also carries each edge's physical length (metres)
    /// alongside its travel-time weight, so a caller can later ask "how far
    /// (in metres) does the time-optimal route between these two nodes run"
    /// via `batch_road_distances`, without changing which route is chosen.
    /// `edge_length_m[i]` is missing/short (e.g. `&[]`, as `build` passes) ->
    /// treated as `0.0`, which is fine since a caller that only needs
    /// shortest-path weight never reads `edge_length_m()`.
    pub fn build_with_length(
        edge_from: &[usize],
        edge_to: &[usize],
        edge_weight_ds: &[usize],
        edge_length_m: &[f64],
    ) -> Self {
        let mut deduped: FxHashMap<(usize, usize), (usize, f64)> =
            FxHashMap::with_capacity_and_hasher(edge_from.len(), Default::default());
        for i in 0..edge_from.len() {
            let (from, to) = (edge_from[i], edge_to[i]);
            if from == to {
                continue;
            }
            let w = edge_weight_ds[i].max(1);
            let len = edge_length_m.get(i).copied().unwrap_or(0.0);
            deduped
                .entry((from, to))
                .and_modify(|existing| {
                    if w < existing.0 {
                        *existing = (w, len);
                    }
                })
                .or_insert((w, len));
        }

        let mut input_graph = fast_paths::InputGraph::new();
        for (&(from, to), &(w, _)) in &deduped {
            input_graph.add_edge(from, to, w);
        }
        input_graph.freeze();
        let fast_graph = fast_paths::prepare(&input_graph);

        let mut edge_weight_ds =
            FxHashMap::with_capacity_and_hasher(deduped.len(), Default::default());
        let mut edge_length_m =
            FxHashMap::with_capacity_and_hasher(deduped.len(), Default::default());
        for ((from, to), (w, len)) in deduped {
            edge_weight_ds.insert((from, to), w as i64);
            edge_length_m.insert((from, to), len);
        }

        Self {
            fast_graph,
            edge_weight_ds,
            edge_length_m,
        }
    }

    pub fn new_calculator(&self) -> fast_paths::PathCalculator {
        fast_paths::create_calculator(&self.fast_graph)
    }

    /// Returns (total_weight_ds, node_path including endpoints) or `None` if
    /// `from`/`to` are in disconnected components of the graph.
    pub fn shortest_path(
        &self,
        calc: &mut fast_paths::PathCalculator,
        from: usize,
        to: usize,
    ) -> Option<(usize, Vec<usize>)> {
        let path = calc.calc_path(&self.fast_graph, from, to)?;
        Some((path.get_weight(), path.get_nodes().clone()))
    }

    pub fn edge_weight_ds(&self, from: usize, to: usize) -> i64 {
        *self.edge_weight_ds.get(&(from, to)).unwrap_or(&0)
    }

    pub fn edge_length_m(&self, from: usize, to: usize) -> f64 {
        *self.edge_length_m.get(&(from, to)).unwrap_or(&0.0)
    }
}

/// Batch shortest-path distance (physical length, metres) for a set of
/// (from_node, to_node) queries against one prepared contraction hierarchy.
/// The batch is split into a handful of chunks per worker thread (not one
/// chunk per query -- `PathCalculator` allocates state sized to the graph's
/// node count, so building a fresh one per query, or even per rayon
/// work-stealing split under `map_init`, dwarfed the actual per-query
/// routing cost in practice); each chunk builds exactly one `PathCalculator`
/// and reuses it for every query in that chunk. Preparing the CH itself,
/// done once in `RoadGraph::build*`, remains the expensive one-time step.
/// Negative node ids (an "unsnapped" sentinel, e.g. from
/// `snap_locations_to_graph`) and CH-disconnected pairs both report
/// `(0.0, false)`, leaving the straight-line Haversine fallback decision to
/// the caller.
pub fn batch_road_distances(
    graph: &RoadGraph,
    from_nodes: &[i64],
    to_nodes: &[i64],
) -> (Vec<f64>, Vec<bool>) {
    let threads = rayon::current_num_threads().max(1);
    // 4 chunks per thread: enough chunks to balance load across workers even
    // when per-chunk routing cost varies, without driving the per-chunk
    // PathCalculator count high enough to matter.
    let chunk_size = (from_nodes.len() / (threads * 4)).max(1);

    from_nodes
        .par_chunks(chunk_size)
        .zip(to_nodes.par_chunks(chunk_size))
        .flat_map_iter(|(from_chunk, to_chunk)| {
            let mut calc = graph.new_calculator();
            from_chunk
                .iter()
                .zip(to_chunk.iter())
                .map(move |(&from, &to)| {
                    if from < 0 || to < 0 {
                        return (0.0, false);
                    }
                    match graph.shortest_path(&mut calc, from as usize, to as usize) {
                        Some((_weight_ds, nodes)) => {
                            let d: f64 = nodes
                                .windows(2)
                                .map(|w| graph.edge_length_m(w[0], w[1]))
                                .sum();
                            (d, true)
                        }
                        None => (0.0, false),
                    }
                })
                .collect::<Vec<_>>()
        })
        .unzip()
}

/// Given the full node path and per-node cumulative time, subsample down to
/// at most `max_points`, always keeping the first and last point.
pub fn subsample_waypoints(
    lats: &[f64],
    lngs: &[f64],
    times: &[i64],
    max_points: usize,
) -> (Vec<f64>, Vec<f64>, Vec<i64>) {
    let n = lats.len();
    if n <= max_points || max_points < 2 {
        return (lats.to_vec(), lngs.to_vec(), times.to_vec());
    }
    let mut idxs = Vec::with_capacity(max_points);
    let step = (n - 1) as f64 / (max_points - 1) as f64;
    for i in 0..max_points {
        let idx = ((i as f64) * step).round() as usize;
        idxs.push(idx.min(n - 1));
    }
    idxs.dedup();
    (
        idxs.iter().map(|&i| lats[i]).collect(),
        idxs.iter().map(|&i| lngs[i]).collect(),
        idxs.iter().map(|&i| times[i]).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shortest_path_over_a_chain() {
        // 0 -> 1 -> 2 -> 3, plus a direct but slower 0 -> 3 edge.
        let edge_from = vec![0, 1, 2, 0];
        let edge_to = vec![1, 2, 3, 3];
        let edge_weight = vec![10, 10, 10, 100];
        let graph = RoadGraph::build(&edge_from, &edge_to, &edge_weight);
        let mut calc = graph.new_calculator();
        let (weight, nodes) = graph.shortest_path(&mut calc, 0, 3).expect("path exists");
        assert_eq!(weight, 30);
        assert_eq!(nodes, vec![0, 1, 2, 3]);
    }

    #[test]
    fn disconnected_pair_returns_none() {
        let edge_from = vec![0, 2];
        let edge_to = vec![1, 3];
        let edge_weight = vec![5, 5];
        let graph = RoadGraph::build(&edge_from, &edge_to, &edge_weight);
        let mut calc = graph.new_calculator();
        assert!(graph.shortest_path(&mut calc, 0, 3).is_none());
    }

    #[test]
    fn shortest_path_length_over_a_chain() {
        // Same topology as `shortest_path_over_a_chain`: 0 -> 1 -> 2 -> 3 is
        // faster by weight (30 vs 100) but the direct 0 -> 3 edge is shorter
        // by length (50 vs 600). Distance must be summed along the
        // time-optimal path, not looked up per query pair directly.
        let edge_from = vec![0, 1, 2, 0];
        let edge_to = vec![1, 2, 3, 3];
        let edge_weight = vec![10, 10, 10, 100];
        let edge_length = vec![100.0, 200.0, 300.0, 50.0];
        let graph = RoadGraph::build_with_length(&edge_from, &edge_to, &edge_weight, &edge_length);
        let (distances, connected) = batch_road_distances(&graph, &[0], &[3]);
        assert_eq!(connected, vec![true]);
        assert_eq!(distances, vec![600.0]);
    }

    #[test]
    fn disconnected_pair_reports_not_connected() {
        let edge_from = vec![0, 2];
        let edge_to = vec![1, 3];
        let edge_weight = vec![5, 5];
        let edge_length = vec![10.0, 10.0];
        let graph = RoadGraph::build_with_length(&edge_from, &edge_to, &edge_weight, &edge_length);
        let (distances, connected) = batch_road_distances(&graph, &[0], &[3]);
        assert_eq!(connected, vec![false]);
        assert_eq!(distances, vec![0.0]);
    }

    #[test]
    fn negative_node_id_reports_not_connected() {
        let edge_from = vec![0, 1];
        let edge_to = vec![1, 2];
        let edge_weight = vec![5, 5];
        let edge_length = vec![10.0, 10.0];
        let graph = RoadGraph::build_with_length(&edge_from, &edge_to, &edge_weight, &edge_length);
        let (distances, connected) = batch_road_distances(&graph, &[-1], &[2]);
        assert_eq!(connected, vec![false]);
        assert_eq!(distances, vec![0.0]);
    }

    #[test]
    fn batch_reuses_one_calculator_across_mixed_queries() {
        // 0 -> 1 -> 2 -> 3 chain (connected), plus an isolated 4 -> 5 edge
        // (disconnected from the chain), queried in one batch alongside a
        // negative "unsnapped" node id, to prove per-index correctness when
        // several query kinds are interleaved against one shared calculator.
        let edge_from = vec![0, 1, 2, 4];
        let edge_to = vec![1, 2, 3, 5];
        let edge_weight = vec![10, 10, 10, 10];
        let edge_length = vec![100.0, 200.0, 300.0, 400.0];
        let graph = RoadGraph::build_with_length(&edge_from, &edge_to, &edge_weight, &edge_length);
        let (distances, connected) = batch_road_distances(&graph, &[0, 0, -1, 4], &[3, 5, 2, 5]);
        assert_eq!(connected, vec![true, false, false, true]);
        assert_eq!(distances, vec![600.0, 0.0, 0.0, 400.0]);
    }

    #[test]
    fn subsample_keeps_endpoints_and_caps_length() {
        let lats: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let lngs: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let times: Vec<i64> = (0..20).collect();
        let (lat, lng, t) = subsample_waypoints(&lats, &lngs, &times, 5);
        assert!(lat.len() <= 5);
        assert_eq!(lat.first(), Some(&0.0));
        assert_eq!(lat.last(), Some(&19.0));
        assert_eq!(lng.first(), Some(&0.0));
        assert_eq!(t.last(), Some(&19));
    }
}
