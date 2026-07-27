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
    node_lat: Vec<f64>,
    node_lng: Vec<f64>,
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
        Self::build_with_length_and_coords(
            edge_from,
            edge_to,
            edge_weight_ds,
            edge_length_m,
            &[],
            &[],
        )
    }

    /// Same as `build_with_length`, but also carries each node's (lat, lng),
    /// so `batch_road_routes` can emit route geometry (waypoint coordinates)
    /// alongside the time-optimal path, not just its total distance. A node
    /// id `>= node_lat.len()` (or when `node_lat`/`node_lng` are empty, as
    /// `build`/`build_with_length` pass) has no known coordinate; `node_coord`
    /// reports `None` for it and `batch_road_routes` drops it from the
    /// emitted waypoints while still accounting for its travel-time weight.
    pub fn build_with_length_and_coords(
        edge_from: &[usize],
        edge_to: &[usize],
        edge_weight_ds: &[usize],
        edge_length_m: &[f64],
        node_lat: &[f64],
        node_lng: &[f64],
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
            node_lat: node_lat.to_vec(),
            node_lng: node_lng.to_vec(),
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

    /// Returns `(lat, lng)` for `node`, or `None` if the graph was built
    /// without coordinates (`build`/`build_with_length`) or `node` is out of
    /// range for the coordinate arrays supplied to `build_with_length_and_coords`.
    pub fn node_coord(&self, node: usize) -> Option<(f64, f64)> {
        if node < self.node_lat.len() {
            Some((self.node_lat[node], self.node_lng[node]))
        } else {
            None
        }
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

/// Batch shortest-path route geometry: for each (from_node, to_node) query,
/// walks the same time-optimal node path `batch_road_distances` uses to sum
/// distance, but instead emits the path's waypoint coordinates (decimated to
/// at most `max_waypoints` via `subsample_waypoints`) and per-waypoint
/// cumulative travel-time weight. Nodes with no known coordinate (see
/// `RoadGraph::node_coord`) are dropped from the emitted waypoints but still
/// contribute their edge weight to the cumulative total, keeping later
/// waypoints' cumulative values correct.
///
/// Results are flattened across all queries: `lats`/`lngs`/`cum_weight_ds`
/// hold every query's waypoints back-to-back, and `starts[i]..ends[i]` is
/// query `i`'s slice into those flat arrays (the same flat-output +
/// user-boundary convention used elsewhere for variable-length-per-group
/// kernels). A disconnected pair or an unsnapped (negative) node id reports
/// `connected[i] == false` and an empty `starts[i]..ends[i]` slice.
type RouteBatch = (
    Vec<f64>,
    Vec<f64>,
    Vec<i64>,
    Vec<bool>,
    Vec<usize>,
    Vec<usize>,
);
type RouteQueryResult = (Vec<f64>, Vec<f64>, Vec<i64>, bool);

pub fn batch_road_routes(
    graph: &RoadGraph,
    from_nodes: &[i64],
    to_nodes: &[i64],
    max_waypoints: usize,
) -> RouteBatch {
    let threads = rayon::current_num_threads().max(1);
    let chunk_size = (from_nodes.len() / (threads * 4)).max(1);

    let per_query: Vec<RouteQueryResult> = from_nodes
        .par_chunks(chunk_size)
        .zip(to_nodes.par_chunks(chunk_size))
        .flat_map_iter(|(from_chunk, to_chunk)| {
            let mut calc = graph.new_calculator();
            from_chunk
                .iter()
                .zip(to_chunk.iter())
                .map(move |(&from, &to)| {
                    if from < 0 || to < 0 {
                        return (Vec::new(), Vec::new(), Vec::new(), false);
                    }
                    match graph.shortest_path(&mut calc, from as usize, to as usize) {
                        Some((_weight_ds, nodes)) => {
                            let mut lats_full = Vec::with_capacity(nodes.len());
                            let mut lngs_full = Vec::with_capacity(nodes.len());
                            let mut cum_full: Vec<i64> = Vec::with_capacity(nodes.len());
                            let mut cum = 0i64;
                            for (i, &node) in nodes.iter().enumerate() {
                                if i > 0 {
                                    cum += graph.edge_weight_ds(nodes[i - 1], node);
                                }
                                if let Some((lat, lng)) = graph.node_coord(node) {
                                    lats_full.push(lat);
                                    lngs_full.push(lng);
                                    cum_full.push(cum);
                                }
                            }
                            let (lat_d, lng_d, cum_d) = subsample_waypoints(
                                &lats_full,
                                &lngs_full,
                                &cum_full,
                                max_waypoints,
                            );
                            (lat_d, lng_d, cum_d, true)
                        }
                        None => (Vec::new(), Vec::new(), Vec::new(), false),
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut out_lats = Vec::new();
    let mut out_lngs = Vec::new();
    let mut out_cum = Vec::new();
    let mut connected = Vec::with_capacity(per_query.len());
    let mut starts = Vec::with_capacity(per_query.len());
    let mut ends = Vec::with_capacity(per_query.len());
    for (lat_d, lng_d, cum_d, is_connected) in per_query {
        let start = out_lats.len();
        out_lats.extend(lat_d);
        out_lngs.extend(lng_d);
        out_cum.extend(cum_d);
        starts.push(start);
        ends.push(out_lats.len());
        connected.push(is_connected);
    }

    (out_lats, out_lngs, out_cum, connected, starts, ends)
}

/// Batch OD (origin-destination) desire-line aggregation: for each
/// (from_node, to_node, flow) triple, walks the time-optimal path and adds
/// `flow` to every edge the path crosses, so many OD pairs' flows can be
/// aggregated onto a shared road-graph edge (the Overture-native analogue of
/// stplanr's `overline`/`overline2`). Disconnected pairs and unsnapped
/// (negative) node ids contribute their flow to the returned `dropped_flow`
/// total instead of any edge. Edges are returned sorted by descending total
/// flow.
type EdgeFlowMap = FxHashMap<(usize, usize), f64>;

pub fn batch_route_edge_flows(
    graph: &RoadGraph,
    from_nodes: &[i64],
    to_nodes: &[i64],
    flows: &[f64],
) -> (Vec<usize>, Vec<usize>, Vec<f64>, f64) {
    let threads = rayon::current_num_threads().max(1);
    let chunk_size = (from_nodes.len() / (threads * 4)).max(1);

    let (edge_maps, dropped): (Vec<EdgeFlowMap>, Vec<f64>) = from_nodes
        .par_chunks(chunk_size)
        .zip(to_nodes.par_chunks(chunk_size))
        .zip(flows.par_chunks(chunk_size))
        .map(|((from_chunk, to_chunk), flow_chunk)| {
            let mut calc = graph.new_calculator();
            let mut local: FxHashMap<(usize, usize), f64> = FxHashMap::default();
            let mut local_dropped = 0.0;
            for ((&from, &to), &flow) in from_chunk
                .iter()
                .zip(to_chunk.iter())
                .zip(flow_chunk.iter())
            {
                if from < 0 || to < 0 {
                    local_dropped += flow;
                    continue;
                }
                match graph.shortest_path(&mut calc, from as usize, to as usize) {
                    Some((_weight_ds, nodes)) => {
                        for w in nodes.windows(2) {
                            *local.entry((w[0], w[1])).or_insert(0.0) += flow;
                        }
                    }
                    None => {
                        local_dropped += flow;
                    }
                }
            }
            (local, local_dropped)
        })
        .unzip();

    let mut merged: FxHashMap<(usize, usize), f64> = FxHashMap::default();
    for m in edge_maps {
        for (k, v) in m {
            *merged.entry(k).or_insert(0.0) += v;
        }
    }
    let dropped_flow: f64 = dropped.iter().sum();

    let mut entries: Vec<((usize, usize), f64)> = merged.into_iter().collect();
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut edge_from_out = Vec::with_capacity(entries.len());
    let mut edge_to_out = Vec::with_capacity(entries.len());
    let mut flow_out = Vec::with_capacity(entries.len());
    for ((f, t), flow) in entries {
        edge_from_out.push(f);
        edge_to_out.push(t);
        flow_out.push(flow);
    }

    (edge_from_out, edge_to_out, flow_out, dropped_flow)
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

    #[test]
    fn batch_routes_returns_full_chain_geometry_under_cap() {
        let edge_from = vec![0, 1, 2];
        let edge_to = vec![1, 2, 3];
        let edge_weight = vec![10, 10, 10];
        let edge_length = vec![100.0, 100.0, 100.0];
        let node_lat = vec![0.0, 1.0, 2.0, 3.0];
        let node_lng = vec![0.0, 1.0, 2.0, 3.0];
        let graph = RoadGraph::build_with_length_and_coords(
            &edge_from,
            &edge_to,
            &edge_weight,
            &edge_length,
            &node_lat,
            &node_lng,
        );
        let (lats, lngs, cum, connected, starts, ends) = batch_road_routes(&graph, &[0], &[3], 10);
        assert_eq!(connected, vec![true]);
        assert_eq!(starts, vec![0]);
        assert_eq!(ends, vec![4]);
        assert_eq!(lats, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(lngs, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(cum, vec![0, 10, 20, 30]);
    }

    #[test]
    fn batch_routes_decimates_a_long_chain() {
        let n = 20;
        let edge_from: Vec<usize> = (0..n - 1).collect();
        let edge_to: Vec<usize> = (1..n).collect();
        let edge_weight = vec![10; n - 1];
        let edge_length = vec![100.0; n - 1];
        let node_lat: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let node_lng: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let graph = RoadGraph::build_with_length_and_coords(
            &edge_from,
            &edge_to,
            &edge_weight,
            &edge_length,
            &node_lat,
            &node_lng,
        );
        let (lats, lngs, cum, connected, starts, ends) =
            batch_road_routes(&graph, &[0], &[(n - 1) as i64], 5);
        assert_eq!(connected, vec![true]);
        let len = ends[0] - starts[0];
        assert!(len <= 5);
        assert_eq!(lats.first(), Some(&0.0));
        assert_eq!(lats.last(), Some(&((n - 1) as f64)));
        assert_eq!(lngs.first(), Some(&0.0));
        assert_eq!(cum.first(), Some(&0));
        assert_eq!(cum.last(), Some(&(((n - 1) * 10) as i64)));
    }

    #[test]
    fn batch_routes_reports_disconnected_and_unsnapped_as_zero_waypoints() {
        let edge_from = vec![0, 2];
        let edge_to = vec![1, 3];
        let edge_weight = vec![5, 5];
        let edge_length = vec![10.0, 10.0];
        let node_lat = vec![0.0, 1.0, 2.0, 3.0];
        let node_lng = vec![0.0, 1.0, 2.0, 3.0];
        let graph = RoadGraph::build_with_length_and_coords(
            &edge_from,
            &edge_to,
            &edge_weight,
            &edge_length,
            &node_lat,
            &node_lng,
        );
        let (lats, _lngs, _cum, connected, starts, ends) =
            batch_road_routes(&graph, &[0, -1], &[3, 2], 10);
        assert_eq!(connected, vec![false, false]);
        assert_eq!(starts, vec![0, 0]);
        assert_eq!(ends, vec![0, 0]);
        assert!(lats.is_empty());
    }

    #[test]
    fn batch_route_edge_flows_accumulates_along_shared_path() {
        // 0 -> 1 -> 2 -> 3 chain; queries (0,3), (1,3), (2,3) share
        // increasingly short suffixes of the chain, plus one unsnapped query
        // whose flow must land in `dropped_flow` instead of any edge.
        let edge_from = vec![0, 1, 2];
        let edge_to = vec![1, 2, 3];
        let edge_weight = vec![10, 10, 10];
        let graph = RoadGraph::build(&edge_from, &edge_to, &edge_weight);
        let (edge_from_out, edge_to_out, flow_out, dropped_flow) =
            batch_route_edge_flows(&graph, &[0, 1, 2, -1], &[3, 3, 3, 2], &[5.0, 3.0, 0.5, 1.0]);
        assert_eq!(edge_from_out, vec![2, 1, 0]);
        assert_eq!(edge_to_out, vec![3, 2, 1]);
        assert_eq!(flow_out, vec![8.5, 8.0, 5.0]);
        assert_eq!(dropped_flow, 1.0);
    }
}
