//! H3-grid-based sparse candidate graph for `stvd_emd`'s Sinkhorn solver.
//!
//! The dense path evaluates every `(i, j)` pair between two distributions.
//! At `reg=0.01` and cost values in metres, `exp(-cost/reg)` underflows to
//! *exactly* `0.0` in float32 for all but a point's closest few matches (see
//! `stvd_emd.rs`'s module docs), so the dense
//! computation is already discarding almost all of that work -- it just does
//! so by brute force. This module builds a much smaller candidate graph
//! instead: for each point, only pairs with plausibly-nearby points (found
//! via H3 grid-disk search) become graph edges, bounding both memory and the
//! sparse Sinkhorn solver's (`stvd_sparse_sinkhorn.rs`) per-iteration work to
//! `O((n+m) * avg_candidates_per_point)` instead of `O(n*m)`.

use h3o::{CellIndex, LatLng, Resolution};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::utils::haversine::haversine_km;

use super::stvd_emd::pair_cost_m;

/// Candidate-search knobs. Defaults are a starting point for empirical
/// tuning, not a final answer.
#[derive(Clone, Copy, Debug)]
pub struct CandidateGraphConfig {
    /// Internal H3 index resolution used purely for neighbor search --
    /// independent of whatever resolution (if any) produced the caller's
    /// `location_id`s, since this kernel only ever sees raw lat/lng.
    pub resolution: Resolution,
    /// Ring radius searched around each point's home cell.
    pub k_ring: u32,
    /// Hard cap for escalating `k_ring`, before falling back to a
    /// brute-force nearest-cell lookup for points that still found nothing.
    pub k_ring_max: u32,
    /// Minimum candidates to gather before stopping escalation. Stopping at
    /// the *first* nonempty ring (this field's earlier, simpler design) was
    /// measured to diverge sharply from the dense reference on sparse point
    /// distributions -- Sinkhorn's soft assignment genuinely spreads mass
    /// over several nearby candidates, not just whichever ring happened to
    /// contain the first match, so a small floor is needed even though the
    /// ring already found *something*.
    pub k_min: usize,
    /// Maximum candidates kept per point, after sorting by cost. The ring
    /// search only filters by *space*; a real `build_stvd`-shaped input can
    /// have many time-bin rows per location (up to 144) and several
    /// locations per H3 cell, so an unbounded spatial match can pull in
    /// hundreds of temporally-irrelevant rows (e.g. 3am matched against
    /// 3pm) before cost-based filtering ever gets a say. Keeping only the
    /// cheapest `max_candidates` bounds memory/compute directly, and drops
    /// exactly the rows that would already contribute ~0 to the dense
    /// computation's Gibbs kernel at typical `reg`.
    pub max_candidates: usize,
}

impl Default for CandidateGraphConfig {
    fn default() -> Self {
        Self { resolution: Resolution::Eight, k_ring: 2, k_ring_max: 32, k_min: 8, max_candidates: 64 }
    }
}

/// A directed bipartite candidate graph over `0..n` (distribution A) and
/// `0..m` (distribution B), stored as two CSR views of the same edge set so
/// the sparse Sinkhorn solver can iterate a row's (or column's) candidates
/// without touching the other `n*m - |edges|` pairs.
pub struct CandidateGraph {
    /// `offsets_a[i]..offsets_a[i+1]` indexes into `edges_a` for point `i`'s
    /// candidates on the B side, as `(b_index, cost_m)`.
    pub offsets_a: Vec<u32>,
    pub edges_a: Vec<(u32, f32)>,
    /// `offsets_b[j]..offsets_b[j+1]` indexes into `edges_b` for point `j`'s
    /// candidates on the A side, as `(a_index, cost_m)`.
    pub offsets_b: Vec<u32>,
    pub edges_b: Vec<(u32, f32)>,
}

impl CandidateGraph {
    pub fn edge_count(&self) -> usize {
        self.edges_a.len()
    }
}

/// Bucket each point of one distribution by its H3 cell. Points whose
/// coordinates don't form a valid `LatLng` (should not occur -- callers
/// validate finiteness upstream -- but handled defensively) are simply
/// omitted from the bucket map; they still get candidates via the
/// guaranteed-connectivity fallback in `nearby_candidates`, which works from
/// raw lat/lng rather than the bucket map.
fn bucket_by_cell(lats: &[f64], lngs: &[f64], resolution: Resolution) -> FxHashMap<u64, Vec<u32>> {
    let mut buckets: FxHashMap<u64, Vec<u32>> = FxHashMap::default();
    for (idx, (&lat, &lng)) in lats.iter().zip(lngs).enumerate() {
        if let Ok(ll) = LatLng::new(lat, lng) {
            let cell = u64::from(ll.to_cell(resolution));
            buckets.entry(cell).or_default().push(idx as u32);
        }
    }
    buckets
}

/// Haversine distance (km) from a raw point to an H3 cell's center, used
/// only by the fallback path to rank distinct far-side cells. `f64::INFINITY`
/// for a malformed cell index keeps the fallback's `min_by` total and simply
/// deprioritizes that cell rather than panicking.
fn cell_center_distance_km(lat: f64, lng: f64, cell: u64) -> f64 {
    match CellIndex::try_from(cell) {
        Ok(cell) => {
            let center = LatLng::from(cell);
            haversine_km(lat, lng, center.lat(), center.lng())
        }
        Err(_) => f64::INFINITY,
    }
}

/// Find `other`-side candidates for one point of `this` side: escalating
/// H3 ring search first, falling back to the single nearest distinct
/// far-side cell if the search still finds nothing by `k_ring_max` -- so
/// every point gets at least one candidate (see module docs: a point that
/// nobody nearby "prefers" can still be required to receive mass by its own
/// marginal constraint, so it must never end up with an empty candidate
/// set).
#[allow(clippy::too_many_arguments)]
fn nearby_candidates(
    home_lat: f64,
    home_lng: f64,
    home_t: f64,
    other_lats: &[f64],
    other_lngs: &[f64],
    other_ts: &[f64],
    other_buckets: &FxHashMap<u64, Vec<u32>>,
    other_distinct_cells: &[u64],
    cyclical_period: f64,
    r: f64,
    config: CandidateGraphConfig,
) -> Vec<(u32, f32)> {
    let mut found = Vec::new();
    let cost_to = |other_idx: u32| {
        let j = other_idx as usize;
        pair_cost_m(
            home_lat,
            home_lng,
            home_t,
            other_lats[j],
            other_lngs[j],
            other_ts[j],
            cyclical_period,
            r,
        )
    };

    if let Ok(home_cell) = LatLng::new(home_lat, home_lng).map(|ll| ll.to_cell(config.resolution)) {
        let mut k = config.k_ring.max(1);
        loop {
            found.clear();
            let ring: Vec<(CellIndex, u32)> = home_cell.grid_disk_distances(k);
            for (neighbor_cell, _distance) in ring {
                if let Some(indices) = other_buckets.get(&u64::from(neighbor_cell)) {
                    found.extend(indices.iter().map(|&idx| (idx, cost_to(idx))));
                }
            }
            if found.len() >= config.k_min || k >= config.k_ring_max {
                break;
            }
            k = (k * 2).min(config.k_ring_max);
        }
        if found.len() > config.max_candidates {
            found.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).expect("costs are always finite"));
            found.truncate(config.max_candidates);
        }
    }

    if found.is_empty()
        && let Some(&nearest_cell) = other_distinct_cells.iter().min_by(|&&a, &&b| {
            cell_center_distance_km(home_lat, home_lng, a)
                .partial_cmp(&cell_center_distance_km(home_lat, home_lng, b))
                .expect("distances are always finite or +inf, never NaN")
        })
        && let Some(indices) = other_buckets.get(&nearest_cell)
    {
        found.extend(indices.iter().map(|&idx| (idx, cost_to(idx))));
        if found.len() > config.max_candidates {
            found.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).expect("costs are always finite"));
            found.truncate(config.max_candidates);
        }
    }

    found
}

/// Build `offsets`/`edges` (grouped by `group_of`) from an edge list already
/// sorted by `(group_of, other)`. Mirrors `mean_area_volume_impl`'s
/// sort-then-linear-scan style in `../collective/stvd.rs`: cheap because the
/// input is already sorted, so grouping is one pass with no hashing.
fn build_csr(sorted_edges: &[(u32, u32, f32)], group_count: usize, group_of: impl Fn(&(u32, u32, f32)) -> u32, other_of: impl Fn(&(u32, u32, f32)) -> u32) -> (Vec<u32>, Vec<(u32, f32)>) {
    let mut offsets = vec![0u32; group_count + 1];
    for edge in sorted_edges {
        offsets[group_of(edge) as usize + 1] += 1;
    }
    for i in 0..group_count {
        offsets[i + 1] += offsets[i];
    }
    let values: Vec<(u32, f32)> = sorted_edges.iter().map(|edge| (other_of(edge), edge.2)).collect();
    (offsets, values)
}

/// Build the bidirectional candidate graph between distribution A (`n`
/// points) and distribution B (`m` points).
///
/// Searches both directions (every A point for nearby B points, and every B
/// point for nearby A points) and unions the results: a B point that no
/// nearby A point happens to prefer can still be required to receive mass
/// by its own marginal constraint, so a one-directional search could
/// silently strand it. Edges found from both directions are deduplicated
/// (they compute the identical cost, since `pair_cost_m` is a pure function
/// of the same six inputs regardless of search direction).
#[allow(clippy::too_many_arguments)]
pub fn build_candidate_graph(
    lats_a: &[f64],
    lngs_a: &[f64],
    ts_a: &[f64],
    lats_b: &[f64],
    lngs_b: &[f64],
    ts_b: &[f64],
    cyclical_period: f64,
    r: f64,
    config: CandidateGraphConfig,
) -> CandidateGraph {
    let n = lats_a.len();
    let m = lats_b.len();

    let buckets_a = bucket_by_cell(lats_a, lngs_a, config.resolution);
    let buckets_b = bucket_by_cell(lats_b, lngs_b, config.resolution);
    let distinct_cells_a: Vec<u64> = buckets_a.keys().copied().collect();
    let distinct_cells_b: Vec<u64> = buckets_b.keys().copied().collect();

    let edges_from_a: Vec<(u32, u32, f32)> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            nearby_candidates(
                lats_a[i], lngs_a[i], ts_a[i], lats_b, lngs_b, ts_b, &buckets_b, &distinct_cells_b,
                cyclical_period, r, config,
            )
            .into_iter()
            .map(move |(j, cost)| (i as u32, j, cost))
        })
        .collect();

    let edges_from_b: Vec<(u32, u32, f32)> = (0..m)
        .into_par_iter()
        .flat_map_iter(|j| {
            nearby_candidates(
                lats_b[j], lngs_b[j], ts_b[j], lats_a, lngs_a, ts_a, &buckets_a, &distinct_cells_a,
                cyclical_period, r, config,
            )
            .into_iter()
            .map(move |(i, cost)| (i, j as u32, cost))
        })
        .collect();

    let mut all_edges = edges_from_a;
    all_edges.extend(edges_from_b);

    // Dedup by (i, j): edges discovered from both directions collapse into
    // one (same cost either way -- pair_cost_m is deterministic in its
    // inputs). Sorted-then-adjacent-scan, same pattern as
    // mean_area_volume_impl's presence-entry dedup.
    all_edges.par_sort_unstable_by_key(|&(i, j, _)| (i, j));
    all_edges.dedup_by_key(|&mut (i, j, _)| (i, j));

    let (offsets_a, edges_a) = build_csr(&all_edges, n, |&(i, _, _)| i, |&(_, j, _)| j);

    let mut by_b = all_edges.clone();
    by_b.par_sort_unstable_by_key(|&(i, j, _)| (j, i));
    let (offsets_b, edges_b) = build_csr(&by_b, m, |&(_, j, _)| j, |&(i, _, _)| i);

    CandidateGraph { offsets_a, edges_a, offsets_b, edges_b }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> CandidateGraphConfig {
        CandidateGraphConfig::default()
    }

    #[test]
    fn every_point_has_at_least_one_candidate() {
        // Two clusters far enough apart that plain ring search at the
        // default k_ring wouldn't bridge them -- exercises the fallback.
        let lats_a = [37.7749, 37.7750, -33.8688];
        let lngs_a = [-122.4194, -122.4195, 151.2093];
        let ts_a = [480.0, 480.0, 480.0];
        let lats_b = [37.7751, 51.5074];
        let lngs_b = [-122.4196, -0.1278];
        let ts_b = [480.0, 480.0];

        let graph = build_candidate_graph(
            &lats_a, &lngs_a, &ts_a, &lats_b, &lngs_b, &ts_b, 1440.0, 10.0, default_config(),
        );

        for i in 0..lats_a.len() {
            let start = graph.offsets_a[i] as usize;
            let end = graph.offsets_a[i + 1] as usize;
            assert!(end > start, "point {i} of A has no candidates");
        }
        for j in 0..lats_b.len() {
            let start = graph.offsets_b[j] as usize;
            let end = graph.offsets_b[j + 1] as usize;
            assert!(end > start, "point {j} of B has no candidates");
        }
    }

    #[test]
    fn no_duplicate_edges_after_bidirectional_union() {
        let lats_a = [0.0, 0.0];
        let lngs_a = [0.0, 0.001];
        let ts_a = [480.0, 480.0];
        let lats_b = [0.0, 0.0];
        let lngs_b = [0.0005, 0.0015];
        let ts_b = [480.0, 480.0];

        let graph = build_candidate_graph(
            &lats_a, &lngs_a, &ts_a, &lats_b, &lngs_b, &ts_b, 1440.0, 10.0, default_config(),
        );

        let mut seen = rustc_hash::FxHashSet::default();
        for i in 0..lats_a.len() {
            let start = graph.offsets_a[i] as usize;
            let end = graph.offsets_a[i + 1] as usize;
            for &(j, _) in &graph.edges_a[start..end] {
                assert!(seen.insert((i as u32, j)), "duplicate edge ({i}, {j})");
            }
        }
    }

    #[test]
    fn by_a_and_by_b_views_agree_on_the_same_edge_set() {
        let lats_a = [0.0, 0.0, 10.0];
        let lngs_a = [0.0, 0.001, 10.0];
        let ts_a = [480.0, 500.0, 100.0];
        let lats_b = [0.0, 0.0];
        let lngs_b = [0.0005, 0.0015];
        let ts_b = [480.0, 500.0];

        let graph = build_candidate_graph(
            &lats_a, &lngs_a, &ts_a, &lats_b, &lngs_b, &ts_b, 1440.0, 10.0, default_config(),
        );

        let mut from_a: Vec<(u32, u32)> = Vec::new();
        for i in 0..lats_a.len() {
            let start = graph.offsets_a[i] as usize;
            let end = graph.offsets_a[i + 1] as usize;
            for &(j, _) in &graph.edges_a[start..end] {
                from_a.push((i as u32, j));
            }
        }
        let mut from_b: Vec<(u32, u32)> = Vec::new();
        for j in 0..lats_b.len() {
            let start = graph.offsets_b[j] as usize;
            let end = graph.offsets_b[j + 1] as usize;
            for &(i, _) in &graph.edges_b[start..end] {
                from_b.push((i, j as u32));
            }
        }
        from_a.sort_unstable();
        from_b.sort_unstable();
        assert_eq!(from_a, from_b);
    }

    #[test]
    fn candidate_costs_match_direct_pair_cost() {
        let lats_a = [0.0];
        let lngs_a = [0.0];
        let ts_a = [480.0];
        let lats_b = [0.0];
        let lngs_b = [0.001];
        let ts_b = [500.0];

        let graph = build_candidate_graph(
            &lats_a, &lngs_a, &ts_a, &lats_b, &lngs_b, &ts_b, 1440.0, 10.0, default_config(),
        );

        let expected = pair_cost_m(0.0, 0.0, 480.0, 0.0, 0.001, 500.0, 1440.0, 10.0);
        assert_eq!(graph.edges_a[0].1, expected);
    }

    #[test]
    fn empty_ring_search_falls_back_across_antipodal_points() {
        // Two points on opposite sides of the globe: ring search at
        // k_ring_max never bridges them, forcing the brute-force fallback.
        let lats_a = [0.0];
        let lngs_a = [0.0];
        let ts_a = [0.0];
        let lats_b = [0.0];
        let lngs_b = [179.9];
        let ts_b = [0.0];

        let graph = build_candidate_graph(
            &lats_a, &lngs_a, &ts_a, &lats_b, &lngs_b, &ts_b, 1440.0, 10.0, default_config(),
        );

        assert_eq!(graph.edge_count(), 1);
        assert!(graph.edges_a[0].1.is_finite());
    }
}
