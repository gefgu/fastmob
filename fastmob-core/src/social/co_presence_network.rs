//! RECAST temporal-contact construction, paper RND, and validation kernels.
//!
//! The classifier follows Section 5 of Vaz de Melo et al. (2015).  Event
//! graphs are UTC-day snapshots; RND samples each possible pair in a snapshot
//! with the paper's degree-product probability, and T-RND applies RND to every
//! snapshot independently.

#[cfg(feature = "numkong")]
use numkong::SparseIntersect;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

fn profile(stage: &str, start: Instant, edges: usize) {
    if std::env::var_os("FASTMOB_RECAST_PROFILE").is_some() {
        eprintln!(
            "recast {stage} seconds={:.6} edges={edges}",
            start.elapsed().as_secs_f64()
        );
    }
}

type Edge = (u32, u32);
const DAY_MS: i64 = 86_400_000;

fn edge(a: u32, b: u32) -> Edge {
    if a < b { (a, b) } else { (b, a) }
}
fn stream_seed(seed: u64, replica: usize, window: usize) -> u64 {
    seed ^ (replica as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (window as u64).wrapping_mul(0xD1B5_4A32_D192_ED03)
}

#[derive(Clone)]
struct Interval {
    user: u32,
    start: i64,
    end: i64,
}

#[derive(Clone)]
pub struct TemporalEvents {
    pub events: Vec<FxHashSet<Edge>>,
    pub window_starts_ms: Vec<i64>,
}

/// Build UTC-day event graphs. Contacts require the supplied strict overlap.
pub fn event_graphs(
    users: &[u32],
    locations: &[u32],
    starts: &[i64],
    ends: &[i64],
    min_contact_ms: i64,
) -> TemporalEvents {
    if users.is_empty() {
        return TemporalEvents {
            events: Vec::new(),
            window_starts_ms: Vec::new(),
        };
    }
    let min_window = starts.iter().copied().min().unwrap().div_euclid(DAY_MS);
    let max_window = ends
        .iter()
        .copied()
        .filter(|&x| x > i64::MIN)
        .map(|x| (x - 1).div_euclid(DAY_MS))
        .max()
        .unwrap_or(min_window);
    let steps = (max_window - min_window + 1).max(0) as usize;
    let mut groups: FxHashMap<(i64, u32), Vec<Interval>> = FxHashMap::default();
    for i in 0..users.len() {
        if ends[i] <= starts[i] {
            continue;
        }
        for window in starts[i].div_euclid(DAY_MS)..=(ends[i] - 1).div_euclid(DAY_MS) {
            let lo = starts[i].max(window * DAY_MS);
            let hi = ends[i].min((window + 1) * DAY_MS);
            if hi - lo >= min_contact_ms {
                groups
                    .entry((window, locations[i]))
                    .or_default()
                    .push(Interval {
                        user: users[i],
                        start: lo,
                        end: hi,
                    });
            }
        }
    }
    // Buckets are independent once intervals have been assigned. Sort first so
    // parallel scheduling cannot affect the subsequent per-day merge order.
    let mut buckets: Vec<_> = groups.into_iter().collect();
    buckets.sort_unstable_by_key(|((window, location), _)| (*window, *location));
    let bucket_edges: Vec<(usize, Vec<Edge>)> = buckets
        .into_par_iter()
        .map(|((window, _), mut intervals)| {
            intervals.sort_unstable_by_key(|x| (x.start, x.end, x.user));
            let mut expiry = BinaryHeap::new();
            let mut active: FxHashMap<u32, usize> = FxHashMap::default();
            let mut edges = FxHashSet::default();
            for current in intervals {
                // All retained intervals are long enough. Since starts are
                // sorted, an active user qualifies exactly when some end is
                // >= current.start + min_contact_ms. Keep one count per user
                // so repeated overlapping stays don't repeat pair work.
                let required_end = current.start + min_contact_ms;
                while let Some(&Reverse((end, user))) = expiry.peek() {
                    if end >= required_end {
                        break;
                    }
                    expiry.pop();
                    let count = active.get_mut(&user).unwrap();
                    *count -= 1;
                    if *count == 0 {
                        active.remove(&user);
                    }
                }
                for &user in active.keys() {
                    if user != current.user {
                        edges.insert(edge(user, current.user));
                    }
                }
                *active.entry(current.user).or_insert(0) += 1;
                expiry.push(Reverse((current.end, current.user)));
            }
            let mut edges: Vec<_> = edges.into_iter().collect();
            edges.sort_unstable();
            ((window - min_window) as usize, edges)
        })
        .collect();
    let mut events: Vec<FxHashSet<Edge>> = (0..steps).map(|_| FxHashSet::default()).collect();
    for (window, edges) in bucket_edges {
        events[window].extend(edges);
    }
    TemporalEvents {
        events,
        window_starts_ms: (0..steps)
            .map(|i| (min_window + i as i64) * DAY_MS)
            .collect(),
    }
}

pub struct TemporalGraphOutput {
    pub window_starts_ms: Vec<i64>,
    pub edge_offsets: Vec<u64>,
    pub edge_from: Vec<u32>,
    pub edge_to: Vec<u32>,
}
pub fn flatten_events(temporal: &TemporalEvents) -> TemporalGraphOutput {
    let mut offsets = Vec::with_capacity(temporal.events.len() + 1);
    let mut from = Vec::new();
    let mut to = Vec::new();
    offsets.push(0);
    for event in &temporal.events {
        let mut edges: Vec<_> = event.iter().copied().collect();
        edges.sort_unstable();
        for (u, v) in edges {
            from.push(u);
            to.push(v);
        }
        offsets.push(from.len() as u64);
    }
    TemporalGraphOutput {
        window_starts_ms: temporal.window_starts_ms.clone(),
        edge_offsets: offsets,
        edge_from: from,
        edge_to: to,
    }
}

fn aggregate_from_counts(
    counts: FxHashMap<Edge, usize>,
    steps: usize,
) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let mut values: Vec<_> = counts.into_iter().collect();
    values.sort_unstable_by_key(|(e, _)| *e);
    let denom = steps.max(1) as f64;
    let mut from = Vec::with_capacity(values.len());
    let mut to = Vec::with_capacity(values.len());
    let mut persistence = Vec::with_capacity(values.len());
    for ((u, v), n) in values {
        from.push(u);
        to.push(v);
        persistence.push((n as f64 / denom) as f32);
    }
    (from, to, persistence)
}

fn aggregate_serial(events: &[FxHashSet<Edge>]) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let mut counts: FxHashMap<Edge, usize> = FxHashMap::default();
    for event in events {
        for &e in event {
            *counts.entry(e).or_insert(0) += 1;
        }
    }
    aggregate_from_counts(counts, events.len())
}

fn aggregate(events: &[FxHashSet<Edge>]) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
    let counts = events
        .par_iter()
        .fold(FxHashMap::default, |mut counts, event| {
            for &e in event {
                *counts.entry(e).or_insert(0) += 1;
            }
            counts
        })
        .reduce(FxHashMap::default, |mut left, right| {
            for (edge, count) in right {
                *left.entry(edge).or_insert(0) += count;
            }
            left
        });
    aggregate_from_counts(counts, events.len())
}

fn build_adjacency(node_count: usize, from: &[u32], to: &[u32]) -> Vec<Vec<u32>> {
    let mut adjacency = vec![Vec::new(); node_count];
    for i in 0..from.len() {
        adjacency[from[i] as usize].push(to[i]);
        adjacency[to[i] as usize].push(from[i]);
    }
    adjacency.par_iter_mut().for_each(|v| {
        v.sort_unstable();
        v.dedup();
    });
    adjacency
}
fn build_adjacency_serial(node_count: usize, from: &[u32], to: &[u32]) -> Vec<Vec<u32>> {
    let mut adjacency = vec![Vec::new(); node_count];
    for i in 0..from.len() {
        adjacency[from[i] as usize].push(to[i]);
        adjacency[to[i] as usize].push(from[i]);
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    adjacency
}
// perf (10M-row RECAST, `perf record`/`perf report`) showed this loop at
// ~35-40% of total cycles -- the single dominant cost in the whole RECAST
// classifier. `a`/`b` are node adjacency lists from `build_adjacency_csr`,
// which are already sorted and duplicate-free (each edge contributes at most
// one entry per side), satisfying numkong's `SparseIntersect` precondition.
// When the `numkong` feature is enabled (the project default), this calls
// its SIMD sorted-set intersection -- on this host (Ice Lake-SP) that's a
// real AVX-512 VP2INTERSECT-style block intersection, not just a compiler
// hint. `#[cfg(not(feature = "numkong"))]` below keeps the branchless
// scalar version as a correctness-equivalent fallback for builds where
// numkong is disabled (e.g. CI wheel builds avoiding its manylinux/
// musllinux cross-build friction, see fastmob-core/Cargo.toml).
#[inline(always)]
fn count_common(a: &[u32], b: &[u32]) -> usize {
    #[cfg(feature = "numkong")]
    {
        u32::sparse_intersection_size(a, b)
    }
    #[cfg(not(feature = "numkong"))]
    {
        // `perf stat -e branches,branch-misses` measured only a 0.80%
        // misprediction rate on the if/else version -- not the dominant
        // cost, but real (~3.5% of total cycles by rough estimate:
        // mispredicts * ~17-cycle penalty / total cycles). Resolving each
        // comparison to a 0/1 value first and using that directly for the
        // advance amount (instead of branching on it) compiles to
        // `setcc`+add rather than conditional jumps, removing that
        // misprediction cost outright. Same three-way outcome per
        // iteration as a branchy if/else: advance `i`, advance `j`, or
        // advance both and count a match.
        let (mut i, mut j, mut n) = (0, 0, 0);
        while i < a.len() && j < b.len() {
            let x = a[i];
            let y = b[j];
            n += (x == y) as usize;
            i += (x <= y) as usize;
            j += (x >= y) as usize;
        }
        n
    }
}
/// Flat CSR adjacency: `flat[offsets[u]..offsets[u+1]]` is node `u`'s sorted,
/// deduplicated neighbor list. Unlike `Vec<Vec<u32>>` (one heap allocation per
/// node), this is one contiguous buffer, which matters a lot here: `overlaps`
/// below does millions of neighbor-list lookups at arbitrary node indices, and
/// a flat buffer turns each lookup from "chase a pointer to a scattered heap
/// block" into "one offset lookup into a buffer that's already resident."
/// Sorting each node's slice is comparatively cheap next to that, so it stays
/// a plain serial loop rather than adding unsafe disjoint-slice parallelism.
fn build_adjacency_csr(node_count: usize, from: &[u32], to: &[u32]) -> (Vec<usize>, Vec<u32>) {
    let mut degree = vec![0usize; node_count];
    for i in 0..from.len() {
        degree[from[i] as usize] += 1;
        degree[to[i] as usize] += 1;
    }
    let mut offsets = vec![0usize; node_count + 1];
    for i in 0..node_count {
        offsets[i + 1] = offsets[i] + degree[i];
    }
    let mut flat = vec![0u32; offsets[node_count] as usize];
    let mut cursor = offsets.clone();
    for i in 0..from.len() {
        let (u, v) = (from[i], to[i]);
        flat[cursor[u as usize] as usize] = v;
        cursor[u as usize] += 1;
        flat[cursor[v as usize] as usize] = u;
        cursor[v as usize] += 1;
    }
    // `from`/`to` are already unique edges, so each side contributes at most
    // one entry per neighbor -- no duplicates possible, sort only.
    for i in 0..node_count {
        let s = offsets[i] as usize;
        let e = offsets[i + 1] as usize;
        flat[s..e].sort_unstable();
    }
    (offsets, flat)
}
fn overlaps(node_count: usize, from: &[u32], to: &[u32]) -> Vec<f32> {
    let (offsets, flat) = build_adjacency_csr(node_count, from, to);
    // Dense neighborhoods are cheaper to intersect wordwise. Sparse lists stay
    // in CSR; this also avoids an O(n^2) allocation for large sparse graphs.
    let words = node_count.div_ceil(64);
    let mut bit_offsets = vec![usize::MAX; node_count];
    let mut bit_words = 0usize;
    for u in 0..node_count {
        if offsets[u + 1] - offsets[u] >= words.saturating_mul(2).max(64)
            && bit_words.saturating_add(words) <= 512 * 1024 * 1024
        {
            bit_offsets[u] = bit_words;
            bit_words += words;
        }
    }
    let mut bits = vec![0u64; bit_words];
    for u in 0..node_count {
        if bit_offsets[u] != usize::MAX {
            for &v in &flat[offsets[u]..offsets[u + 1]] {
                bits[bit_offsets[u] + v as usize / 64] |= 1u64 << (v % 64);
            }
        }
    }
    (0..from.len())
        .into_par_iter()
        .map(|i| {
            let a =
                &flat[offsets[from[i] as usize] as usize..offsets[from[i] as usize + 1] as usize];
            let b = &flat[offsets[to[i] as usize] as usize..offsets[to[i] as usize + 1] as usize];
            let (ao, bo) = (bit_offsets[from[i] as usize], bit_offsets[to[i] as usize]);
            let inter = match (ao != usize::MAX, bo != usize::MAX) {
                (true, true) => bits[ao..ao + words]
                    .iter()
                    .zip(&bits[bo..bo + words])
                    .map(|(&x, &y)| (x & y).count_ones() as usize)
                    .sum(),
                (true, false) => b
                    .iter()
                    .filter(|&&v| bits[ao + v as usize / 64] & (1u64 << (v % 64)) != 0)
                    .count(),
                (false, true) => a
                    .iter()
                    .filter(|&&v| bits[bo + v as usize / 64] & (1u64 << (v % 64)) != 0)
                    .count(),
                (false, false) => count_common(a, b),
            };
            let union = a.len() + b.len() - inter;
            if union == 0 {
                0.0
            } else {
                inter as f32 / union as f32
            }
        })
        .collect()
}
fn overlaps_serial(node_count: usize, from: &[u32], to: &[u32]) -> Vec<f32> {
    let (offsets, flat) = build_adjacency_csr(node_count, from, to);
    (0..from.len())
        .map(|i| {
            let a =
                &flat[offsets[from[i] as usize] as usize..offsets[from[i] as usize + 1] as usize];
            let b = &flat[offsets[to[i] as usize] as usize..offsets[to[i] as usize + 1] as usize];
            let inter = count_common(a, b);
            let union = a.len() + b.len() - inter;
            if union == 0 {
                0.0
            } else {
                inter as f32 / union as f32
            }
        })
        .collect()
}

/// Sorts node indices `0..degree.len()` by descending degree using counting
/// sort: O(n + max_degree) rather than a comparison sort's O(n log n), and
/// degree values here are small bounded integers (bounded by how many
/// distinct contacts one node can have within a single day's event graph),
/// so this is both asymptotically cheaper and touches `degree` sequentially
/// instead of through a comparator's random-access indirection. Tie order
/// among equal-degree nodes doesn't matter: `sample_expected_degree_edges`
/// only relies on probabilities being non-increasing along the scan, and
/// equal-degree nodes are statistically exchangeable.
fn sort_nodes_by_degree_desc(degree: &[usize]) -> Vec<u32> {
    let n = degree.len();
    let max_degree = degree.iter().copied().max().unwrap_or(0);
    let mut starts = vec![0u32; max_degree + 1];
    for &d in degree {
        starts[d] += 1;
    }
    let mut cumulative = 0u32;
    for d in (0..=max_degree).rev() {
        let count = starts[d];
        starts[d] = cumulative;
        cumulative += count;
    }
    let mut order = vec![0u32; n];
    for (i, &d) in degree.iter().enumerate() {
        order[starts[d] as usize] = i as u32;
        starts[d] += 1;
    }
    order
}

/// Miller & Hagberg (2011) fast expected-degree ("Chung-Lu") sampler: emits
/// pair (u, v) independently with probability min(degree[u]*degree[v]/sum, 1)
/// -- exactly the distribution the naive `for u { for v in (u+1).. {
/// rng.gen_bool(p) } }` loop draws from -- but in O(node_count + edges
/// emitted) expected time instead of O(node_count^2). A degree-product null
/// model visits every possible pair regardless of how sparse the *actual*
/// contact graph is, which dominated RECAST's wall time at realistic
/// node_count (see git history for the perf profile that motivated this).
///
/// The trick: process nodes in descending-degree order, so for a fixed row
/// `u` the per-pair probability is non-increasing as the scan moves right.
/// That lets a geometric-distributed jump land directly on the next
/// candidate `v` at a rate bounded by the *previous* landed pair's true
/// probability `p` (valid since probabilities only shrink from there), then
/// thin that candidate via acceptance-rejection against its own true
/// probability `q` -- reproducing independent per-pair Bernoulli(p_uv)
/// draws without visiting the (typically overwhelming) majority of pairs
/// that would have been rejected anyway. `emit` receives original node ids,
/// not sorted positions, and is called once per accepted pair with u < v
/// already guaranteed false -- callers must canonicalize if they need u < v.
fn sample_expected_degree_edges(
    node_count: usize,
    degree: &[usize],
    sum: usize,
    rng: &mut Xoshiro256PlusPlus,
    emit: impl FnMut(u32, u32),
) {
    if sum == 0 || node_count < 2 {
        return;
    }
    let order = sort_nodes_by_degree_desc(degree);
    let seq: Vec<f64> = order.iter().map(|&i| degree[i as usize] as f64).collect();
    sample_ordered_edges(&order, &seq, sum, rng, emit);
}

fn sample_ordered_edges(
    order: &[u32],
    seq: &[f64],
    sum: usize,
    rng: &mut Xoshiro256PlusPlus,
    mut emit: impl FnMut(u32, u32),
) {
    if sum == 0 {
        return;
    }
    let rho = 1.0 / sum as f64;
    let n = order.len();
    for u in 0..n {
        let mut v = u + 1;
        if v >= n {
            continue;
        }
        let factor = seq[u] * rho;
        let mut p = (seq[v] * factor).min(1.0);
        while v < n && p > 0.0 {
            if p != 1.0 {
                let r: f64 = rng.r#gen();
                let jump = r.ln() / (1.0 - p).ln();
                if !jump.is_finite() {
                    break;
                }
                v += jump.floor() as usize;
            }
            if v < n {
                let q = (seq[v] * factor).min(1.0);
                if rng.r#gen::<f64>() < q / p {
                    emit(order[u], order[v]);
                }
                v += 1;
                p = q;
            }
        }
    }
}

/// Paper RND: sample pair (i,j) with p_ij = d_i*d_j / sum_k d_k.
/// As in the paper, this preserves the degree distribution in expectation,
/// rather than preserving every realization's degrees exactly.
pub fn rnd(
    node_count: usize,
    event: &FxHashSet<Edge>,
    rng: &mut Xoshiro256PlusPlus,
) -> FxHashSet<Edge> {
    let mut degree = vec![0usize; node_count];
    for &(u, v) in event {
        degree[u as usize] += 1;
        degree[v as usize] += 1;
    }
    let sum: usize = degree.iter().sum();
    let mut result = FxHashSet::default();
    sample_expected_degree_edges(node_count, &degree, sum, rng, |a, b| {
        result.insert(edge(a, b));
    });
    result
}

pub fn t_rnd(
    node_count: usize,
    temporal: &TemporalEvents,
    replica: usize,
    seed: u64,
) -> TemporalEvents {
    let events = temporal
        .events
        .par_iter()
        .enumerate()
        .map(|(window, event)| {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(stream_seed(seed, replica, window));
            rnd(node_count, event, &mut rng)
        })
        .collect();
    TemporalEvents {
        events,
        window_starts_ms: temporal.window_starts_ms.clone(),
    }
}

/// Standalone paper-RND generator for one simple event graph.
pub fn rnd_graph(
    node_count: usize,
    edge_from: &[u32],
    edge_to: &[u32],
    seed: u64,
) -> (Vec<u32>, Vec<u32>) {
    let event: FxHashSet<Edge> = edge_from
        .iter()
        .zip(edge_to)
        .filter_map(|(&u, &v)| (u != v).then_some(edge(u, v)))
        .collect();
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut values: Vec<_> = rnd(node_count, &event, &mut rng).into_iter().collect();
    values.sort_unstable();
    values.into_iter().unzip()
}

/// Standalone T-RND generator for a flattened temporal graph.
pub fn t_rnd_graph(
    node_count: usize,
    window_starts_ms: &[i64],
    edge_offsets: &[u64],
    edge_from: &[u32],
    edge_to: &[u32],
    replica: usize,
    seed: u64,
) -> TemporalGraphOutput {
    let events = (0..window_starts_ms.len())
        .map(|i| {
            let lo = edge_offsets[i] as usize;
            let hi = edge_offsets[i + 1] as usize;
            edge_from[lo..hi]
                .iter()
                .zip(&edge_to[lo..hi])
                .filter_map(|(&u, &v)| (u != v).then_some(edge(u, v)))
                .collect()
        })
        .collect();
    flatten_events(&t_rnd(
        node_count,
        &TemporalEvents {
            events,
            window_starts_ms: window_starts_ms.to_vec(),
        },
        replica,
        seed,
    ))
}

fn threshold(values: &mut [f32], p_rnd: f64) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values[(((values.len() as f64) * (1.0 - p_rnd)).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1)] as f64
}
fn relationship_class(p: f32, o: f32, pt: f64, ot: f64) -> u8 {
    match (p as f64 > pt, o as f64 > ot) {
        (true, true) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (false, false) => 3,
    }
}

pub struct RecastOutput {
    pub edge_from: Vec<u32>,
    pub edge_to: Vec<u32>,
    pub persistence: Vec<f32>,
    pub topological_overlap: Vec<f32>,
    pub classes: Vec<u8>,
    pub persistence_threshold: f64,
    pub overlap_threshold: f64,
    pub time_steps: usize,
}
/// Generate all replica/window random graphs in one flat Rayon stage.  The
/// result is reassembled in replica/window order, independent of scheduling.
fn random_replicas(
    node_count: usize,
    temporal: &TemporalEvents,
    replicas: usize,
    seed: u64,
) -> Vec<TemporalEvents> {
    let replicas = replicas.max(1);
    let windows = temporal.events.len();
    if windows == 0 {
        return (0..replicas)
            .map(|_| TemporalEvents {
                events: Vec::new(),
                window_starts_ms: Vec::new(),
            })
            .collect();
    }
    let generated: Vec<FxHashSet<Edge>> = (0..replicas * windows)
        .into_par_iter()
        .map(|index| {
            let replica = index / windows;
            let window = index % windows;
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(stream_seed(seed, replica, window));
            rnd(node_count, &temporal.events[window], &mut rng)
        })
        .collect();
    generated
        .chunks(windows)
        .map(|events| TemporalEvents {
            events: events.to_vec(),
            window_starts_ms: temporal.window_starts_ms.clone(),
        })
        .collect()
}

fn classify_events_with_randoms(
    node_count: usize,
    temporal: &TemporalEvents,
    p_rnd: f64,
    randoms: &[TemporalEvents],
) -> (RecastOutput, Vec<f32>, Vec<f32>) {
    let (edge_from, edge_to, persistence) = aggregate(&temporal.events);
    let time_steps = temporal.events.len();
    if edge_from.is_empty() {
        return (
            RecastOutput {
                edge_from,
                edge_to,
                persistence,
                topological_overlap: Vec::new(),
                classes: Vec::new(),
                persistence_threshold: 1.0,
                overlap_threshold: 1.0,
                time_steps,
            },
            Vec::new(),
            Vec::new(),
        );
    }
    let topological_overlap = overlaps(node_count, &edge_from, &edge_to);
    // Replica-level tasks own their aggregation and serial overlap calculation;
    // this avoids nesting Rayon work beneath the flat replica/window stage.
    let null_metrics: Vec<_> = randoms
        .par_iter()
        .map(|random| {
            let (rf, rt, rp) = aggregate_serial(&random.events);
            let overlap = overlaps_serial(node_count, &rf, &rt);
            (rp, overlap)
        })
        .collect();
    let mut null_persistence = Vec::new();
    let mut null_overlap = Vec::new();
    for (persistence, overlap) in null_metrics {
        null_persistence.extend(persistence);
        null_overlap.extend(overlap);
    }
    let persistence_threshold = threshold(&mut null_persistence, p_rnd);
    let overlap_threshold = threshold(&mut null_overlap, p_rnd);
    let classes = persistence
        .iter()
        .zip(&topological_overlap)
        .map(|(&p, &o)| relationship_class(p, o, persistence_threshold, overlap_threshold))
        .collect();
    (
        RecastOutput {
            edge_from,
            edge_to,
            persistence,
            topological_overlap,
            classes,
            persistence_threshold,
            overlap_threshold,
            time_steps,
        },
        null_persistence,
        null_overlap,
    )
}

/// Exact tail selection: retain enough values for any realized edge count,
/// then partition instead of sorting the full null distribution.
struct NullTail {
    values: Vec<f32>,
    total: usize,
    keep: usize,
    upper: bool,
}
impl NullTail {
    fn new(max_values: usize, p: f64) -> Self {
        let upper = p <= 0.5;
        let fraction = if upper { p } else { 1.0 - p };
        let keep = ((max_values as f64 * fraction).ceil() as usize).saturating_add(2);
        Self {
            values: Vec::new(),
            total: 0,
            keep,
            upper,
        }
    }
    fn extend(&mut self, values: &[f32]) {
        self.total += values.len();
        // Bound transient storage as well as retained storage.
        for chunk in values.chunks(65_536) {
            self.values.extend_from_slice(chunk);
            if self.values.len() > self.keep.saturating_mul(2).max(65_536) {
                self.trim();
            }
        }
    }
    fn trim(&mut self) {
        if self.values.len() > self.keep {
            let upper = self.upper;
            self.values.select_nth_unstable_by(self.keep, |a, b| {
                if upper {
                    b.total_cmp(a)
                } else {
                    a.total_cmp(b)
                }
            });
            self.values.truncate(self.keep);
        }
    }
    fn threshold(&mut self, p: f64) -> f64 {
        if self.total == 0 {
            return 1.0;
        }
        self.trim();
        let rank = ((self.total as f64 * (1.0 - p)).ceil() as usize)
            .saturating_sub(1)
            .min(self.total - 1);
        let index = if self.upper {
            self.total - 1 - rank
        } else {
            rank
        };
        let upper = self.upper;
        *self
            .values
            .select_nth_unstable_by(index, |a, b| {
                if upper {
                    b.total_cmp(a)
                } else {
                    a.total_cmp(b)
                }
            })
            .1 as f64
    }

    fn merge(&mut self, partial: NullTail) {
        self.total += partial.total;
        self.values.extend(partial.values);
        if self.values.len() > self.keep.saturating_mul(2).max(65_536) {
            self.trim();
        }
    }
}

fn histogram_threshold(histogram: &FxHashMap<u32, usize>, p: f64) -> f64 {
    let total: usize = histogram.values().sum();
    if total == 0 {
        return 1.0;
    }
    let rank = ((total as f64 * (1.0 - p)).ceil() as usize)
        .saturating_sub(1)
        .min(total - 1);
    let mut bins: Vec<_> = histogram
        .iter()
        .map(|(&bits, &count)| (f32::from_bits(bits), count))
        .collect();
    bins.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut cumulative = 0;
    for (value, count) in bins {
        cumulative += count;
        if cumulative > rank {
            return value as f64;
        }
    }
    unreachable!()
}

/// Only one replica's aggregate and overlap are resident at a time.
fn classify_events_streaming(
    node_count: usize,
    temporal: TemporalEvents,
    p_rnd: f64,
    replicas: usize,
    seed: u64,
) -> RecastOutput {
    let started = Instant::now();
    let (edge_from, edge_to, persistence) = aggregate(&temporal.events);
    profile("observed_aggregate", started, edge_from.len());
    let time_steps = temporal.events.len();
    if edge_from.is_empty() {
        return RecastOutput {
            edge_from,
            edge_to,
            persistence,
            topological_overlap: Vec::new(),
            classes: Vec::new(),
            persistence_threshold: 1.0,
            overlap_threshold: 1.0,
            time_steps,
        };
    }
    let started = Instant::now();
    let sampling: Vec<_> = temporal
        .events
        .par_iter()
        .map(|event| {
            let mut degree = vec![0usize; node_count];
            for &(a, b) in event {
                degree[a as usize] += 1;
                degree[b as usize] += 1;
            }
            let sum = degree.iter().sum::<usize>();
            let order = sort_nodes_by_degree_desc(&degree);
            let seq: Vec<f64> = order.iter().map(|&u| degree[u as usize] as f64).collect();
            (order, seq, sum)
        })
        .collect();
    // The null sampler needs degrees, not the original event edge sets.
    drop(temporal);
    profile("sampling_metadata", started, edge_from.len());
    let started = Instant::now();
    let topological_overlap = overlaps(node_count, &edge_from, &edge_to);
    profile("observed_overlap", started, edge_from.len());
    let replicas = replicas.max(1);
    let max_values = node_count.saturating_mul(node_count.saturating_sub(1)) / 2;
    let mut null_overlap = NullTail::new(max_values.saturating_mul(replicas), p_rnd);
    let mut persistence_histogram = FxHashMap::default();
    // Replica streams are independent. Fixed sub-pools use all CPU cores but
    // return only threshold summaries, so no worker retains another replica's
    // random graph.
    let workers_per_replica = rayon::current_num_threads().div_ceil(replicas).max(1);
    thread::scope(|scope| {
        let (sender, receiver) = mpsc::channel();
        for replica in 0..replicas {
            let sender = sender.clone();
            let sampling = &sampling;
            scope.spawn(move || {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(workers_per_replica)
                    .build()
                    .unwrap();
                let started = Instant::now();
                let counts = pool.install(|| {
                    sampling
                        .par_iter()
                        .enumerate()
                        .fold(
                            FxHashMap::default,
                            |mut counts, (window, (order, seq, sum))| {
                                let mut rng = Xoshiro256PlusPlus::seed_from_u64(stream_seed(
                                    seed, replica, window,
                                ));
                                sample_ordered_edges(order, seq, *sum, &mut rng, |a, b| {
                                    *counts.entry(edge(a, b)).or_insert(0usize) += 1;
                                });
                                counts
                            },
                        )
                        .reduce(FxHashMap::default, |mut left, right| {
                            for (edge, count) in right {
                                *left.entry(edge).or_insert(0) += count;
                            }
                            left
                        })
                });
                profile("null_sample_aggregate", started, counts.len());
                let started = Instant::now();
                let (from, to, persistence) = aggregate_from_counts(counts, time_steps);
                profile("null_order", started, from.len());
                let started = Instant::now();
                let overlap = pool.install(|| overlaps(node_count, &from, &to));
                profile("null_overlap", started, from.len());
                let started = Instant::now();
                let mut histogram = FxHashMap::default();
                for value in persistence {
                    *histogram.entry(value.to_bits()).or_insert(0usize) += 1;
                }
                let mut tail = NullTail::new(max_values.saturating_mul(replicas), p_rnd);
                tail.extend(&overlap);
                profile("null_tail", started, from.len());
                sender.send((histogram, tail)).unwrap();
            });
        }
        drop(sender);
        for (histogram, tail) in receiver {
            for (value, count) in histogram {
                *persistence_histogram.entry(value).or_insert(0usize) += count;
            }
            null_overlap.merge(tail);
        }
    });
    let started = Instant::now();
    let persistence_threshold = histogram_threshold(&persistence_histogram, p_rnd);
    let overlap_threshold = null_overlap.threshold(p_rnd);
    let classes = persistence
        .iter()
        .zip(&topological_overlap)
        .map(|(&p, &o)| relationship_class(p, o, persistence_threshold, overlap_threshold))
        .collect();
    profile("thresholds_and_classes", started, edge_from.len());
    RecastOutput {
        edge_from,
        edge_to,
        persistence,
        topological_overlap,
        classes,
        persistence_threshold,
        overlap_threshold,
        time_steps,
    }
}

/// Faithful RECAST classification. Class codes: 0 Friends, 1 Bridges, 2 Acquaintances, 3 Random.
#[allow(clippy::too_many_arguments)]
pub fn recast_classify(
    node_count: usize,
    users: &[u32],
    locations: &[u32],
    starts: &[i64],
    ends: &[i64],
    min_contact_ms: i64,
    p_rnd: f64,
    replicas: usize,
    seed: u64,
) -> RecastOutput {
    let started = Instant::now();
    let temporal = event_graphs(users, locations, starts, ends, min_contact_ms);
    profile(
        "events",
        started,
        temporal.events.iter().map(|x| x.len()).sum(),
    );
    classify_events_streaming(node_count, temporal, p_rnd, replicas, seed)
}

fn average_clustering(node_count: usize, event: &FxHashSet<Edge>) -> f64 {
    if node_count == 0 {
        return 0.0;
    }
    let (from, to): (Vec<_>, Vec<_>) = event.iter().copied().unzip();
    let adjacency = build_adjacency(node_count, &from, &to);
    adjacency
        .par_iter()
        .map(|neighbors| {
            if neighbors.len() < 2 {
                return 0.0;
            }
            let mut links = 0usize;
            for i in 0..neighbors.len() {
                for &v in &neighbors[(i + 1)..] {
                    if adjacency[neighbors[i] as usize].binary_search(&v).is_ok() {
                        links += 1;
                    }
                }
            }
            2.0 * links as f64 / (neighbors.len() * (neighbors.len() - 1)) as f64
        })
        .sum::<f64>()
        / node_count as f64
}
fn cumulative_clustering(node_count: usize, temporal: &TemporalEvents) -> Vec<f64> {
    let mut aggregate = FxHashSet::default();
    temporal
        .events
        .iter()
        .map(|event| {
            aggregate.extend(event.iter().copied());
            average_clustering(node_count, &aggregate)
        })
        .collect()
}
fn average_clustering_serial(node_count: usize, event: &FxHashSet<Edge>) -> f64 {
    if node_count == 0 {
        return 0.0;
    }
    let (from, to): (Vec<_>, Vec<_>) = event.iter().copied().unzip();
    let adjacency = build_adjacency_serial(node_count, &from, &to);
    let total: f64 = adjacency
        .iter()
        .map(|neighbors| {
            if neighbors.len() < 2 {
                return 0.0;
            }
            let mut links = 0usize;
            for i in 0..neighbors.len() {
                for &v in &neighbors[(i + 1)..] {
                    if adjacency[neighbors[i] as usize].binary_search(&v).is_ok() {
                        links += 1;
                    }
                }
            }
            2.0 * links as f64 / (neighbors.len() * (neighbors.len() - 1)) as f64
        })
        .sum();
    total / node_count as f64
}
fn cumulative_clustering_serial(node_count: usize, temporal: &TemporalEvents) -> Vec<f64> {
    let mut aggregate = FxHashSet::default();
    temporal
        .events
        .iter()
        .map(|event| {
            aggregate.extend(event.iter().copied());
            average_clustering_serial(node_count, &aggregate)
        })
        .collect()
}
fn mean_std(series: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
    if series.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let n = series.len() as f64;
    let means: Vec<_> = (0..series[0].len())
        .map(|i| series.iter().map(|x| x[i]).sum::<f64>() / n)
        .collect();
    let std = (0..means.len())
        .map(|i| {
            (series
                .iter()
                .map(|x| (x[i] - means[i]).powi(2))
                .sum::<f64>()
                / n)
                .sqrt()
        })
        .collect();
    (means, std)
}
fn random_only_events(temporal: &TemporalEvents, output: &RecastOutput) -> TemporalEvents {
    let random: FxHashSet<Edge> = output
        .edge_from
        .iter()
        .zip(&output.edge_to)
        .zip(&output.classes)
        .filter_map(|((&u, &v), &class)| (class == 3).then_some((u, v)))
        .collect();
    TemporalEvents {
        events: temporal
            .events
            .iter()
            .map(|event| {
                event
                    .iter()
                    .filter(|e| random.contains(e))
                    .copied()
                    .collect()
            })
            .collect(),
        window_starts_ms: temporal.window_starts_ms.clone(),
    }
}

pub struct RecastValidationOutput {
    pub classification: RecastOutput,
    pub observed_persistence: Vec<f32>,
    pub observed_overlap: Vec<f32>,
    pub random_persistence: Vec<f32>,
    pub random_overlap: Vec<f32>,
    pub full_observed_clustering: Vec<f64>,
    pub full_random_mean: Vec<f64>,
    pub full_random_std: Vec<f64>,
    pub random_only_observed_clustering: Vec<f64>,
    pub random_only_random_mean: Vec<f64>,
    pub random_only_random_std: Vec<f64>,
    pub graph: TemporalGraphOutput,
}
#[allow(clippy::too_many_arguments)]
pub fn validate_recast(
    node_count: usize,
    users: &[u32],
    locations: &[u32],
    starts: &[i64],
    ends: &[i64],
    min_contact_ms: i64,
    p_rnd: f64,
    replicas: usize,
    seed: u64,
) -> RecastValidationOutput {
    let temporal = event_graphs(users, locations, starts, ends, min_contact_ms);
    let randoms = random_replicas(node_count, &temporal, replicas, seed);
    let (classification, random_persistence, random_overlap) =
        classify_events_with_randoms(node_count, &temporal, p_rnd, &randoms);
    let full_observed_clustering = cumulative_clustering(node_count, &temporal);
    let full_series: Vec<_> = randoms
        .par_iter()
        .map(|x| cumulative_clustering_serial(node_count, x))
        .collect();
    let (full_random_mean, full_random_std) = mean_std(&full_series);
    let observed_random = random_only_events(&temporal, &classification);
    let random_only_observed_clustering = cumulative_clustering(node_count, &observed_random);
    let random_only_series: Vec<_> = randoms
        .par_iter()
        .map(|x| cumulative_clustering_serial(node_count, &random_only_events(x, &classification)))
        .collect();
    let (random_only_random_mean, random_only_random_std) = mean_std(&random_only_series);
    RecastValidationOutput {
        observed_persistence: classification.persistence.clone(),
        observed_overlap: classification.topological_overlap.clone(),
        classification,
        random_persistence,
        random_overlap,
        full_observed_clustering,
        full_random_mean,
        full_random_std,
        random_only_observed_clustering,
        random_only_random_mean,
        random_only_random_std,
        graph: flatten_events(&temporal),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rayon::ThreadPoolBuilder;

    #[test]
    fn adaptive_overlap_matches_sparse_reference() {
        // Dense clique, sparse chains, isolated nodes, and mixed neighborhoods.
        let mut from = Vec::new();
        let mut to = Vec::new();
        for u in 0..250u32 {
            for v in u + 1..300u32 {
                if v < 180 || (u * 17 + v * 13) % 41 == 0 {
                    from.push(u);
                    to.push(v);
                }
            }
        }
        assert_eq!(overlaps(310, &from, &to), overlaps_serial(310, &from, &to));
    }

    #[test]
    fn exact_tail_and_histogram_match_sorted_quantiles() {
        let original: Vec<f32> = (0..200_123)
            .map(|i| ((i * 7919) % 107) as f32 / 107.0)
            .collect();
        for p in [0.0, 0.00001, 0.001, 0.1, 0.5, 0.9, 0.99999, 1.0] {
            let mut tail = NullTail::new(original.len() + 199, p);
            let mut histogram = FxHashMap::default();
            for chunk in original.chunks(1337) {
                tail.extend(chunk);
            }
            for &v in &original {
                *histogram.entry(v.to_bits()).or_insert(0) += 1;
            }
            let expected = threshold(&mut original.clone(), p);
            assert_eq!(tail.threshold(p), expected);
            assert_eq!(histogram_threshold(&histogram, p), expected);
            assert_eq!(NullTail::new(10, p).threshold(p), 1.0);
        }
    }

    #[test]
    fn streaming_matches_materialized_reference() {
        let (users, locations, starts, ends) = parallel_fixture();
        let temporal = event_graphs(&users, &locations, &starts, &ends, 60_000);
        for seed in [0, 9, 42] {
            for p in [0.0, 0.001, 0.5, 1.0] {
                let randoms = random_replicas(4, &temporal, 5, seed);
                let expected = classify_events_with_randoms(4, &temporal, p, &randoms).0;
                let actual = classify_events_streaming(4, temporal.clone(), p, 5, seed);
                assert_eq!(actual.edge_from, expected.edge_from);
                assert_eq!(actual.edge_to, expected.edge_to);
                assert_eq!(actual.persistence, expected.persistence);
                assert_eq!(actual.topological_overlap, expected.topological_overlap);
                assert_eq!(actual.classes, expected.classes);
                assert_eq!(actual.persistence_threshold, expected.persistence_threshold);
                assert_eq!(actual.overlap_threshold, expected.overlap_threshold);
            }
        }
    }

    fn parallel_fixture() -> (Vec<u32>, Vec<u32>, Vec<i64>, Vec<i64>) {
        (
            vec![0, 1, 2, 0, 1, 3, 2, 3],
            vec![0, 0, 0, 1, 1, 1, 0, 1],
            vec![
                0,
                60_000,
                120_000,
                DAY_MS,
                DAY_MS + 60_000,
                DAY_MS + 120_000,
                2 * DAY_MS,
                2 * DAY_MS + 60_000,
            ],
            vec![
                300_000,
                360_000,
                420_000,
                DAY_MS + 300_000,
                DAY_MS + 360_000,
                DAY_MS + 420_000,
                2 * DAY_MS + 300_000,
                2 * DAY_MS + 360_000,
            ],
        )
    }
    #[test]
    fn interval_contacts_require_overlap() {
        let t = event_graphs(&[0, 1, 2], &[0, 0, 0], &[0, 5, 20], &[10, 15, 30], 1);
        assert_eq!(t.events.len(), 1);
        assert!(t.events[0].contains(&(0, 1)));
        assert_eq!(t.events[0].len(), 1);
    }
    #[test]
    fn interval_contacts_require_minimum_duration() {
        let t = event_graphs(&[0, 1], &[0, 0], &[0, 60_000], &[360_000, 300_000], 300_000);
        assert!(t.events[0].is_empty());
        let t = event_graphs(&[0, 1], &[0, 0], &[0, 60_000], &[360_000, 300_000], 240_000);
        assert!(t.events[0].contains(&(0, 1)));
    }
    #[test]
    fn event_graphs_matches_naive_o_n_squared_reference_on_dense_overlaps() {
        // Stress test for the swap_remove + index-sort rewrite of the sweep
        // in `event_graphs`: many users at one location on one day, with
        // heavily overlapping and staggered intervals (the exact shape that
        // made a busy H3-cell/day bucket slow at 100M-row scale), compared
        // against a straightforward O(n^2) all-pairs reference that doesn't
        // rely on any sweep/expiry mechanics at all.
        let n = 400usize;
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut users = Vec::new();
        let mut locations = Vec::new();
        let mut starts = Vec::new();
        let mut ends = Vec::new();
        for u in 0..n {
            let start = (next() % 500_000) as i64;
            let duration = 60_000 + (next() % 400_000) as i64;
            users.push(u as u32);
            locations.push(0u32);
            starts.push(start);
            ends.push(start + duration);
        }
        let min_contact_ms = 120_000;
        let observed = event_graphs(&users, &locations, &starts, &ends, min_contact_ms);
        assert_eq!(observed.events.len(), 1);

        let mut expected = FxHashSet::default();
        for i in 0..n {
            for j in (i + 1)..n {
                let overlap = starts[i].max(starts[j])..ends[i].min(ends[j]);
                if overlap.end - overlap.start >= min_contact_ms {
                    expected.insert(edge(users[i], users[j]));
                }
            }
        }
        assert_eq!(observed.events[0], expected);
        assert!(
            !expected.is_empty(),
            "test fixture produced no overlaps to check"
        );
    }
    #[test]
    fn rnd_is_seeded_simple_graph() {
        let event: FxHashSet<Edge> = [(0, 1), (0, 2), (1, 3)].into_iter().collect();
        let mut a = Xoshiro256PlusPlus::seed_from_u64(4);
        let mut b = Xoshiro256PlusPlus::seed_from_u64(4);
        assert_eq!(rnd(4, &event, &mut a), rnd(4, &event, &mut b));
    }
    #[test]
    fn rnd_edge_cases_emit_nothing() {
        let empty: FxHashSet<Edge> = FxHashSet::default();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        assert!(rnd(0, &empty, &mut rng).is_empty());
        assert!(rnd(1, &empty, &mut rng).is_empty());
        assert!(rnd(5, &empty, &mut rng).is_empty());
    }
    #[test]
    fn sample_expected_degree_edges_matches_theoretical_probabilities() {
        // Empirical acceptance-frequency check for the Miller-Hagberg
        // sampler against the exact per-pair Bernoulli(d_u*d_v/sum) model it
        // replaces the naive O(n^2) loop for -- a wrong node/position
        // mapping, counting-sort bug, or skip/thinning bug would shift these
        // frequencies even though it might still "look like a graph".
        let node_count = 12;
        let degree = [8usize, 6, 6, 5, 4, 3, 3, 2, 2, 1, 1, 1];
        let sum: usize = degree.iter().sum();
        let trials = 40_000u64;
        let mut observed = vec![0u64; node_count * node_count];
        for seed in 0..trials {
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
            sample_expected_degree_edges(node_count, &degree, sum, &mut rng, |a, b| {
                let (lo, hi) = if a < b { (a, b) } else { (b, a) };
                observed[lo as usize * node_count + hi as usize] += 1;
            });
        }
        for u in 0..node_count {
            for v in (u + 1)..node_count {
                let expected_p = ((degree[u] * degree[v]) as f64 / sum as f64).min(1.0);
                let observed_p = observed[u * node_count + v] as f64 / trials as f64;
                // Binomial standard error at worst case p=0.5 is
                // 0.5/sqrt(40000) ~= 0.0025; 0.02 is a generous margin.
                assert!(
                    (observed_p - expected_p).abs() < 0.02,
                    "pair ({u},{v}): expected {expected_p:.4}, observed {observed_p:.4}"
                );
            }
        }
    }
    #[test]
    fn sort_nodes_by_degree_desc_matches_comparison_sort() {
        let degree = [3usize, 0, 3, 7, 1, 0, 5, 1, 7, 2];
        let mut expected: Vec<u32> = (0..degree.len() as u32).collect();
        expected.sort_by(|&a, &b| degree[b as usize].cmp(&degree[a as usize]));
        let actual = sort_nodes_by_degree_desc(&degree);
        assert_eq!(actual.len(), expected.len());
        // Counting sort may break ties differently than the comparison sort
        // (both are valid: within-tie order doesn't affect the sampler's
        // correctness), so compare degree *sequences* rather than exact
        // node-id order.
        let actual_degrees: Vec<usize> = actual.iter().map(|&i| degree[i as usize]).collect();
        let expected_degrees: Vec<usize> = expected.iter().map(|&i| degree[i as usize]).collect();
        assert_eq!(actual_degrees, expected_degrees);
        let mut actual_sorted = actual.clone();
        actual_sorted.sort_unstable();
        assert_eq!(actual_sorted, (0..degree.len() as u32).collect::<Vec<_>>());
    }
    #[test]
    fn sort_nodes_by_degree_desc_handles_empty_and_all_zero() {
        assert!(sort_nodes_by_degree_desc(&[]).is_empty());
        let all_zero = sort_nodes_by_degree_desc(&[0, 0, 0]);
        let mut sorted = all_zero.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, vec![0, 1, 2]);
    }
    #[test]
    fn recast_emits_all_four_relationship_classes() {
        assert_eq!(relationship_class(2.0, 2.0, 1.0, 1.0), 0);
        assert_eq!(relationship_class(2.0, 1.0, 1.0, 1.0), 1);
        assert_eq!(relationship_class(1.0, 2.0, 1.0, 1.0), 2);
        assert_eq!(relationship_class(1.0, 1.0, 1.0, 1.0), 3);
    }
    #[test]
    fn classifier_is_stable_across_rayon_thread_counts() {
        let (users, locations, starts, ends) = parallel_fixture();
        let single = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let many = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let one = single
            .install(|| recast_classify(4, &users, &locations, &starts, &ends, 60_000, 0.5, 3, 9));
        let four = many
            .install(|| recast_classify(4, &users, &locations, &starts, &ends, 60_000, 0.5, 3, 9));
        assert_eq!(one.edge_from, four.edge_from);
        assert_eq!(one.edge_to, four.edge_to);
        assert_eq!(one.persistence, four.persistence);
        assert_eq!(one.topological_overlap, four.topological_overlap);
        assert_eq!(one.classes, four.classes);
        assert_eq!(one.persistence_threshold, four.persistence_threshold);
        assert_eq!(one.overlap_threshold, four.overlap_threshold);
    }
    #[test]
    fn temporal_randomization_is_stable_across_rayon_thread_counts() {
        let (users, locations, starts, ends) = parallel_fixture();
        let temporal = event_graphs(&users, &locations, &starts, &ends, 60_000);
        let single = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let many = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let one = single.install(|| flatten_events(&t_rnd(4, &temporal, 2, 17)));
        let four = many.install(|| flatten_events(&t_rnd(4, &temporal, 2, 17)));
        assert_eq!(one.edge_offsets, four.edge_offsets);
        assert_eq!(one.edge_from, four.edge_from);
        assert_eq!(one.edge_to, four.edge_to);
    }
    #[test]
    fn validation_keeps_classification_stable_across_rayon_thread_counts() {
        let (users, locations, starts, ends) = parallel_fixture();
        let single = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let many = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let one = single
            .install(|| validate_recast(4, &users, &locations, &starts, &ends, 60_000, 0.5, 3, 23));
        let four = many
            .install(|| validate_recast(4, &users, &locations, &starts, &ends, 60_000, 0.5, 3, 23));
        assert_eq!(one.classification.edge_from, four.classification.edge_from);
        assert_eq!(one.classification.edge_to, four.classification.edge_to);
        assert_eq!(one.classification.classes, four.classification.classes);
        assert_eq!(
            one.classification.persistence_threshold,
            four.classification.persistence_threshold
        );
        assert_eq!(
            one.classification.overlap_threshold,
            four.classification.overlap_threshold
        );
        for (left, right) in one.full_random_mean.iter().zip(&four.full_random_mean) {
            assert!((left - right).abs() < 1e-12);
        }
        for (left, right) in one
            .random_only_random_std
            .iter()
            .zip(&four.random_only_random_std)
        {
            assert!((left - right).abs() < 1e-12);
        }
    }
}
