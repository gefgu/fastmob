//! RECAST temporal-contact construction, paper RND, and validation kernels.
//!
//! The classifier follows Section 5 of Vaz de Melo et al. (2015).  Event
//! graphs are UTC-day snapshots; RND samples each possible pair in a snapshot
//! with the paper's degree-product probability, and T-RND applies RND to every
//! snapshot independently.

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};
use std::cmp::Ordering;

type Edge = (u32, u32);
const DAY_MS: i64 = 86_400_000;

fn edge(a: u32, b: u32) -> Edge {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
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
            if lo < hi {
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
            let mut active: Vec<Interval> = Vec::new();
            let mut edges = FxHashSet::default();
            for current in intervals {
                active.retain(|other| other.end > current.start);
                for other in &active {
                    if other.user != current.user
                        && other.end.min(current.end) - current.start >= min_contact_ms
                    {
                        edges.insert(edge(other.user, current.user));
                    }
                }
                active.push(current);
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
) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let mut values: Vec<_> = counts.into_iter().collect();
    values.sort_unstable_by_key(|(e, _)| *e);
    let denom = steps.max(1) as f64;
    let mut from = Vec::with_capacity(values.len());
    let mut to = Vec::with_capacity(values.len());
    let mut persistence = Vec::with_capacity(values.len());
    for ((u, v), n) in values {
        from.push(u);
        to.push(v);
        persistence.push(n as f64 / denom);
    }
    (from, to, persistence)
}

fn aggregate_serial(events: &[FxHashSet<Edge>]) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let mut counts: FxHashMap<Edge, usize> = FxHashMap::default();
    for event in events {
        for &e in event {
            *counts.entry(e).or_insert(0) += 1;
        }
    }
    aggregate_from_counts(counts, events.len())
}

fn aggregate(events: &[FxHashSet<Edge>]) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
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
fn count_common(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut n) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                n += 1;
                i += 1;
                j += 1;
            }
        }
    }
    n
}
fn overlaps(node_count: usize, from: &[u32], to: &[u32]) -> Vec<f64> {
    let adjacency = build_adjacency(node_count, from, to);
    (0..from.len())
        .into_par_iter()
        .map(|i| {
            let a = &adjacency[from[i] as usize];
            let b = &adjacency[to[i] as usize];
            let inter = count_common(a, b);
            let union = a.len() + b.len() - inter;
            if union == 0 {
                0.0
            } else {
                inter as f64 / union as f64
            }
        })
        .collect()
}
fn overlaps_serial(node_count: usize, from: &[u32], to: &[u32]) -> Vec<f64> {
    let adjacency = build_adjacency_serial(node_count, from, to);
    (0..from.len())
        .map(|i| {
            let a = &adjacency[from[i] as usize];
            let b = &adjacency[to[i] as usize];
            let inter = count_common(a, b);
            let union = a.len() + b.len() - inter;
            if union == 0 {
                0.0
            } else {
                inter as f64 / union as f64
            }
        })
        .collect()
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
    if sum == 0 {
        return FxHashSet::default();
    }
    let mut result = FxHashSet::default();
    for u in 0..node_count {
        for v in (u + 1)..node_count {
            let probability = ((degree[u] * degree[v]) as f64 / sum as f64).min(1.0);
            if rng.gen_bool(probability) {
                result.insert((u as u32, v as u32));
            }
        }
    }
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

fn threshold(values: &mut [f64], p_rnd: f64) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values[(((values.len() as f64) * (1.0 - p_rnd)).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1)]
}
fn relationship_class(p: f64, o: f64, pt: f64, ot: f64) -> u8 {
    match (p > pt, o > ot) {
        (true, true) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (false, false) => 3,
    }
}

pub struct RecastOutput {
    pub edge_from: Vec<u32>,
    pub edge_to: Vec<u32>,
    pub persistence: Vec<f64>,
    pub topological_overlap: Vec<f64>,
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
) -> (RecastOutput, Vec<f64>, Vec<f64>) {
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

fn classify_events(
    node_count: usize,
    temporal: &TemporalEvents,
    p_rnd: f64,
    replicas: usize,
    seed: u64,
) -> (RecastOutput, Vec<f64>, Vec<f64>) {
    let randoms = random_replicas(node_count, temporal, replicas, seed);
    classify_events_with_randoms(node_count, temporal, p_rnd, &randoms)
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
    let temporal = event_graphs(users, locations, starts, ends, min_contact_ms);
    classify_events(node_count, &temporal, p_rnd, replicas, seed).0
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
    pub observed_persistence: Vec<f64>,
    pub observed_overlap: Vec<f64>,
    pub random_persistence: Vec<f64>,
    pub random_overlap: Vec<f64>,
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
    fn rnd_is_seeded_simple_graph() {
        let event: FxHashSet<Edge> = [(0, 1), (0, 2), (1, 3)].into_iter().collect();
        let mut a = Xoshiro256PlusPlus::seed_from_u64(4);
        let mut b = Xoshiro256PlusPlus::seed_from_u64(4);
        assert_eq!(rnd(4, &event, &mut a), rnd(4, &event, &mut b));
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
