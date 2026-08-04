//! RECAST temporal-contact construction and classification kernels.
//!
//! Inputs are factorized user/location ids and staypoint intervals.  All
//! expensive temporal graph construction, exact degree-preserving T-RND
//! randomization, and graph metrics stay in Rust.

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

#[derive(Clone)]
struct Interval {
    user: u32,
    start: i64,
    end: i64,
}

/// Event graphs keyed by UTC-aligned fixed windows.  A contact exists only
/// when two intervals overlap strictly at the same global location.
fn event_graphs(
    users: &[u32],
    locations: &[u32],
    starts: &[i64],
    ends: &[i64],
    min_contact_ms: i64,
) -> (Vec<FxHashSet<Edge>>, usize) {
    if users.is_empty() {
        return (Vec::new(), 0);
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
        let first = starts[i].div_euclid(DAY_MS);
        let last = (ends[i] - 1).div_euclid(DAY_MS);
        for window in first..=last {
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
    let mut events: Vec<FxHashSet<Edge>> = (0..steps).map(|_| FxHashSet::default()).collect();
    for ((window, _), mut intervals) in groups {
        intervals.sort_unstable_by_key(|x| (x.start, x.end, x.user));
        let mut active: Vec<Interval> = Vec::new();
        let target = &mut events[(window - min_window) as usize];
        for current in intervals {
            active.retain(|other| other.end > current.start);
            for other in &active {
                if other.user != current.user
                    && other.end.min(current.end) - current.start >= min_contact_ms
                {
                    target.insert(edge(other.user, current.user));
                }
            }
            active.push(current);
        }
    }
    (events, steps)
}

fn aggregate(events: &[FxHashSet<Edge>], steps: usize) -> (Vec<u32>, Vec<u32>, Vec<f64>) {
    let mut counts: FxHashMap<Edge, usize> = FxHashMap::default();
    for event in events {
        for &e in event {
            *counts.entry(e).or_insert(0) += 1;
        }
    }
    let mut values: Vec<(Edge, usize)> = counts.into_iter().collect();
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

fn count_common(a: &[u32], b: &[u32]) -> usize {
    let (mut i, mut j, mut out) = (0, 0, 0);
    while i < a.len() && j < b.len() {
        match a[i].cmp(&b[j]) {
            Ordering::Less => i += 1,
            Ordering::Greater => j += 1,
            Ordering::Equal => {
                out += 1;
                i += 1;
                j += 1;
            }
        }
    }
    out
}

fn overlap(adjacency: &[Vec<u32>], u: u32, v: u32) -> f64 {
    let a = &adjacency[u as usize];
    let b = &adjacency[v as usize];
    let inter = count_common(a, b);
    let union = a.len() + b.len() - inter;
    if union == 0 {
        0.0
    } else {
        inter as f64 / union as f64
    }
}

fn overlaps(node_count: usize, from: &[u32], to: &[u32]) -> Vec<f64> {
    let adjacency = build_adjacency(node_count, from, to);
    (0..from.len())
        .into_par_iter()
        .map(|i| overlap(&adjacency, from[i], to[i]))
        .collect()
}

/// Exact simple-graph degree-preserving double-edge swaps.
fn randomize(
    event: &FxHashSet<Edge>,
    swaps_per_edge: usize,
    rng: &mut Xoshiro256PlusPlus,
) -> FxHashSet<Edge> {
    let mut set = event.clone();
    let mut edges: Vec<Edge> = set.iter().copied().collect();
    if edges.len() < 2 || swaps_per_edge == 0 {
        return set;
    }
    let target = swaps_per_edge.saturating_mul(edges.len());
    let max_attempts = target.saturating_mul(100).max(100);
    let mut accepted = 0;
    for _ in 0..max_attempts {
        if accepted >= target {
            break;
        }
        let i = rng.gen_range(0..edges.len());
        let mut j = rng.gen_range(0..edges.len());
        while j == i {
            j = rng.gen_range(0..edges.len());
        }
        let (a, b) = edges[i];
        let (c, d) = edges[j];
        let (x, y) = if rng.gen_bool(0.5) {
            (edge(a, d), edge(c, b))
        } else {
            (edge(a, c), edge(b, d))
        };
        if x.0 == x.1 || y.0 == y.1 || x == y || set.contains(&x) || set.contains(&y) {
            continue;
        }
        set.remove(&(a, b));
        set.remove(&(c, d));
        set.insert(x);
        set.insert(y);
        edges[i] = x;
        edges[j] = y;
        accepted += 1;
    }
    set
}

fn threshold(values: &mut [f64], p_rnd: f64) -> f64 {
    if values.is_empty() {
        return 1.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    let index = (((values.len() as f64) * (1.0 - p_rnd)).ceil() as usize)
        .saturating_sub(1)
        .min(values.len() - 1);
    values[index]
}

fn relationship_class(
    persistence: f64,
    overlap: f64,
    persistence_threshold: f64,
    overlap_threshold: f64,
) -> u8 {
    match (
        persistence > persistence_threshold,
        overlap > overlap_threshold,
    ) {
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

/// Faithful RECAST classification.  Class codes: 0 Friends, 1 Bridges,
/// 2 Acquaintances, 3 Random.
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
    swaps_per_edge: usize,
) -> RecastOutput {
    let (events, time_steps) = event_graphs(users, locations, starts, ends, min_contact_ms);
    let (edge_from, edge_to, persistence) = aggregate(&events, time_steps);
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
    let topological_overlap = overlaps(node_count, &edge_from, &edge_to);
    let mut null_persistence = Vec::new();
    let mut null_overlap = Vec::new();
    for replica in 0..replicas.max(1) {
        let mut randomized = Vec::with_capacity(events.len());
        for (window, event) in events.iter().enumerate() {
            let stream = seed
                ^ ((replica as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
                ^ (window as u64).wrapping_mul(0xD1B5_4A32_D192_ED03);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(stream);
            randomized.push(randomize(event, swaps_per_edge, &mut rng));
        }
        let (rf, rt, rp) = aggregate(&randomized, time_steps);
        null_persistence.extend(rp);
        null_overlap.extend(overlaps(node_count, &rf, &rt));
    }
    let persistence_threshold = threshold(&mut null_persistence, p_rnd);
    let overlap_threshold = threshold(&mut null_overlap, p_rnd);
    let classes = persistence
        .iter()
        .zip(&topological_overlap)
        .map(|(&p, &o)| relationship_class(p, o, persistence_threshold, overlap_threshold))
        .collect();
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn interval_contacts_require_overlap() {
        let (events, steps) = event_graphs(&[0, 1, 2], &[0, 0, 0], &[0, 5, 20], &[10, 15, 30], 1);
        assert_eq!(steps, 1);
        assert!(events[0].contains(&(0, 1)));
        assert_eq!(events[0].len(), 1);
    }
    #[test]
    fn interval_contacts_require_minimum_duration() {
        let users = [0, 1];
        let locations = [0, 0];
        let starts = [0, 60_000];
        let ends = [360_000, 300_000]; // Four minutes of overlap.
        let (events, _) = event_graphs(&users, &locations, &starts, &ends, 300_000);
        assert!(events[0].is_empty());
        let (events, _) = event_graphs(&users, &locations, &starts, &ends, 240_000);
        assert!(events[0].contains(&(0, 1)));
    }
    #[test]
    fn swaps_preserve_degrees_and_edge_count() {
        let event: FxHashSet<Edge> = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5)]
            .into_iter()
            .collect();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(4);
        let result = randomize(&event, 10, &mut rng);
        assert_eq!(event.len(), result.len());
        let degree =
            |edges: &FxHashSet<Edge>, n| edges.iter().filter(|&&(a, b)| a == n || b == n).count();
        for n in 0..6 {
            assert_eq!(degree(&event, n), degree(&result, n));
        }
    }
    #[test]
    fn recast_emits_all_four_relationship_classes() {
        assert_eq!(relationship_class(2.0, 2.0, 1.0, 1.0), 0); // Friends
        assert_eq!(relationship_class(2.0, 1.0, 1.0, 1.0), 1); // Bridges
        assert_eq!(relationship_class(1.0, 2.0, 1.0, 1.0), 2); // Acquaintances
        assert_eq!(relationship_class(1.0, 1.0, 1.0, 1.0), 3); // Random
    }
}
