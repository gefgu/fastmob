use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

type DailyMotifsResult = Result<(Vec<String>, Vec<i32>, Vec<i64>), String>;

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Visit<'a> {
    uid: &'a str,
    purpose: &'a str,
    start_hour: u32,
    end_hour: u32,
    date_id: i32,
    duration_minutes: Option<f64>,
}

#[derive(Debug)]
struct DailyMotifResult {
    user_id: String,
    date_id: i32,
    motif_id: i64,
}

// ---------------------------------------------------------------------------
// next_permutation helper
// ---------------------------------------------------------------------------

fn next_permutation(v: &mut [usize]) -> bool {
    let n = v.len();
    if n < 2 {
        return false;
    }

    let mut i = n - 1;
    loop {
        if i == 0 {
            return false;
        }
        i -= 1;
        if v[i] < v[i + 1] {
            break;
        }
    }

    let mut j = n - 1;
    while v[j] <= v[i] {
        j -= 1;
    }
    v.swap(i, j);
    v[i + 1..].reverse();
    true
}

// ---------------------------------------------------------------------------
// canonical_adjacency_form_internal — pure Rust, no PyO3 overhead
// ---------------------------------------------------------------------------

// Return a u64 instead of a String
pub fn canonical_adjacency_form_internal(
    n_nodes: usize,
    edges: &[(u32, u32)],
) -> Result<i64, String> {
    if n_nodes > 6 {
        return Err("This optimized function supports at most 6 nodes".to_string());
    }

    let n = n_nodes;
    let mut adj = vec![false; n * n];
    for &(u, v) in edges {
        adj[(u as usize) * n + (v as usize)] = true;
    }

    let mut perm: Vec<usize> = (0..n).collect();
    let mut best: i64 = 0; // Switched to i64 for maximum speed

    loop {
        let mut val: i64 = 0;
        for i in 0..n {
            for j in 0..n {
                val <<= 1;
                if adj[perm[i] * n + perm[j]] {
                    val |= 1;
                }
            }
        }
        if val > best {
            best = val;
        }

        if !next_permutation(&mut perm) {
            break;
        }
    }

    // PACKING LOGIC:
    // Shift the node count 36 bits to the left, then combine with the matrix integer
    let combined_id: i64 = ((n as i64) << 36) | best;

    Ok(combined_id)
}

// ---------------------------------------------------------------------------
// canonical_adjacency_form — public wrapper
// ---------------------------------------------------------------------------

pub fn canonical_adjacency_form(n_nodes: u32, edges: Vec<(u32, u32)>) -> Result<i64, String> {
    canonical_adjacency_form_internal(n_nodes as usize, &edges)
}

// ---------------------------------------------------------------------------
// Primary-home detection
// ---------------------------------------------------------------------------

/// Returns the `uid` of the primary home node for the given user visits, or
/// `None` if no HOME rows exist (in which case the user is skipped entirely).
///
/// Rule (matches the Python implementation in `_compute_primary_home_node_id`):
/// 1. Filter rows where `purpose == "HOME"` AND night window (`start_hour >= 22
///    OR start_hour < 6`).  Sum `duration_minutes` per `uid`.  Pick the uid
///    with the highest total.
/// 2. If no night HOME rows exist, fall back to the most frequent HOME uid
///    (ignoring time of day).
/// 3. Return `None` when no HOME rows at all.
fn compute_primary_home_node_id<'a>(visits: &[Visit<'a>]) -> Option<&'a str> {
    // Step 1: night HOME visits
    let mut night_duration: FxHashMap<&str, f64> = FxHashMap::default();
    for visit in visits {
        if visit.purpose == "HOME" && (visit.start_hour >= 22 || visit.start_hour < 6) {
            *night_duration.entry(visit.uid).or_insert(0.0) +=
                visit.duration_minutes.unwrap_or(0.0);
        }
    }

    if !night_duration.is_empty() {
        return night_duration
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(uid, _)| uid);
    }

    // Step 2: most frequent HOME uid regardless of hour
    let mut home_counts: FxHashMap<&str, usize> = FxHashMap::default();
    for visit in visits {
        if visit.purpose == "HOME" {
            *home_counts.entry(visit.uid).or_insert(0) += 1;
        }
    }

    if home_counts.is_empty() {
        return None;
    }

    home_counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(uid, _)| uid)
}

// ---------------------------------------------------------------------------
// Per-day graph construction
// ---------------------------------------------------------------------------

fn compute_motif_from_daily_visits<'a>(
    user_id: &str,
    daily_visits: &[Visit<'a>],
    primary_home: &'a str,
    last_night_node: Option<&'a str>,
    next_day_first_node: Option<&'a str>,
) -> DailyMotifResult {
    let date_id = daily_visits.first().map(|v| v.date_id).unwrap_or(0);
    let home_suffix = "_HOME";

    // Build sequence: [start] ++ day_visits ++ [closure]
    let mut sequence: Vec<&str> = Vec::with_capacity(daily_visits.len() + 2);
    let start_node = last_night_node.unwrap_or(primary_home);
    sequence.push(start_node);

    for visit in daily_visits {
        sequence.push(visit.uid);
    }

    // Loop closure: ensure day ends at a home node
    let last_node = *sequence.last().expect("sequence is never empty");
    if !last_node.ends_with(home_suffix) {
        if let Some(next) = next_day_first_node {
            if next.ends_with(home_suffix) {
                sequence.push(next);
            } else {
                sequence.push(primary_home);
            }
        } else {
            sequence.push(primary_home);
        }
    }

    // Remove consecutive duplicates
    let mut cleaned: Vec<&str> = vec![sequence[0]];
    for &node in sequence.iter().skip(1) {
        if node != *cleaned.last().unwrap() {
            cleaned.push(node);
        }
    }

    // Build unique node list; force primary_home to index 0
    let mut unique_nodes: Vec<&str> = cleaned.clone();
    unique_nodes.sort_unstable();
    unique_nodes.dedup();
    if let Some(pos) = unique_nodes.iter().position(|&x| x == primary_home) {
        unique_nodes.remove(pos);
    }
    unique_nodes.insert(0, primary_home);

    let id_map: FxHashMap<&str, u32> = unique_nodes
        .iter()
        .enumerate()
        .map(|(i, &name)| (name, i as u32))
        .collect();

    // Collect edges (deduplicated)
    let mut edges_set: FxHashSet<(u32, u32)> = FxHashSet::default();
    for window in cleaned.windows(2) {
        let u = id_map[window[0]];
        let v = id_map[window[1]];
        edges_set.insert((u, v));
    }

    let edges: Vec<(u32, u32)> = edges_set.into_iter().collect();
    let n_nodes = unique_nodes.len();

    let motif_id: i64 = if n_nodes > 6 {
        -1
    } else {
        canonical_adjacency_form_internal(n_nodes, &edges).unwrap_or(-1)
    };

    DailyMotifResult {
        user_id: user_id.to_string(),
        date_id,
        motif_id,
    }
}

// ---------------------------------------------------------------------------
// Per-user processing
// ---------------------------------------------------------------------------

fn process_single_user<'a>(user_id: &str, visits: &[Visit<'a>]) -> Vec<DailyMotifResult> {
    let primary_home = match compute_primary_home_node_id(visits) {
        Some(h) => h,
        None => return vec![],
    };

    // visits are already sorted by [user_id, start_timestamp] from Python
    // so date_ids are already grouped by day within each user
    let daily_chunks: Vec<&[Visit<'_>]> =
        visits.chunk_by(|a, b| a.date_id == b.date_id).collect();

    let mut results = Vec::with_capacity(daily_chunks.len());

    for i in 0..daily_chunks.len() {
        let current_day = daily_chunks[i];

        // Look-back: last visit of the previous day
        // If that visit is the primary home and ended after 03:00, it "bleeds" into today
        let last_night_node: Option<&str> = if i > 0 {
            let prev_day = daily_chunks[i - 1];
            if let Some(last_visit) = prev_day.last() {
                if last_visit.uid == primary_home
                    && (last_visit.end_hour > 3 || last_visit.end_hour == 23)
                {
                    Some(last_visit.uid)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Look-ahead: first visit of the next day
        // If it starts before 03:00 and is a HOME node, use it as closure
        let next_day_first_node: Option<&str> = if i + 1 < daily_chunks.len() {
            let next_day = daily_chunks[i + 1];
            if let Some(first_visit) = next_day.first() {
                if first_visit.start_hour < 3 && first_visit.uid.ends_with("_HOME") {
                    Some(first_visit.uid)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let result = compute_motif_from_daily_visits(
            user_id,
            current_day,
            primary_home,
            last_night_node,
            next_day_first_node,
        );
        results.push(result);
    }

    results
}

// ---------------------------------------------------------------------------
// Public kernel: compute_daily_motifs
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub fn compute_daily_motifs(
    unique_ids: Vec<String>,
    purposes: Vec<String>,
    start_hours: Vec<u32>,
    end_hours: Vec<u32>,
    date_ids: Vec<i32>,
    durations: Vec<Option<f64>>,
    user_ranges: Vec<(usize, usize)>,
    user_id_labels: Vec<String>,
) -> DailyMotifsResult {
    let n = unique_ids.len();

    // Validate all column vectors have the same length
    if purposes.len() != n
        || start_hours.len() != n
        || end_hours.len() != n
        || date_ids.len() != n
        || durations.len() != n
    {
        return Err("All input column vectors must have the same length".to_string());
    }
    if user_id_labels.len() != user_ranges.len() {
        return Err("user_id_labels and user_ranges must have the same length".to_string());
    }

    // Build Visit slices from the flat arrays (borrows into the input Vecs)
    let all_visits: Vec<Visit<'_>> = (0..n)
        .map(|i| Visit {
            uid: &unique_ids[i],
            purpose: &purposes[i],
            start_hour: start_hours[i],
            end_hour: end_hours[i],
            date_id: date_ids[i],
            duration_minutes: durations[i],
        })
        .collect();

    // Parallel processing: each user's range is independent
    let all_results: Vec<DailyMotifResult> = user_ranges
        .par_iter()
        .zip(user_id_labels.par_iter())
        .flat_map(|(&(start, end), uid)| {
            let user_visits = &all_visits[start..end];
            process_single_user(uid, user_visits)
        })
        .collect();

    // Unzip into three parallel output vectors
    let mut out_user_ids = Vec::with_capacity(all_results.len());
    let mut out_date_ids = Vec::with_capacity(all_results.len());
    let mut out_motif_ids = Vec::with_capacity(all_results.len());

    for r in all_results {
        out_user_ids.push(r.user_id);
        out_date_ids.push(r.date_id);
        out_motif_ids.push(r.motif_id);
    }

    Ok((out_user_ids, out_date_ids, out_motif_ids))
}
