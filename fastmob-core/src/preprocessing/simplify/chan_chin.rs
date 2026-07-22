//! Chan-Chin simplification (Chan & Chin, 1996): a fast approximate
//! min-#-segments simplification under a maximum perpendicular-distance
//! tolerance, built on the shared [`super::corridor::farthest_feasible_index`]
//! Wedge primitive.
//!
//! A single greedy forward pass (always extend the current anchor as far as
//! the wedge stays feasible) is only a 2-approximation of the optimal
//! minimum-segment simplification. This implementation strengthens that
//! result by also running the mirrored *backward* pass (anchored from the
//! trajectory's end), taking the union of feasible edges discovered by both
//! passes, and finding the shortest (fewest-segment) path from the first to
//! the last point over that combined edge set via BFS. Every edge in the
//! combined graph is independently known-feasible (point-to-segment
//! distance is symmetric in traversal direction), so any path found is a
//! valid simplification; combining both directions can only match or beat
//! either single pass.

use std::collections::VecDeque;

use crate::utils::haversine::project_local_planar_km;

use super::corridor::farthest_feasible_index;

/// Greedily walk forward from index 0, repeatedly jumping to the farthest
/// feasible index reachable from the current anchor. Returns the ascending
/// sequence of anchor indices from `0` to `coords.len() - 1`.
fn forward_wedge_path(coords: &[(f64, f64)], tolerance: f64) -> Vec<usize> {
    let n = coords.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    let mut path = vec![0usize];
    let mut anchor = 0usize;
    while anchor < n - 1 {
        let next = farthest_feasible_index(coords, anchor, tolerance);
        path.push(next);
        anchor = next;
    }
    path
}

/// Mirror of [`forward_wedge_path`], anchored from the trajectory's end.
/// Returns the ascending sequence of anchor indices from `0` to
/// `coords.len() - 1` (the reversed-frame path is mapped back to original
/// indices and reversed before returning).
fn backward_wedge_path(coords: &[(f64, f64)], tolerance: f64) -> Vec<usize> {
    let n = coords.len();
    let reversed: Vec<(f64, f64)> = coords.iter().rev().copied().collect();
    let reversed_path = forward_wedge_path(&reversed, tolerance);

    let mut path: Vec<usize> = reversed_path
        .into_iter()
        .map(|idx| n - 1 - idx)
        .collect();
    path.reverse();
    path
}

/// Return the 0-based local indices retained by Chan-Chin simplification
/// for a single user's trajectory.
///
/// `epsilon_km` is the maximum perpendicular distance, in kilometres,
/// allowed between an original point and its nearest retained segment.
///
/// @usedBy `fastmob-core/src/preprocessing/simplify/mod.rs::simplify_user_slice`
/// (method = `chan_chin`).
pub fn chan_chin_indices(lats: &[f64], lngs: &[f64], epsilon_km: f64) -> Vec<usize> {
    let n = lats.len();
    if n <= 2 {
        return (0..n).collect();
    }

    let coords = project_local_planar_km(lats, lngs);
    let forward = forward_wedge_path(&coords, epsilon_km);
    let backward = backward_wedge_path(&coords, epsilon_km);

    // Vertex indices are bounded (0..n), so a dense adjacency list indexed
    // by vertex avoids the overhead of a hash map.
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for window in forward.windows(2).chain(backward.windows(2)) {
        let (a, b) = (window[0], window[1]);
        adjacency[a].push(b);
        adjacency[b].push(a);
    }

    let mut previous: Vec<Option<usize>> = vec![None; n];
    let mut visited = vec![false; n];
    let mut queue = VecDeque::new();
    visited[0] = true;
    queue.push_back(0usize);

    while let Some(u) = queue.pop_front() {
        if u == n - 1 {
            break;
        }
        for &v in &adjacency[u] {
            if !visited[v] {
                visited[v] = true;
                previous[v] = Some(u);
                queue.push_back(v);
            }
        }
    }

    let mut path = vec![n - 1];
    let mut current = n - 1;
    while current != 0 {
        current = previous[current].expect("forward pass guarantees 0..n-1 connectivity");
        path.push(current);
    }
    path.reverse();
    path
}
