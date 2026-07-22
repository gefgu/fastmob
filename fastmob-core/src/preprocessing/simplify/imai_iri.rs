//! Imai-Iri simplification (Imai & Iri, 1988): builds the full feasible-edge
//! table using the shared [`super::corridor::farthest_feasible_index`] Wedge
//! primitive, then finds the minimum-segment path from the first to the
//! last point.
//!
//! For a fixed anchor `i`, feasibility of the candidate segment `(i, j)` is
//! monotonic in `j`: if all points strictly between `i` and `j` fit within
//! `tolerance` of segment `i -> j`, removing points further from `j`
//! (considering a smaller `j`) can only relax the constraint. This means
//! the reachable set from `i` is exactly the contiguous range
//! `(i, farthest_feasible_index(i)]`, so the minimum-segment path can be
//! found with a standard greedy interval-covering scan (at each anchor,
//! jump to whichever reachable index has the farthest onward reach) instead
//! of a general-purpose shortest-path search.

use crate::utils::haversine::project_local_planar_km;

use super::corridor::farthest_feasible_index;

/// Return the 0-based local indices retained by Imai-Iri simplification for
/// a single user's trajectory.
///
/// `epsilon_km` is the maximum perpendicular distance, in kilometres,
/// allowed between an original point and its nearest retained segment.
///
/// @usedBy `fastmob-core/src/preprocessing/simplify/mod.rs::simplify_user_slice`
/// (method = `imai_iri`).
pub fn imai_iri_indices(lats: &[f64], lngs: &[f64], epsilon_km: f64) -> Vec<usize> {
    let n = lats.len();
    if n <= 2 {
        return (0..n).collect();
    }

    let coords = project_local_planar_km(lats, lngs);
    let farthest_reach: Vec<usize> = (0..n - 1)
        .map(|i| farthest_feasible_index(&coords, i, epsilon_km))
        .collect();

    let mut path = vec![0usize];
    let mut anchor = 0usize;

    while anchor < n - 1 {
        let limit = farthest_reach[anchor];
        if limit >= n - 1 {
            path.push(n - 1);
            break;
        }

        let mut best = anchor + 1;
        let mut best_reach = farthest_reach[best];
        for (candidate, &reach) in farthest_reach
            .iter()
            .enumerate()
            .take(limit + 1)
            .skip(anchor + 2)
        {
            if reach > best_reach {
                best = candidate;
                best_reach = reach;
            }
        }

        path.push(best);
        anchor = best;
    }

    path
}
