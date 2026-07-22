//! Small greedy simplification family: MinDistance, MinTimeDelta, and
//! MaxDistance. All three are single-pass, streaming algorithms (no
//! recursion, no lookahead beyond a small buffer), ported from MovingPandas'
//! `MinDistanceGeneralizer`, `MinTimeDeltaGeneralizer`, and
//! `MaxDistanceGeneralizer`.

use crate::utils::haversine::{haversine_km, point_to_segment_distance_km, project_local_planar_km};

/// Return the 0-based local indices retained by MinDistance simplification:
/// the first point, then every subsequent point whose haversine distance
/// from the previously *kept* point is at least `min_distance_km`, plus the
/// last point (always kept, matching MovingPandas' `MinDistanceGeneralizer`).
///
/// @usedBy `fastmob-core/src/preprocessing/simplify/mod.rs::simplify_user_slice`
/// (method = `min_distance`).
pub fn min_distance_indices(lats: &[f64], lngs: &[f64], min_distance_km: f64) -> Vec<usize> {
    let n = lats.len();
    if n == 0 {
        return Vec::new();
    }

    let mut keep = vec![0usize];
    let mut prev = 0usize;

    for i in 1..n {
        let dist = haversine_km(lats[prev], lngs[prev], lats[i], lngs[i]);
        if dist >= min_distance_km {
            keep.push(i);
            prev = i;
        }
    }

    if *keep.last().expect("keep always has at least one index") != n - 1 {
        keep.push(n - 1);
    }
    keep
}

/// Return the 0-based local indices retained by MinTimeDelta simplification:
/// the first point, then every subsequent point whose timestamp is at least
/// `min_time_delta_s` seconds after the previously *kept* point's timestamp,
/// plus the last point (always kept, matching MovingPandas'
/// `MinTimeDeltaGeneralizer`).
///
/// @usedBy `fastmob-core/src/preprocessing/simplify/mod.rs::simplify_user_slice`
/// (method = `min_time_delta`).
pub fn min_time_delta_indices(times: &[f64], min_time_delta_s: f64) -> Vec<usize> {
    let n = times.len();
    if n == 0 {
        return Vec::new();
    }

    let mut keep = vec![0usize];
    let mut prev = 0usize;

    for i in 1..n {
        let delta = times[i] - times[prev];
        if delta >= min_time_delta_s {
            keep.push(i);
            prev = i;
        }
    }

    if *keep.last().expect("keep always has at least one index") != n - 1 {
        keep.push(n - 1);
    }
    keep
}

/// Return the 0-based local indices retained by MaxDistance simplification,
/// a single-pass streaming approximation of Douglas-Peucker ported from
/// MovingPandas' `MaxDistanceGeneralizer`.
///
/// Maintains a running anchor point and a buffer of skipped points since
/// that anchor. For each new point, forms the candidate replacement segment
/// `anchor -> point` and checks whether any buffered point deviates more
/// than `max_distance_km` from it (using [`point_to_segment_distance_km`]
/// over locally-projected planar coordinates); if so, the point *before*
/// the current one is marked as a kept vertex, the anchor moves to the
/// current point, and the buffer is cleared. The last point is always kept.
///
/// @usedBy `fastmob-core/src/preprocessing/simplify/mod.rs::simplify_user_slice`
/// (method = `max_distance`).
pub fn max_distance_indices(lats: &[f64], lngs: &[f64], max_distance_km: f64) -> Vec<usize> {
    let n = lats.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    let coords = project_local_planar_km(lats, lngs);

    let mut keep_rows = vec![0usize];
    let mut anchor = coords[0];
    let mut buffered: Vec<(f64, f64)> = Vec::new();

    for (idx, &current) in coords.iter().enumerate().skip(1) {
        let exceeds = buffered
            .iter()
            .any(|&p| point_to_segment_distance_km(p, anchor, current) > max_distance_km);

        if exceeds {
            anchor = current;
            buffered.clear();
            keep_rows.push(idx - 1);
        }
        buffered.push(current);
    }

    keep_rows.push(n - 1);
    keep_rows
}
