//! AngleChange segmentation, ported from MovingPandas'
//! `AngleChangeSplitter` (`trajectory_splitter.py`).
//!
//! MovingPandas tracks a running "comparison direction" (`comp_dir`,
//! initialized to the first point's own direction, which is always `0.0`
//! since there is no incoming heading into the first point) and bumps the
//! segment group whenever a point's incoming heading deviates from
//! `comp_dir` by at least `min_angle` degrees *and* the point's incoming
//! speed is at least `min_speed`; `comp_dir` is then reset to that point's
//! heading. This module reproduces that exact state machine, row-for-row.

use crate::utils::haversine::{angular_difference, bearing_deg, haversine_km};

/// AngleChange segment ids for one user's chronologically-sorted slice.
///
/// Point `0` always starts segment `0` (there is no incoming heading to
/// evaluate). For `i >= 1`, the incoming heading `bearing_deg(i-1, i)` and
/// incoming speed (`haversine_km(i-1, i) / dt * 3600`, `0.0` for non-positive
/// `dt`) are compared against the running `comp_dir` reference bearing;
/// deviations of at least `min_angle_deg` while at least `min_speed_kmh`
/// bump the segment id and become the new reference bearing.
///
/// @usedBy `fastmob-core/src/preprocessing/segment/mod.rs::segment_user_slice`
/// (method = `angle_change`).
pub fn angle_change_segment_ids(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    min_angle_deg: f64,
    min_speed_kmh: f64,
) -> Vec<u32> {
    let n = lats.len();
    let mut out = vec![0u32; n];
    if n == 0 {
        return out;
    }

    // Matches MovingPandas: direction[0] is undefined (no previous point),
    // treated as 0.0, and used as the initial comparison bearing.
    let mut comp_dir = 0.0f64;
    let mut group = 0u32;

    for i in 1..n {
        let dt = times[i] - times[i - 1];
        let dist_km = haversine_km(lats[i - 1], lngs[i - 1], lats[i], lngs[i]);
        let speed_kmh = if dt > 0.0 { dist_km / dt * 3600.0 } else { 0.0 };

        if speed_kmh >= min_speed_kmh {
            let direction = bearing_deg(lats[i - 1], lngs[i - 1], lats[i], lngs[i]);
            if angular_difference(comp_dir, direction) >= min_angle_deg {
                comp_dir = direction;
                group += 1;
            }
        }

        out[i] = group;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn times_seq(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 * 60.0).collect()
    }

    #[test]
    fn empty_and_single_point_inputs() {
        assert_eq!(angle_change_segment_ids(&[], &[], &[], 45.0, 0.0), Vec::<u32>::new());
        assert_eq!(
            angle_change_segment_ids(&[0.0], &[0.0], &[0.0], 45.0, 0.0),
            vec![0]
        );
    }

    #[test]
    fn straight_line_never_splits() {
        // Heading due north the whole way: no angle change at all.
        let lats: Vec<f64> = (0..5).map(|i| 48.0 + i as f64 * 0.01).collect();
        let lngs = vec![2.0; 5];
        let times = times_seq(5);
        let ids = angle_change_segment_ids(&lats, &lngs, &times, 45.0, 0.0);
        assert_eq!(ids, vec![0, 0, 0, 0, 0]);
    }

    #[test]
    fn a_ninety_degree_turn_splits_at_the_turn_point() {
        // 0 -> 1 heads due north; 1 -> 2 heads due east: a 90 deg change.
        let lats = vec![48.0, 48.01, 48.01];
        let lngs = vec![2.0, 2.0, 2.02];
        let times = times_seq(3);
        let ids = angle_change_segment_ids(&lats, &lngs, &times, 45.0, 0.0);
        assert_eq!(ids, vec![0, 0, 1], "the turn is registered at the point it lands on");
    }

    #[test]
    fn below_min_speed_never_splits() {
        // Same 90 deg turn, but a very high min_speed_kmh threshold means
        // the (slow) points never qualify for evaluation.
        let lats = vec![48.0, 48.01, 48.01];
        let lngs = vec![2.0, 2.0, 2.02];
        let times = times_seq(3);
        let ids = angle_change_segment_ids(&lats, &lngs, &times, 45.0, 1_000_000.0);
        assert_eq!(ids, vec![0, 0, 0]);
    }
}
