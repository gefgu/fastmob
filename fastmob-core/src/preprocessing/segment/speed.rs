//! Speed segmentation, adapted from MovingPandas' `SpeedSplitter`
//! (`trajectory_splitter.py`).
//!
//! MovingPandas' `SpeedSplitter` filters out every point whose incoming
//! speed falls outside `[speed, max_speed]` ("non-moving" points), then runs
//! `ObservationGapSplitter` on the *remaining* points, which drops rows
//! entirely. fastmob's segmentation has a hard row-preservation contract
//! (see the project plan's "Segmentation output shape" decision), so this
//! module keeps every row: a maximal run of consecutive "non-moving" points
//! spanning at least `duration_s` becomes its own bracketing segment between
//! the moving segment before it and the moving segment after it; a
//! below-threshold non-moving run is folded into the surrounding segment
//! (i.e. it does not split anything), the row-preserving analogue of
//! `ObservationGapSplitter` never seeing a gap there.

use crate::utils::haversine::haversine_km;

/// Speed segment ids for one user's chronologically-sorted slice.
///
/// Point `0` always starts segment `0` (there is no incoming speed to
/// evaluate). For `i >= 1`, the incoming speed
/// (`haversine_km(i-1, i) / dt * 3600`, `0.0` for non-positive `dt`)
/// classifies point `i` as "moving" when it falls in
/// `[speed_min_kmh, speed_max_kmh]`. Maximal runs of consecutive
/// non-moving points spanning (`times[run_end] - times[run_start]`) at
/// least `duration_s` are bracketed by a new segment id on entry and a
/// further new segment id on exit; shorter non-moving runs are folded into
/// the surrounding segment with no split.
///
/// @usedBy `fastmob-core/src/preprocessing/segment/mod.rs::segment_user_slice`
/// (method = `speed`).
pub fn speed_segment_ids(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    speed_min_kmh: f64,
    speed_max_kmh: f64,
    duration_s: f64,
) -> Vec<u32> {
    let n = lats.len();
    let mut out = vec![0u32; n];
    if n == 0 {
        return out;
    }

    let is_moving = |i: usize| -> bool {
        let dt = times[i] - times[i - 1];
        let dist_km = haversine_km(lats[i - 1], lngs[i - 1], lats[i], lngs[i]);
        let speed_kmh = if dt > 0.0 { dist_km / dt * 3600.0 } else { 0.0 };
        speed_kmh >= speed_min_kmh && speed_kmh <= speed_max_kmh
    };

    let mut group = 0u32;
    let mut i = 1usize;
    while i < n {
        if is_moving(i) {
            out[i] = group;
            i += 1;
            continue;
        }

        let run_start = i;
        let mut run_end = i;
        while run_end + 1 < n && !is_moving(run_end + 1) {
            run_end += 1;
        }
        let run_duration_s = times[run_end] - times[run_start];

        if run_duration_s >= duration_s {
            group += 1;
            for out_i in out.iter_mut().take(run_end + 1).skip(run_start) {
                *out_i = group;
            }
            group += 1;
        } else {
            for out_i in out.iter_mut().take(run_end + 1).skip(run_start) {
                *out_i = group;
            }
        }

        i = run_end + 1;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_single_point_inputs() {
        assert_eq!(
            speed_segment_ids(&[], &[], &[], 0.0, f64::INFINITY, 300.0),
            Vec::<u32>::new()
        );
        assert_eq!(
            speed_segment_ids(&[0.0], &[0.0], &[0.0], 0.0, f64::INFINITY, 300.0),
            vec![0]
        );
    }

    #[test]
    fn a_long_non_moving_run_brackets_a_new_segment() {
        // Point 0->1: a fast hop (moving). Points 1,2,3 parked at the same
        // spot for 10 minutes total (above a 5-minute duration threshold).
        // Points 4,5 moving again.
        let lats = vec![48.0, 48.02, 48.02, 48.02, 48.03, 48.04];
        let lngs = vec![2.0; 6];
        let times = vec![0.0, 60.0, 180.0, 480.0, 540.0, 600.0];

        let ids = speed_segment_ids(&lats, &lngs, &times, 5.0, f64::INFINITY, 300.0);
        assert_eq!(ids[0], 0);
        assert_eq!(
            ids[1], 0,
            "the fast hop into the parked spot is still moving"
        );
        assert_eq!(ids[2], 1, "the parked run gets its own bracketing segment");
        assert_eq!(ids[3], 1);
        assert!(ids[4] > ids[1], "moving resumes in a fresh segment");
        assert_eq!(ids[4], ids[5]);
    }

    #[test]
    fn a_short_non_moving_run_does_not_split() {
        // A single stationary point between two moving legs, but the parked
        // duration (60s) is well under the 5-minute threshold.
        let lats = vec![48.0, 48.01, 48.01, 48.02];
        let lngs = vec![2.0; 4];
        let times = vec![0.0, 60.0, 120.0, 180.0];

        let ids = speed_segment_ids(&lats, &lngs, &times, 5.0, f64::INFINITY, 300.0);
        assert_eq!(ids, vec![0, 0, 0, 0]);
    }
}
