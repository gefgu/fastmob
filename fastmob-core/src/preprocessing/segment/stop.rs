//! Stop segmentation, adapted from MovingPandas' `StopSplitter`
//! (`trajectory_splitter.py`).
//!
//! Reuses [`detect_stops_for_user`] directly (per the project plan's reuse
//! note) rather than reimplementing stop detection. MovingPandas'
//! `StopSplitter` builds "between stops" trajectories and drops every row
//! strictly inside a detected stop's `[entry_time, leaving_time]` window
//! (see `StopSplitter.get_time_ranges_between_stops`). fastmob's
//! segmentation has a hard row-preservation contract, so instead of
//! dropping those rows, this module gives them their own segment id that
//! brackets the moving segment before the stop and the moving segment
//! after it — i.e. every row is kept, and a detected stop is represented as
//! a segment of its own rather than a gap.

use crate::preprocessing::stay_locations::detect_stops_for_user;

/// Stop segment ids for one user's chronologically-sorted slice.
///
/// Runs [`detect_stops_for_user`] once to get the user's `Stop` entry/leaving
/// time windows, then walks the slice in time order: point `0` starts
/// segment `0`; every time a point's membership in "inside some detected
/// stop's `[entry_time_s, leaving_time_s]` window" flips relative to the
/// previous point, a new segment id starts. This yields alternating
/// moving/stop segments with the detected stop's own rows isolated in their
/// own segment, exactly bracketed by the moving segments on either side.
///
/// @usedBy `fastmob-core/src/preprocessing/segment/mod.rs::segment_user_slice`
/// (method = `stop`).
#[allow(clippy::too_many_arguments)]
pub fn stop_segment_ids(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    stop_radius_km: f64,
    minutes_for_a_stop: f64,
    no_data_for_minutes: f64,
    min_speed_kmh: f64,
) -> Vec<u32> {
    let n = lats.len();
    let mut out = vec![0u32; n];
    if n == 0 {
        return out;
    }

    // include_last=true: segmentation needs the trailing open stay closed
    // like every other detected stop, unlike generate_staypoints() where it's
    // now a user-facing choice (see stay_locations.rs).
    let stops = detect_stops_for_user(
        lats,
        lngs,
        times,
        stop_radius_km,
        minutes_for_a_stop,
        no_data_for_minutes,
        min_speed_kmh,
        true,
    );
    if stops.is_empty() {
        return out;
    }

    let mut group = 0u32;
    let mut stop_idx = 0usize;
    let mut in_stop = false;

    for i in 0..n {
        while stop_idx < stops.len() && times[i] > stops[stop_idx].leaving_time_s {
            stop_idx += 1;
        }
        let currently_in_stop = stop_idx < stops.len()
            && times[i] >= stops[stop_idx].entry_time_s
            && times[i] <= stops[stop_idx].leaving_time_s;

        if i == 0 {
            in_stop = currently_in_stop;
            out[0] = 0;
            continue;
        }

        if currently_in_stop != in_stop {
            group += 1;
            in_stop = currently_in_stop;
        }
        out[i] = group;
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_single_point_inputs() {
        assert_eq!(
            stop_segment_ids(&[], &[], &[], 0.2, 20.0, 1e12, f64::INFINITY),
            Vec::<u32>::new()
        );
        assert_eq!(
            stop_segment_ids(&[0.0], &[0.0], &[0.0], 0.2, 20.0, 1e12, f64::INFINITY),
            vec![0]
        );
    }

    #[test]
    fn a_detected_stop_is_bracketed_by_two_moving_segments() {
        // Moving away from Paris for 2 points (each hop too short-lived to
        // register as a stop), then staying within a small radius for 40
        // minutes (well above a 20-minute stop threshold), then moving away
        // again with one further trailing point (so the stop's exit is not
        // conflated with `detect_stops_for_user`'s special "is_last" closing
        // behavior at the very last row of the slice).
        let lats = vec![48.85, 48.86, 48.90, 48.9001, 48.9002, 49.0, 49.5];
        let lngs = vec![2.0, 2.1, 2.5, 2.5001, 2.5002, 3.0, 3.5];
        let times = vec![0.0, 600.0, 1200.0, 1800.0, 3000.0, 3600.0, 4200.0];

        let ids = stop_segment_ids(&lats, &lngs, &times, 0.2, 20.0, 1e12, f64::INFINITY);
        assert_eq!(ids, vec![0, 0, 1, 1, 1, 1, 2]);
    }

    #[test]
    fn no_detected_stops_keeps_everything_in_one_segment() {
        // Steady fast movement the whole time: never satisfies the stop
        // radius/duration criteria.
        let lats: Vec<f64> = (0..6).map(|i| 48.0 + i as f64 * 0.5).collect();
        let lngs = vec![2.0; 6];
        let times: Vec<f64> = (0..6).map(|i| i as f64 * 600.0).collect();

        let ids = stop_segment_ids(&lats, &lngs, &times, 0.2, 20.0, 1e12, f64::INFINITY);
        assert_eq!(ids, vec![0, 0, 0, 0, 0, 0]);
    }
}
