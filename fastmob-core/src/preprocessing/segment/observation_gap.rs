//! ObservationGap segmentation, ported from MovingPandas'
//! `ObservationGapSplitter` (`trajectory_splitter.py`).
//!
//! MovingPandas marks a new group whenever the time delta since the
//! previous point strictly exceeds `gap`; this module reproduces the same
//! cumulative-sum-of-boolean-gaps grouping, row-for-row.

/// ObservationGap segment ids for one user's chronologically-sorted slice.
///
/// Point `0` always starts segment `0`. For `i >= 1`, a new segment starts
/// at `i` whenever `times[i] - times[i-1] > gap_s`.
///
/// @usedBy `fastmob-core/src/preprocessing/segment/mod.rs::segment_user_slice`
/// (method = `observation_gap`).
pub fn observation_gap_segment_ids(times: &[f64], gap_s: f64) -> Vec<u32> {
    let n = times.len();
    let mut out = vec![0u32; n];
    if n == 0 {
        return out;
    }

    let mut group = 0u32;
    for i in 1..n {
        if times[i] - times[i - 1] > gap_s {
            group += 1;
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
        assert_eq!(observation_gap_segment_ids(&[], 60.0), Vec::<u32>::new());
        assert_eq!(observation_gap_segment_ids(&[0.0], 60.0), vec![0]);
    }

    #[test]
    fn a_gap_above_threshold_splits_into_two_segments() {
        // 3 points 60s apart, then a 1-hour gap, then 2 more points 60s apart.
        let times = vec![0.0, 60.0, 120.0, 3720.0, 3780.0];
        let ids = observation_gap_segment_ids(&times, 600.0);
        assert_eq!(ids, vec![0, 0, 0, 1, 1]);
    }

    #[test]
    fn a_gap_at_or_below_threshold_never_splits() {
        let times = vec![0.0, 60.0, 660.0, 720.0];
        let ids = observation_gap_segment_ids(&times, 600.0);
        assert_eq!(ids, vec![0, 0, 0, 0]);
    }
}
