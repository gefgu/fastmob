//! Zheng et al. physically-consistent-trajectory outlier detector, ported
//! from MoveTK's `ZhengOutlierDetector.h`
//! (`movetk::outlierdetection::zheng_outlier_detector_tag`; same SIGSPATIAL
//! 2019 "Maximum Physically Consistent Trajectories" paper as
//! `greedy.rs::smart_greedy_keep_mask`).
//!
//! Documented discrepancy versus the literal upstream source: the C++ loop
//! in `ZhengOutlierDetector.h` declares `auto prev = first, it =
//! std::next(first)` once and only increments `it` in the loop's increment
//! clause (`it++`) — `prev` is never reassigned anywhere in the loop body,
//! so every predicate call in that implementation compares the *first*
//! element of the whole input range against the current element, not the
//! immediately preceding one. That contradicts the class's own doc comment
//! ("for elements at index i and i-1, the predicate holds"), which strongly
//! suggests the fixed-`prev` behavior is an upstream bug rather than
//! intentional. This port follows the documented adjacent-pair intent
//! instead; if a real MoveTK-driver parity test disagrees with this kernel,
//! that is the expected, disclosed cause (see the project plan for the
//! oracle-comparison writeup).

use super::greedy::is_consistent_pair;

/// Zheng keep-mask for one user's point sequence: a single forward pass
/// groups points into maximal runs where every adjacent pair satisfies
/// [`is_consistent_pair`] (speed implied by consecutive points does not
/// exceed `max_speed_kmh`); runs strictly longer than `min_seg_size` points
/// are kept in full, shorter runs are dropped entirely.
///
/// Sequences of 0 or 1 points are always kept unchanged: there is no
/// meaningful notion of run length to compare against `min_seg_size` for a
/// single point, and dropping a lone point here would contradict every
/// other shipped algorithm's convention of leaving trivial trajectories
/// untouched.
///
/// @usedBy `fastmob-core/src/preprocessing/outliers/mod.rs::outlier_user_slice`
/// (method = `zheng`).
pub fn zheng_keep_mask(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    max_speed_kmh: f64,
    min_seg_size: usize,
) -> Vec<bool> {
    let n = lats.len();
    let mut keep = vec![false; n];
    if n <= 1 {
        keep.fill(true);
        return keep;
    }

    let mut range_start = 0usize;
    let mut segment_size = 1usize;

    for i in 1..n {
        if is_consistent_pair(lats, lngs, times, i - 1, i, max_speed_kmh) {
            segment_size += 1;
        } else {
            if segment_size > min_seg_size {
                keep[range_start..i].fill(true);
            }
            range_start = i;
            segment_size = 1;
        }
    }
    if segment_size > min_seg_size {
        keep[range_start..n].fill(true);
    }

    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn times_seq(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 * 3600.0).collect()
    }

    #[test]
    fn drops_short_segments_and_keeps_long_ones() {
        // A slow run of 4 points, a lone teleport, then a slow run of 3
        // points. With min_seg_size=1 the lone middle point's 1-length
        // "segment" is dropped while both surrounding runs survive.
        let mut lats: Vec<f64> = (0..4).map(|i| i as f64 * 0.001).collect();
        lats.push(5.0); // index 4: teleport, breaks consistency both ways
        lats.extend((0..3).map(|i| 0.004 + i as f64 * 0.001));
        let lngs = vec![0.0; lats.len()];
        let times = times_seq(lats.len());

        let keep = zheng_keep_mask(&lats, &lngs, &times, 10.0, 1);
        assert_eq!(keep, vec![true, true, true, true, false, true, true, true]);
    }

    #[test]
    fn min_seg_size_drops_a_run_at_or_below_threshold() {
        // Two isolated consistent 2-point runs separated by a teleport;
        // with min_seg_size=2, neither 2-point run is long enough to keep.
        let lats = vec![0.0, 0.001, 5.0, 5.001];
        let lngs = vec![0.0; 4];
        let times = times_seq(4);

        let keep = zheng_keep_mask(&lats, &lngs, &times, 10.0, 2);
        assert_eq!(keep, vec![false, false, false, false]);
    }

    #[test]
    fn trivial_sequences_always_kept() {
        assert_eq!(zheng_keep_mask(&[], &[], &[], 10.0, 1), Vec::<bool>::new());
        assert_eq!(zheng_keep_mask(&[0.0], &[0.0], &[0.0], 10.0, 1), vec![true]);
    }
}
