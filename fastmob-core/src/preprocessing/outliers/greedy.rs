//! Greedy and SmartGreedy outlier detectors, ported from MoveTK's
//! `GreedyOutlierDetector.h` / `SmartGreedyOutlierDetector.h`
//! (`movetk::outlierdetection::greedy_outlier_detector_tag` /
//! `smart_greedy_outlier_detector_tag`; B. Custers, M. van de Kerkhof,
//! W. Meulemans, B. Speckmann & F. Staals (2019), "Maximum Physically
//! Consistent Trajectories", SIGSPATIAL 2019).

use crate::utils::haversine::haversine_km;

/// Speed-bounded physical-consistency predicate shared by Greedy,
/// SmartGreedy, and Zheng: two points are "consistent" when the constant
/// speed implied by their haversine distance and time delta does not exceed
/// `max_speed_kmh`. Non-positive time deltas (duplicate/out-of-order
/// timestamps) are always inconsistent.
pub(crate) fn is_consistent_pair(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    i: usize,
    j: usize,
    max_speed_kmh: f64,
) -> bool {
    let dt = times[j] - times[i];
    if dt <= 0.0 {
        return false;
    }
    let dist = haversine_km(lats[i], lngs[i], lats[j], lngs[j]);
    dist / dt * 3600.0 <= max_speed_kmh
}

/// Greedy outlier detector keep-mask for one user's point sequence.
///
/// The first point is always kept. Every subsequent point `i` is kept iff
/// [`is_consistent_pair`] holds between the immediately preceding
/// *original* point `i - 1` and `i` — matching `GreedyOutlierDetector.h`'s
/// documented behavior ("The first element of the range is always added.
/// An arbitrary element is added if the previous element and the element
/// satisfy the predicate"), i.e. a fixed per-original-adjacent-pair test
/// rather than a running "last accepted point" anchor.
///
/// @usedBy `fastmob-core/src/preprocessing/outliers/mod.rs::outlier_user_slice`
/// (method = `greedy`).
pub fn greedy_keep_mask(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    max_speed_kmh: f64,
) -> Vec<bool> {
    let n = lats.len();
    let mut keep = vec![false; n];
    if n == 0 {
        return keep;
    }
    keep[0] = true;
    for (i, keep_i) in keep.iter_mut().enumerate().skip(1) {
        *keep_i = is_consistent_pair(lats, lngs, times, i - 1, i, max_speed_kmh);
    }
    keep
}

/// SmartGreedy outlier detector keep-mask for one user's point sequence,
/// ported from `SmartGreedyOutlierDetector.h`'s `O(n*k)` multi-hypothesis
/// search (`k` = number of concurrently tracked candidate subsequences).
///
/// Maintains a set of candidate index chains, each internally consistent
/// under [`is_consistent_pair`]. For every point, every existing chain
/// whose last point is consistent with it is extended by it (a single point
/// may extend more than one chain simultaneously); if it extends no chain,
/// it starts a new one. After scanning the whole slice, the *longest* chain
/// is kept (ties broken by first-found order: the upstream algorithm
/// returns every maximal-length chain and leaves the caller to pick one,
/// but a single keep-mask needs exactly one).
///
/// @usedBy `fastmob-core/src/preprocessing/outliers/mod.rs::outlier_user_slice`
/// (method = `smart_greedy`).
pub fn smart_greedy_keep_mask(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    max_speed_kmh: f64,
) -> Vec<bool> {
    let n = lats.len();
    let mut keep = vec![false; n];
    if n == 0 {
        return keep;
    }

    let mut chains: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        let mut extended = false;
        for chain in chains.iter_mut() {
            let &last = chain.last().expect("chains are never empty");
            if is_consistent_pair(lats, lngs, times, last, i, max_speed_kmh) {
                chain.push(i);
                extended = true;
            }
        }
        if !extended {
            chains.push(vec![i]);
        }
    }

    let best = chains
        .iter()
        .max_by_key(|chain| chain.len())
        .expect("at least one chain exists when n > 0");
    for &idx in best {
        keep[idx] = true;
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
    fn greedy_keeps_first_point_and_rejects_a_single_teleport() {
        // 1 degree latitude ~= 111 km; each hour step moving 0.001 deg is a
        // ~0.11 km/h pace, well under any reasonable threshold, except the
        // point at index 2 teleports far away.
        let lats = vec![0.0, 0.001, 5.0, 0.003, 0.004];
        let lngs = vec![0.0; 5];
        let times = times_seq(5);

        let keep = greedy_keep_mask(&lats, &lngs, &times, 10.0);
        assert!(keep[0]);
        assert!(
            !keep[2],
            "teleported point breaks the adjacent-pair speed test"
        );
    }

    #[test]
    fn greedy_empty_and_single_point_inputs() {
        assert_eq!(greedy_keep_mask(&[], &[], &[], 10.0), Vec::<bool>::new());
        assert_eq!(greedy_keep_mask(&[0.0], &[0.0], &[0.0], 10.0), vec![true]);
    }

    #[test]
    fn smart_greedy_picks_the_longest_consistent_chain() {
        // Points 0,1,3,4 form a slow, consistent chain; point 2 is a
        // teleport that is inconsistent with its neighbors.
        let lats = vec![0.0, 0.001, 5.0, 0.003, 0.004];
        let lngs = vec![0.0; 5];
        let times = times_seq(5);

        let keep = smart_greedy_keep_mask(&lats, &lngs, &times, 10.0);
        assert_eq!(keep, vec![true, true, false, true, true]);
    }

    #[test]
    fn smart_greedy_empty_and_single_point_inputs() {
        assert_eq!(
            smart_greedy_keep_mask(&[], &[], &[], 10.0),
            Vec::<bool>::new()
        );
        assert_eq!(
            smart_greedy_keep_mask(&[0.0], &[0.0], &[0.0], 10.0),
            vec![true]
        );
    }
}
