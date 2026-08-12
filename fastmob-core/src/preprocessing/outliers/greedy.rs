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
/// ported from `SmartGreedyOutlierDetector.h`'s multi-hypothesis search.
///
/// Maintains a set of active chains, each represented *only* by its current
/// tail point-index and length (not the original's full `Vec<usize>`
/// history per chain). For every point `i`, every active tail consistent
/// with `i` is a candidate predecessor; among those, the one giving the
/// longest resulting chain is recorded in `pred[i]`. Critically, *every*
/// chain `i` extends is then collapsed into a single new chain at tail `i`
/// (the longest of them) rather than left coexisting as separate chains
/// that now redundantly share the same tail: any future point's
/// consistency with a tail depends only on that tail's own coordinates, so
/// once two chains share a tail, the shorter one can never do anything the
/// longer one can't, and keeping both only multiplies future redundant
/// work. This is also the actual root cause fixed here: the original
/// implementation *did not* collapse same-tail duplicates, so a point
/// extending `k` existing chains created `k` separate chains all ending at
/// the same point, and this could compound across many points -- observed
/// in practice as 70+GB of RSS and a 900s timeout on a 4M-row dense GPS
/// trace, not merely a large-but-linear memory bill.
///
/// This preserves the original's traversal semantics exactly (still scans
/// only *active tails* per point, matching its typical-case speed when
/// data isn't pathologically fragmented -- unlike a naive "scan every
/// prior point" reformulation, which was tried and measured 200x slower
/// than the original at 100k points specifically because it abandons that
/// tails-only pruning) while producing an identical maximal chain length,
/// since collapsing dominated same-tail duplicates never discards a chain
/// that could have led to a strictly better final answer. Memory is now
/// `O(active tails)` flat entries (`<= n`, and typically far smaller)
/// instead of unbounded nested `Vec`s. Worst-case time remains `O(n^2)` if
/// the active-tail count itself grows close to `n` (heavy fragmentation) --
/// inherent to this predicate without a spatial index -- but every
/// operation is now `O(1)` per tail instead of `O(history length)`.
///
/// Ties (multiple tails yielding the same candidate length) are broken by
/// keeping the *first* found while scanning tails in their existing order;
/// exact tie-break equivalence with the original for every possible input
/// is not guaranteed (ties are inherently implementation-defined for this
/// heuristic -- the upstream algorithm itself returns *every*
/// maximal-length chain and leaves the caller to pick one), but the
/// maximal chain length, and therefore the number of points kept, matches.
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

    // Active chains: parallel arrays of (tail point-index, chain length).
    let mut tails: Vec<usize> = Vec::new();
    let mut tail_len: Vec<usize> = Vec::new();
    let mut pred: Vec<Option<usize>> = vec![None; n];

    for i in 0..n {
        let mut best_len_here = 1usize;
        let mut best_pred: Option<usize> = None;
        let mut matched = vec![false; tails.len()];
        let mut any_matched = false;

        for (k, &t) in tails.iter().enumerate() {
            if is_consistent_pair(lats, lngs, times, t, i, max_speed_kmh) {
                matched[k] = true;
                any_matched = true;
                let candidate_len = tail_len[k] + 1;
                if candidate_len > best_len_here {
                    best_len_here = candidate_len;
                    best_pred = Some(t);
                }
            }
        }
        pred[i] = best_pred;

        if any_matched {
            let mut survivors_tails = Vec::with_capacity(tails.len());
            let mut survivors_len = Vec::with_capacity(tail_len.len());
            for (k, &t) in tails.iter().enumerate() {
                if !matched[k] {
                    survivors_tails.push(t);
                    survivors_len.push(tail_len[k]);
                }
            }
            tails = survivors_tails;
            tail_len = survivors_len;
        }
        tails.push(i);
        tail_len.push(best_len_here);
    }

    let best_k = (0..tails.len())
        .max_by_key(|&k| tail_len[k])
        .expect("n > 0, so at least one chain exists");
    let mut cur = tails[best_k];
    loop {
        keep[cur] = true;
        match pred[cur] {
            Some(prev) => cur = prev,
            None => break,
        }
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
