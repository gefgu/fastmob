//! Shared Wedge/cone feasibility primitive used by the Chan-Chin and
//! Imai-Iri simplification algorithms.
//!
//! Both algorithms need to answer the same question repeatedly: "starting
//! from anchor point `start`, how far along the trajectory can a single
//! straight replacement segment extend while staying within `tolerance` of
//! every point in between?" This module answers that question by
//! incrementally intersecting, for each candidate point, the cone of
//! directions through the anchor whose resulting line stays within
//! `tolerance` of that point (the "wedge"). The candidate segment remains
//! feasible for as long as the running wedge intersection is non-empty.

/// Shift `angle` by a multiple of 2*pi so it lands within `pi` of
/// `reference`. Used to keep wedge bounds and new candidate directions in
/// the same angular branch before intersecting them.
fn normalize_near(angle: f64, reference: f64) -> f64 {
    let two_pi = std::f64::consts::TAU;
    let mut normalized = angle;
    while normalized - reference > std::f64::consts::PI {
        normalized -= two_pi;
    }
    while normalized - reference < -std::f64::consts::PI {
        normalized += two_pi;
    }
    normalized
}

/// Return the farthest local index `j` (`start < j < coords.len()`) such
/// that a single straight segment from `coords[start]` to `coords[j]` stays
/// within `tolerance` of every point strictly between them, using an
/// incremental wedge/cone feasibility test.
///
/// Points within `tolerance` of the anchor impose no directional
/// constraint (any line through the anchor already satisfies the tolerance
/// for them). Points farther than `tolerance` narrow the feasible wedge to
/// `[theta - alpha, theta + alpha]` around the direction `theta` to that
/// point, where `alpha = asin(tolerance / distance)`. The segment is
/// feasible up to (and including) the last point processed before the
/// running wedge intersection becomes empty.
///
/// Returns `start` unchanged when `start` is the last index in `coords`.
///
/// @usedBy `fastmob-core/src/preprocessing/simplify/chan_chin.rs` (forward
/// and backward greedy passes) and
/// `fastmob-core/src/preprocessing/simplify/imai_iri.rs` (per-anchor
/// feasible-edge construction).
pub fn farthest_feasible_index(coords: &[(f64, f64)], start: usize, tolerance: f64) -> usize {
    let n = coords.len();
    if start + 1 >= n {
        return start;
    }

    let (ax, ay) = coords[start];
    let mut wedge: Option<(f64, f64)> = None;
    let mut farthest = start;

    for (k, &(px, py)) in coords.iter().enumerate().take(n).skip(start + 1) {
        let dx = px - ax;
        let dy = py - ay;
        let distance = (dx * dx + dy * dy).sqrt();

        if distance > tolerance {
            let mut theta = dy.atan2(dx);
            let alpha = (tolerance / distance).clamp(-1.0, 1.0).asin();

            let (lo, hi) = match wedge {
                None => (theta - alpha, theta + alpha),
                Some((lo, hi)) => {
                    let center = (lo + hi) / 2.0;
                    theta = normalize_near(theta, center);
                    (lo.max(theta - alpha), hi.min(theta + alpha))
                }
            };

            if lo > hi {
                return farthest;
            }
            wedge = Some((lo, hi));
        }

        farthest = k;
    }

    farthest
}
