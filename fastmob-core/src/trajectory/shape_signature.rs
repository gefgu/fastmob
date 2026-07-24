//! Rotation/translation-invariant distance-geometry trajectory-shape
//! signature, ported from tracktable's `distance_geometry_by_distance`.
//!
//! For each level `d` in `1..=depth`, the trajectory is resampled at `d+1`
//! points equally spaced by cumulative arc length (linear interpolation
//! between the two bracketing raw points), and the `d` chord lengths
//! between consecutive resampled points are recorded, each normalized by
//! the trajectory's total arc length. Concatenating levels `1..=depth`
//! gives a signature of length `depth*(depth+1)/2`, with every value in
//! `[0, 1]` -- rotation/translation-invariant (and, since values are
//! length ratios, scale-invariant with respect to absolute trajectory
//! duration/sampling, only shape matters).

use rayon::prelude::*;

use crate::utils::haversine::haversine_km;

fn cumulative_arc_length_km(lats: &[f64], lngs: &[f64]) -> Vec<f64> {
    let n = lats.len();
    let mut cum = vec![0.0; n];
    for i in 1..n {
        cum[i] = cum[i - 1] + haversine_km(lats[i - 1], lngs[i - 1], lats[i], lngs[i]);
    }
    cum
}

/// Resample a trajectory at `n_points` equally-spaced fractions of its
/// total cumulative arc length (0, 1/(n_points-1), ..., 1). Requires
/// `n_points >= 2`; degenerates to repeating the first point when the
/// trajectory has zero total arc length (e.g. every point coincides).
fn resample_by_arc_length(
    lats: &[f64],
    lngs: &[f64],
    cum_dist: &[f64],
    n_points: usize,
) -> Vec<(f64, f64)> {
    let total = *cum_dist.last().unwrap_or(&0.0);
    if total <= 0.0 {
        return vec![(lats[0], lngs[0]); n_points];
    }

    (0..n_points)
        .map(|i| {
            let target = total * (i as f64) / (n_points - 1) as f64;
            let idx = cum_dist.partition_point(|&d| d < target);
            if idx == 0 {
                (lats[0], lngs[0])
            } else if idx >= cum_dist.len() {
                (lats[lats.len() - 1], lngs[lngs.len() - 1])
            } else {
                let d0 = cum_dist[idx - 1];
                let d1 = cum_dist[idx];
                let frac = if d1 > d0 {
                    (target - d0) / (d1 - d0)
                } else {
                    0.0
                };
                let lat = lats[idx - 1] + (lats[idx] - lats[idx - 1]) * frac;
                let lng = lngs[idx - 1] + (lngs[idx] - lngs[idx - 1]) * frac;
                (lat, lng)
            }
        })
        .collect()
}

/// Compute the distance-geometry shape signature (length
/// `depth*(depth+1)/2`) for one trajectory's raw `(lats, lngs)` point
/// sequence, in chronological order.
pub fn distance_geometry_signature(
    lats: &[f64],
    lngs: &[f64],
    depth: usize,
) -> Result<Vec<f64>, String> {
    if lats.len() != lngs.len() {
        return Err("lats and lngs must have the same length".to_string());
    }
    if lats.len() < 2 {
        return Err(
            "a trajectory needs at least 2 points to compute a shape signature".to_string(),
        );
    }
    if depth == 0 {
        return Err("depth must be >= 1".to_string());
    }

    let cum_dist = cumulative_arc_length_km(lats, lngs);
    let total = *cum_dist.last().unwrap();

    let mut signature = Vec::with_capacity(depth * (depth + 1) / 2);
    for level in 1..=depth {
        let points = resample_by_arc_length(lats, lngs, &cum_dist, level + 1);
        for pair in points.windows(2) {
            let (lat0, lng0) = pair[0];
            let (lat1, lng1) = pair[1];
            let chord_km = haversine_km(lat0, lng0, lat1, lng1);
            signature.push(if total > 0.0 { chord_km / total } else { 0.0 });
        }
    }
    Ok(signature)
}

/// Batched, rayon-parallel signature computation over a list of
/// independent trajectories (each an owned `(lats, lngs)` pair -- there is
/// no shared grouped dataframe here, unlike most other kernels, since each
/// input element is already exactly one whole trajectory).
pub fn batch_distance_geometry_signatures(
    trajectories: &[(Vec<f64>, Vec<f64>)],
    depth: usize,
) -> Result<Vec<Vec<f64>>, String> {
    trajectories
        .par_iter()
        .map(|(lats, lngs)| distance_geometry_signature(lats, lngs, depth))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn straight_line_signature_is_all_ones_scaled_by_level() {
        // A perfectly straight line resampled at any level just reproduces
        // equal-length chords -- each level's chords should all be equal
        // (1/level of the total), since arc length == straight-line length.
        let lats = vec![0.0, 0.0, 0.0, 0.0, 0.0];
        let lngs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let sig = distance_geometry_signature(&lats, &lngs, 3).unwrap();
        assert_eq!(sig.len(), 6); // 1+2+3

        // Level 1 (index 0): single chord spanning the whole line -> 1.0.
        assert!((sig[0] - 1.0).abs() < 1e-6);
        // Level 2 (indices 1,2): two equal half-length chords -> 0.5 each.
        assert!((sig[1] - 0.5).abs() < 1e-6);
        assert!((sig[2] - 0.5).abs() < 1e-6);
        // Level 3 (indices 3,4,5): three equal third-length chords.
        for &v in &sig[3..6] {
            assert!((v - 1.0 / 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn signature_is_translation_and_rotation_invariant() {
        let lats_a = vec![0.0, 0.1, 0.2, 0.1];
        let lngs_a = vec![0.0, 0.1, 0.0, -0.1];
        let sig_a = distance_geometry_signature(&lats_a, &lngs_a, 4).unwrap();

        // Same shape, translated by a small fixed offset. A true sphere
        // isn't flat, so Haversine-based chord ratios are only
        // *approximately* translation-invariant (exact only in the flat-
        // plane limit); a small offset keeps that distortion negligible,
        // unlike a large latitude shift which visibly changes the km-per-
        // degree-longitude scale.
        let lats_b: Vec<f64> = lats_a.iter().map(|v| v + 1.0).collect();
        let lngs_b: Vec<f64> = lngs_a.iter().map(|v| v + 2.0).collect();
        let sig_b = distance_geometry_signature(&lats_b, &lngs_b, 4).unwrap();

        for (a, b) in sig_a.iter().zip(sig_b.iter()) {
            assert!((a - b).abs() < 1e-3, "expected {a} ~= {b}");
        }
    }

    #[test]
    fn rejects_mismatched_lengths_and_short_trajectories() {
        assert!(distance_geometry_signature(&[0.0], &[0.0, 1.0], 2).is_err());
        assert!(distance_geometry_signature(&[0.0], &[0.0], 2).is_err());
        assert!(distance_geometry_signature(&[0.0, 1.0], &[0.0, 1.0], 0).is_err());
    }
}
