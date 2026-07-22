//! Hampel filter outlier detection over the per-user consecutive-point
//! speed (km/h) series.
//!
//! Ported to match the `hampel` PyPI package's Cython kernel
//! (`hampel.extension.hampel`), which is the exact implementation PTRAIL's
//! `Filters.hampel_outlier_detection` delegates to (via `from hampel import
//! hampel`) — matching it precisely keeps the cache-based PTRAIL parity test
//! (see the project plan) meaningful rather than approximate.

use crate::utils::haversine::haversine_km;

/// Compute the per-user consecutive-point speed (km/h) series: `speeds[0]`
/// is `0.0` (no previous point to measure a speed from) and `speeds[i]` for
/// `i >= 1` is the haversine distance between points `i - 1` and `i` divided
/// by their time delta in hours. Non-positive time deltas (duplicate or
/// out-of-order timestamps) are treated as `0.0` speed rather than
/// propagating `inf`/`NaN` into the rolling window statistics below.
fn consecutive_point_speeds_kmh(lats: &[f64], lngs: &[f64], times: &[f64]) -> Vec<f64> {
    let n = lats.len();
    let mut speeds = vec![0.0f64; n];
    for i in 1..n {
        let dt = times[i] - times[i - 1];
        if dt > 0.0 {
            let dist = haversine_km(lats[i - 1], lngs[i - 1], lats[i], lngs[i]);
            speeds[i] = dist / dt * 3600.0;
        }
    }
    speeds
}

/// Return the median of `window`, sorting it in place.
///
/// Every window this module builds has an odd length
/// (`2 * half_window + 1`), so the median is always the sorted middle
/// element — no even-length two-element averaging is needed.
fn window_median(window: &mut [f64]) -> f64 {
    window.sort_by(|a, b| a.total_cmp(b));
    window[window.len() / 2]
}

/// Hampel filter keep-mask for one user's point sequence.
///
/// Uses a centered rolling window of `window_size` points
/// (`half_window = window_size / 2`, integer division); a point at index `i`
/// is flagged as an outlier when `|speed[i] - median(window)| > n_sigma *
/// 1.4826 * MAD(window)`, where `window` is the `2 * half_window + 1`
/// speeds centered on `i`.
///
/// Points within `half_window` of either end of the sequence — including
/// index 0, which has no defined incoming speed — are never evaluated and
/// are always kept, exactly matching the reference kernel's
/// `range(half_window, n - half_window)` loop bounds (no partial/clipped
/// windows at the boundary).
///
/// @usedBy `fastmob-core/src/preprocessing/outliers/mod.rs::outlier_user_slice`
/// (method = `hampel`).
pub fn hampel_keep_mask(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    window_size: usize,
    n_sigma: f64,
) -> Vec<bool> {
    let n = lats.len();
    let mut keep = vec![true; n];
    if n == 0 {
        return keep;
    }

    let speeds = consecutive_point_speeds_kmh(lats, lngs, times);
    let half_window = window_size / 2;
    if half_window >= n {
        return keep;
    }

    let mut window = vec![0.0f64; 2 * half_window + 1];
    for i in half_window..(n - half_window) {
        window.copy_from_slice(&speeds[i - half_window..=i + half_window]);
        let median = window_median(&mut window);
        for w in window.iter_mut() {
            *w = (*w - median).abs();
        }
        let mad = window_median(&mut window);
        let threshold = n_sigma * 1.4826 * mad;
        if (speeds[i] - median).abs() > threshold {
            keep[i] = false;
        }
    }

    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    fn times_seq(n: usize) -> Vec<f64> {
        (0..n).map(|i| i as f64 * 60.0).collect()
    }

    #[test]
    fn keeps_everything_when_sequence_shorter_than_window() {
        let lats = vec![0.0, 0.001, 0.002];
        let lngs = vec![0.0, 0.0, 0.0];
        let times = times_seq(3);
        let keep = hampel_keep_mask(&lats, &lngs, &times, 5, 3.0);
        assert_eq!(keep, vec![true, true, true]);
    }

    #[test]
    fn flags_a_single_speed_spike_in_the_middle() {
        // Steady ~1 km/min points, except point 5 teleports far away and
        // back, producing two speed spikes at indices 5 and 6.
        let mut lats = vec![0.0; 11];
        for (i, lat) in lats.iter_mut().enumerate() {
            *lat = i as f64 * 0.01;
        }
        lats[5] = 10.0; // huge jump in latitude → huge implied speed
        let lngs = vec![0.0; 11];
        let times = times_seq(11);

        let keep = hampel_keep_mask(&lats, &lngs, &times, 5, 3.0);
        assert!(!keep[5], "the teleported point itself should be flagged");
        assert!(
            !keep[6],
            "the point right after should also see a speed spike"
        );
        assert!(keep[0] && keep[1], "boundary points are never evaluated");
    }
}
