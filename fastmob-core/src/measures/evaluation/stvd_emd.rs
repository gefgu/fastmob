use crate::utils::haversine::haversine_km;
use ndarray::{Array1, Array2};
use std::f64::consts::PI;
use wass::earth_mover_distance;

/// Shortest angular distance between two angles (radians), independent of winding direction.
fn angular_diff(theta1: f64, theta2: f64) -> f64 {
    let diff = (theta1 - theta2).abs() % (2.0 * PI);
    diff.min(2.0 * PI - diff)
}

/// Cyclical-time chordal distance (metres) between two timestamps, matching the
/// `r * (cos theta, sin theta)` embedding this module used before switching to an explicit
/// cost matrix: two points on a circle of radius `r` separated by angle `d_theta` are
/// `2 * r * sin(d_theta / 2)` apart in a straight line.
fn temporal_chord_m(t1: f64, t2: f64, cyclical_period: f64, r: f64) -> f64 {
    let theta1 = 2.0 * PI * (t1.rem_euclid(cyclical_period) / cyclical_period);
    let theta2 = 2.0 * PI * (t2.rem_euclid(cyclical_period) / cyclical_period);
    2.0 * r * (angular_diff(theta1, theta2) / 2.0).sin()
}

#[allow(clippy::too_many_arguments)]
pub fn stvd_emd_impl(
    lats_a: &[f64],
    lngs_a: &[f64],
    ts_a: &[f64],
    ws_a: &[f64],
    lats_b: &[f64],
    lngs_b: &[f64],
    ts_b: &[f64],
    ws_b: &[f64],
    alpha: f64,
    cyclical_period: f64,
) -> Result<f64, String> {
    let n = lats_a.len();
    let m = lats_b.len();

    if n == 0 || m == 0 {
        return Err("distributions must be non-empty".to_string());
    }
    if lngs_a.len() != n || ts_a.len() != n || ws_a.len() != n {
        return Err("all arrays for distribution A must have the same length".to_string());
    }
    if lngs_b.len() != m || ts_b.len() != m || ws_b.len() != m {
        return Err("all arrays for distribution B must have the same length".to_string());
    }
    if cyclical_period <= 0.0 {
        return Err("cyclical_period must be positive".to_string());
    }

    let sum_a: f64 = ws_a.iter().sum();
    let sum_b: f64 = ws_b.iter().sum();
    if sum_a <= 0.0 || sum_b <= 0.0 {
        return Err("weights must sum to a positive value".to_string());
    }

    let r = (alpha * cyclical_period) / (2.0 * PI);

    let mut cost = Array2::<f32>::zeros((n, m));
    for i in 0..n {
        for j in 0..m {
            let spatial_m = haversine_km(lats_a[i], lngs_a[i], lats_b[j], lngs_b[j]) * 1000.0;
            let temporal_m = temporal_chord_m(ts_a[i], ts_b[j], cyclical_period, r);
            cost[[i, j]] = (spatial_m.powi(2) + temporal_m.powi(2)).sqrt() as f32;
        }
    }

    let a: Array1<f32> = ws_a.iter().map(|&w| w as f32).collect();
    let b: Array1<f32> = ws_b.iter().map(|&w| w as f32).collect();

    Ok(earth_mover_distance(&a, &b, &cost) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_single_point_is_zero_distance() {
        let value = stvd_emd_impl(
            &[10.0],
            &[20.0],
            &[480.0],
            &[1.0],
            &[10.0],
            &[20.0],
            &[480.0],
            &[1.0],
            10.0,
            1440.0,
        )
        .unwrap();
        assert!(value.abs() < 1e-6, "got {value}");
    }

    #[test]
    fn spatial_displacement_matches_haversine_distance() {
        // Same time bin on both sides so only the spatial term contributes.
        let lat1 = 0.0;
        let lng1 = 0.0;
        let lat2 = 0.0;
        let lng2 = 1.0; // ~111.19 km at the equator
        let expected_m = haversine_km(lat1, lng1, lat2, lng2) * 1000.0;

        let value = stvd_emd_impl(
            &[lat1],
            &[lng1],
            &[480.0],
            &[1.0],
            &[lat2],
            &[lng2],
            &[480.0],
            &[1.0],
            10.0,
            1440.0,
        )
        .unwrap();
        // Sinkhorn is an entropy-regularised approximation, not exact, so allow slack.
        assert!(
            (value - expected_m).abs() / expected_m < 0.05,
            "got {value}, expected close to {expected_m}"
        );
    }

    #[test]
    fn cyclical_time_wraps_around() {
        // 1 minute before midnight vs 1 minute after midnight should be "close" (2 minutes
        // apart), not ~1438 minutes apart, when cyclical_period=1440.
        let near = stvd_emd_impl(
            &[0.0],
            &[0.0],
            &[1439.0],
            &[1.0],
            &[0.0],
            &[0.0],
            &[1.0],
            &[1.0],
            10.0,
            1440.0,
        )
        .unwrap();
        let far = stvd_emd_impl(
            &[0.0],
            &[0.0],
            &[1439.0],
            &[1.0],
            &[0.0],
            &[0.0],
            &[700.0],
            &[1.0],
            10.0,
            1440.0,
        )
        .unwrap();
        assert!(near < far, "near={near} far={far}");
    }

    #[test]
    fn empty_distribution_is_rejected() {
        assert!(stvd_emd_impl(&[], &[], &[], &[], &[0.0], &[0.0], &[0.0], &[1.0], 10.0, 1440.0).is_err());
    }

    #[test]
    fn non_positive_cyclical_period_is_rejected() {
        assert!(stvd_emd_impl(
            &[0.0], &[0.0], &[0.0], &[1.0], &[0.0], &[0.0], &[0.0], &[1.0], 10.0, 0.0,
        )
        .is_err());
    }

    #[test]
    fn zero_weight_sum_is_rejected() {
        assert!(stvd_emd_impl(
            &[0.0], &[0.0], &[0.0], &[0.0], &[0.0], &[0.0], &[0.0], &[1.0], 10.0, 1440.0,
        )
        .is_err());
    }
}
