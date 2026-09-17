use crate::utils::haversine::haversine_km;
use ndarray::{Array1, Array2};
use rayon::prelude::*;
use std::f64::consts::PI;
use wass::{sinkhorn_log, sinkhorn_log_with_convergence};

/// Sinkhorn solver knobs. `wass::earth_mover_distance` hardcodes
/// `reg=0.01, max_iter=200` with no convergence check; this exposes those as
/// real parameters instead, matching the defaults exactly for callers that
/// don't override them.
#[derive(Clone, Copy, Debug)]
pub struct SinkhornConfig {
    pub reg: f64,
    pub max_iter: usize,
    pub tol: Option<f64>,
}

impl Default for SinkhornConfig {
    fn default() -> Self {
        Self { reg: 0.01, max_iter: 200, tol: None }
    }
}

/// Safety budget for the dense complete-bipartite-graph path: above this,
/// `n*m*4` bytes for the cost matrix -- doubled, since `wass::sinkhorn_log`/
/// `sinkhorn_log_with_convergence` additionally build and return a
/// same-sized coupling/"plan" array -- would risk an OOM kill rather than a
/// clean, actionable error. A sparse candidate-graph path for inputs above
/// this budget is tracked separately; until it lands, oversized inputs must
/// be reduced by the caller (e.g. coarser H3 resolution or time bins).
const DENSE_COST_MEMORY_BUDGET_BYTES: u64 = 4 * 1024 * 1024 * 1024; // 4 GiB

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

/// Cost (metres) between one point of distribution A and one of distribution
/// B: a Euclidean combination of great-circle spatial distance and cyclical-
/// time chordal distance. Shared by the dense complete-bipartite-graph path
/// (below) and the H3 candidate-graph path (`stvd_candidate_graph.rs`) so
/// the two can never numerically drift apart.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pair_cost_m(
    lat_a: f64,
    lng_a: f64,
    t_a: f64,
    lat_b: f64,
    lng_b: f64,
    t_b: f64,
    cyclical_period: f64,
    r: f64,
) -> f32 {
    let spatial_m = haversine_km(lat_a, lng_a, lat_b, lng_b) * 1000.0;
    let temporal_m = temporal_chord_m(t_a, t_b, cyclical_period, r);
    (spatial_m.powi(2) + temporal_m.powi(2)).sqrt() as f32
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
    sinkhorn: SinkhornConfig,
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
    if !alpha.is_finite() || !cyclical_period.is_finite() {
        return Err("alpha and cyclical_period must be finite".to_string());
    }
    if lats_a
        .iter()
        .chain(lngs_a)
        .chain(ts_a)
        .chain(lats_b)
        .chain(lngs_b)
        .chain(ts_b)
        .any(|value| !value.is_finite())
    {
        return Err("coordinates and times must be finite".to_string());
    }
    if ws_a
        .iter()
        .chain(ws_b)
        .any(|weight| !weight.is_finite() || *weight < 0.0)
    {
        return Err("weights must be finite and non-negative".to_string());
    }
    if sinkhorn.reg <= 0.0 || !sinkhorn.reg.is_finite() {
        return Err("sinkhorn reg must be positive and finite".to_string());
    }
    if sinkhorn.max_iter == 0 {
        return Err("sinkhorn max_iter must be positive".to_string());
    }
    if let Some(tol) = sinkhorn.tol
        && (tol <= 0.0 || !tol.is_finite())
    {
        return Err("sinkhorn tol must be positive and finite".to_string());
    }

    let sum_a: f64 = ws_a.iter().sum();
    let sum_b: f64 = ws_b.iter().sum();
    if sum_a <= 0.0 || sum_b <= 0.0 {
        return Err("weights must sum to a positive value".to_string());
    }

    let r = (alpha * cyclical_period) / (2.0 * PI);

    let cost_len = n
        .checked_mul(m)
        .ok_or_else(|| "cost matrix dimensions overflow addressable memory".to_string())?;
    let dense_bytes = (cost_len as u64).saturating_mul(2 * std::mem::size_of::<f32>() as u64);
    if dense_bytes > DENSE_COST_MEMORY_BUDGET_BYTES {
        return Err(format!(
            "distributions are too large for stvd_emd's dense Sinkhorn path: {n}x{m} would need \
             ~{:.1} GiB (cost matrix plus the solver's internal coupling matrix), which exceeds \
             the {:.1} GiB safety budget. Reduce distribution size before calling stvd_emd (e.g. \
             a coarser H3 resolution when building locations, or coarser time bins).",
            dense_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            DENSE_COST_MEMORY_BUDGET_BYTES as f64 / (1024.0 * 1024.0 * 1024.0),
        ));
    }
    let mut cost_values = vec![0.0_f32; cost_len];
    // Each cost row is independent. Parallel construction moves the expensive
    // great-circle calculations off the Python thread while preserving the
    // exact matrix layout consumed by the existing Sinkhorn implementation.
    cost_values
        .par_chunks_mut(m)
        .enumerate()
        .for_each(|(i, row)| {
            for j in 0..m {
                row[j] = pair_cost_m(
                    lats_a[i], lngs_a[i], ts_a[i], lats_b[j], lngs_b[j], ts_b[j], cyclical_period, r,
                );
            }
        });
    let cost =
        Array2::from_shape_vec((n, m), cost_values).expect("cost buffer has exactly n * m entries");

    let a: Array1<f32> = ws_a.iter().map(|&w| w as f32).collect();
    let b: Array1<f32> = ws_b.iter().map(|&w| w as f32).collect();

    let reg = sinkhorn.reg as f32;
    let distance = if let Some(tol) = sinkhorn.tol {
        let (_, distance, _iterations) =
            sinkhorn_log_with_convergence(&a, &b, &cost, reg, sinkhorn.max_iter, tol as f32)
                .map_err(|err| err.to_string())?;
        distance
    } else {
        let (_, distance) = sinkhorn_log(&a, &b, &cost, reg, sinkhorn.max_iter);
        distance
    };
    Ok(distance as f64)
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
            SinkhornConfig::default(),
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
            SinkhornConfig::default(),
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
            SinkhornConfig::default(),
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
            SinkhornConfig::default(),
        )
        .unwrap();
        assert!(near < far, "near={near} far={far}");
    }

    #[test]
    fn empty_distribution_is_rejected() {
        assert!(stvd_emd_impl(
            &[],
            &[],
            &[],
            &[],
            &[0.0],
            &[0.0],
            &[0.0],
            &[1.0],
            10.0,
            1440.0,
            SinkhornConfig::default(),
        )
        .is_err());
    }

    #[test]
    fn oversized_dense_input_is_rejected_with_clear_error() {
        // n*m*8 bytes (cost + wass's internal coupling-matrix doubling) must
        // exceed DENSE_COST_MEMORY_BUDGET_BYTES without actually allocating
        // that much: n and m individually stay small and cheap to build.
        let n = 24_000;
        let m = 24_000;
        assert!((n as u64) * (m as u64) * 8 > DENSE_COST_MEMORY_BUDGET_BYTES);
        let lats_a = vec![0.0_f64; n];
        let lngs_a = vec![0.0_f64; n];
        let ts_a = vec![0.0_f64; n];
        let ws_a = vec![1.0_f64; n];
        let lats_b = vec![0.0_f64; m];
        let lngs_b = vec![0.0_f64; m];
        let ts_b = vec![0.0_f64; m];
        let ws_b = vec![1.0_f64; m];
        let err = stvd_emd_impl(
            &lats_a,
            &lngs_a,
            &ts_a,
            &ws_a,
            &lats_b,
            &lngs_b,
            &ts_b,
            &ws_b,
            10.0,
            1440.0,
            SinkhornConfig::default(),
        )
        .unwrap_err();
        assert!(err.contains("too large"), "unexpected error message: {err}");
    }

    #[test]
    fn non_positive_cyclical_period_is_rejected() {
        assert!(stvd_emd_impl(
            &[0.0],
            &[0.0],
            &[0.0],
            &[1.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[1.0],
            10.0,
            0.0,
            SinkhornConfig::default(),
        )
        .is_err());
    }

    #[test]
    fn zero_weight_sum_is_rejected() {
        assert!(stvd_emd_impl(
            &[0.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[1.0],
            10.0,
            1440.0,
            SinkhornConfig::default(),
        )
        .is_err());
    }

    #[test]
    fn invalid_weights_and_coordinates_are_rejected() {
        assert!(stvd_emd_impl(
            &[f64::NAN],
            &[0.0],
            &[0.0],
            &[1.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[1.0],
            10.0,
            1440.0,
            SinkhornConfig::default(),
        )
        .is_err());
        assert!(stvd_emd_impl(
            &[0.0],
            &[0.0],
            &[0.0],
            &[-1.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[1.0],
            10.0,
            1440.0,
            SinkhornConfig::default(),
        )
        .is_err());
    }

    #[test]
    fn invalid_sinkhorn_reg_is_rejected() {
        assert!(stvd_emd_impl(
            &[0.0],
            &[0.0],
            &[0.0],
            &[1.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[1.0],
            10.0,
            1440.0,
            SinkhornConfig { reg: 0.0, ..SinkhornConfig::default() },
        )
        .is_err());
        assert!(stvd_emd_impl(
            &[0.0],
            &[0.0],
            &[0.0],
            &[1.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[1.0],
            10.0,
            1440.0,
            SinkhornConfig { max_iter: 0, ..SinkhornConfig::default() },
        )
        .is_err());
        assert!(stvd_emd_impl(
            &[0.0],
            &[0.0],
            &[0.0],
            &[1.0],
            &[0.0],
            &[0.0],
            &[0.0],
            &[1.0],
            10.0,
            1440.0,
            SinkhornConfig { tol: Some(-1.0), ..SinkhornConfig::default() },
        )
        .is_err());
    }

    #[test]
    fn custom_sinkhorn_config_with_tolerance_converges() {
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
            SinkhornConfig { reg: 0.1, max_iter: 500, tol: Some(1e-6) },
        )
        .unwrap();
        assert!(value.abs() < 1e-3, "got {value}");
    }
}
