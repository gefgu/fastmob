//! Kalman constant-velocity (CV) trajectory smoother.
//!
//! Ported from MovingPandas' `KalmanSmootherCV`, which itself builds a
//! `CombinedLinearGaussianTransitionModel([ConstantVelocity, ConstantVelocity])`
//! (Stone Soup) -- i.e. two *independent* 1D constant-velocity Kalman
//! filters, one per planar axis, each state `[position, velocity]`. Rather
//! than carrying a coupled 4x4 state (`[px,vx,py,vy]`), this kernel runs the
//! same 1D filter/smoother twice (once on `x_km`, once on `y_km`), matching
//! that decoupling exactly and keeping every matrix operation a closed-form
//! 2x2 (no `nalgebra`/`ndarray` dependency needed).
//!
//! Unlike `interpolate` (which inserts new points and changes row count),
//! `smooth` replaces every existing row's position at its original
//! timestamp: output cardinality always equals input cardinality. Invalid
//! rows (per Rule 2 -- excluded from a user's chronological slice) are left
//! as `NaN` in the output, since there is no meaningful smoothed position
//! for a row with no source coordinate.

use std::str::FromStr;

use rayon::prelude::*;

use crate::utils::haversine::{
    local_planar_km_params, project_local_planar_km_with_params, unproject_local_planar_km,
};
use crate::utils::validate_coord_ends;

/// The one shipped named smoothing algorithm (room to grow, matching
/// `InterpolationMethod`'s enum-of-named-methods shape).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothMethod {
    KalmanCv,
}

impl FromStr for SmoothMethod {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "kalman_cv" => Ok(SmoothMethod::KalmanCv),
            other => Err(format!("unknown smooth method: {other:?}")),
        }
    }
}

/// Parameters for the Kalman CV smoother. Isotropic (one noise pair applied
/// to both axes) -- simpler than MovingPandas' scalar-or-(x,y)-pair option;
/// anisotropic support is a natural, low-risk future extension.
#[derive(Clone, Copy)]
pub struct SmoothConfig {
    pub method: SmoothMethod,
    /// Acceleration-noise standard deviation (km), i.e. how strongly the
    /// constant-velocity assumption is enforced. Higher -> less smoothing.
    pub process_noise_std_km: f64,
    /// Assumed GPS measurement noise standard deviation (km). Higher ->
    /// smoother, more-corrected output.
    pub measurement_noise_std_km: f64,
}

impl SmoothConfig {
    pub fn new(
        method: SmoothMethod,
        process_noise_std_km: f64,
        measurement_noise_std_km: f64,
    ) -> Self {
        SmoothConfig {
            method,
            process_noise_std_km,
            measurement_noise_std_km,
        }
    }
}

// ---------------------------------------------------------------------------
// Closed-form 2x2 matrix / 2-vector helpers for a 1D [position, velocity]
// constant-velocity Kalman filter + RTS smoother.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Vec2(f64, f64);

/// Symmetric-by-construction 2x2 matrix `[[a, b], [c, d]]` (covariances kept
/// symmetric by the update equations below, so `b` and `c` coincide in
/// practice, but both are tracked to avoid asserting it).
#[derive(Clone, Copy)]
struct Mat2 {
    a: f64,
    b: f64,
    c: f64,
    d: f64,
}

impl Mat2 {
    fn identity() -> Self {
        Mat2 {
            a: 1.0,
            b: 0.0,
            c: 0.0,
            d: 1.0,
        }
    }

    fn transpose(self) -> Self {
        Mat2 {
            a: self.a,
            b: self.c,
            c: self.b,
            d: self.d,
        }
    }

    fn add(self, o: Mat2) -> Mat2 {
        Mat2 {
            a: self.a + o.a,
            b: self.b + o.b,
            c: self.c + o.c,
            d: self.d + o.d,
        }
    }

    fn sub(self, o: Mat2) -> Mat2 {
        Mat2 {
            a: self.a - o.a,
            b: self.b - o.b,
            c: self.c - o.c,
            d: self.d - o.d,
        }
    }

    fn mul(self, o: Mat2) -> Mat2 {
        Mat2 {
            a: self.a * o.a + self.b * o.c,
            b: self.a * o.b + self.b * o.d,
            c: self.c * o.a + self.d * o.c,
            d: self.c * o.b + self.d * o.d,
        }
    }

    fn mul_vec(self, v: Vec2) -> Vec2 {
        Vec2(self.a * v.0 + self.b * v.1, self.c * v.0 + self.d * v.1)
    }

    fn inverse(self) -> Mat2 {
        let det = self.a * self.d - self.b * self.c;
        let inv_det = if det.abs() < 1e-15 { 0.0 } else { 1.0 / det };
        Mat2 {
            a: self.d * inv_det,
            b: -self.b * inv_det,
            c: -self.c * inv_det,
            d: self.a * inv_det,
        }
    }
}

fn transition(dt: f64) -> Mat2 {
    Mat2 {
        a: 1.0,
        b: dt,
        c: 0.0,
        d: 1.0,
    }
}

/// Discretized white-noise-acceleration process covariance for a 1D
/// constant-velocity model over a `dt`-second step, with `q` the
/// acceleration-noise power spectral density (`process_noise_std_km^2`).
fn process_noise(dt: f64, q: f64) -> Mat2 {
    let dt2 = dt * dt;
    let dt3 = dt2 * dt;
    Mat2 {
        a: q * dt3 / 3.0,
        b: q * dt2 / 2.0,
        c: q * dt2 / 2.0,
        d: q * dt,
    }
}

struct FilterStep {
    x_pred: Vec2,
    p_pred: Mat2,
    x_filt: Vec2,
    p_filt: Mat2,
    f: Mat2,
}

/// Forward Kalman filter + backward RTS smoother over one axis'
/// chronologically-ordered `(position_km, time_s)` samples. Requires
/// `positions.len() >= 2` (callers handle shorter slices as a no-op).
fn kalman_smooth_1d(positions: &[f64], times_s: &[f64], q: f64, r: f64) -> Vec<f64> {
    let n = positions.len();
    debug_assert_eq!(n, times_s.len());

    let mut steps: Vec<FilterStep> = Vec::with_capacity(n);

    // Initialize at the first observation with zero velocity and a wide
    // velocity prior (no information about initial speed yet).
    let x0 = Vec2(positions[0], 0.0);
    let p0 = Mat2 {
        a: r,
        b: 0.0,
        c: 0.0,
        d: 1.0e6,
    };
    steps.push(FilterStep {
        x_pred: x0,
        p_pred: p0,
        x_filt: x0,
        p_filt: p0,
        f: Mat2::identity(),
    });

    let mut x = x0;
    let mut p = p0;

    for i in 1..n {
        let dt = (times_s[i] - times_s[i - 1]).max(1e-6);
        let f = transition(dt);
        let qmat = process_noise(dt, q);

        let x_pred = f.mul_vec(x);
        let p_pred = f.mul(p).mul(f.transpose()).add(qmat);

        // Update step (H = [1, 0], so the observation is the position
        // component only; innovation/covariance stay scalars).
        let innovation = positions[i] - x_pred.0;
        let s = p_pred.a + r;
        let k = Vec2(p_pred.a / s, p_pred.c / s);
        let x_filt = Vec2(x_pred.0 + k.0 * innovation, x_pred.1 + k.1 * innovation);
        // P_filt = (I - K H) P_pred, with K H = [[k.0, 0], [k.1, 0]].
        let p_filt = Mat2 {
            a: (1.0 - k.0) * p_pred.a,
            b: (1.0 - k.0) * p_pred.b,
            c: p_pred.c - k.1 * p_pred.a,
            d: p_pred.d - k.1 * p_pred.b,
        };

        steps.push(FilterStep {
            x_pred,
            p_pred,
            x_filt,
            p_filt,
            f,
        });
        x = x_filt;
        p = p_filt;
    }

    // Backward Rauch-Tung-Striebel smoother pass.
    let mut smoothed_x = vec![Vec2(0.0, 0.0); n];
    let mut smoothed_p = vec![steps[n - 1].p_filt; n];
    smoothed_x[n - 1] = steps[n - 1].x_filt;

    for i in (0..n - 1).rev() {
        let next = &steps[i + 1];
        let gain = steps[i]
            .p_filt
            .mul(next.f.transpose())
            .mul(next.p_pred.inverse());
        let x_diff = Vec2(
            smoothed_x[i + 1].0 - next.x_pred.0,
            smoothed_x[i + 1].1 - next.x_pred.1,
        );
        smoothed_x[i] = Vec2(
            steps[i].x_filt.0 + gain.mul_vec(x_diff).0,
            steps[i].x_filt.1 + gain.mul_vec(x_diff).1,
        );
        let p_diff = smoothed_p[i + 1].sub(next.p_pred);
        smoothed_p[i] = steps[i].p_filt.add(gain.mul(p_diff).mul(gain.transpose()));
    }

    smoothed_x.into_iter().map(|v| v.0).collect()
}

/// Runs the configured smoothing algorithm over one user's
/// chronologically-sorted, already-null-filtered slice. Returns the
/// smoothed `(lats, lngs)`, same length and order as the input slice.
fn kalman_smooth_user_slice(
    lats: &[f64],
    lngs: &[f64],
    times_s: &[f64],
    config: &SmoothConfig,
) -> (Vec<f64>, Vec<f64>) {
    let n = lats.len();
    if n < 2 {
        return (lats.to_vec(), lngs.to_vec());
    }

    let (mean_lat_rad, cos_mean_lat) = local_planar_km_params(lats);
    let projected = project_local_planar_km_with_params(lats, lngs, mean_lat_rad, cos_mean_lat);
    let xs: Vec<f64> = projected.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = projected.iter().map(|p| p.1).collect();

    let q = config.process_noise_std_km * config.process_noise_std_km;
    let r = config.measurement_noise_std_km * config.measurement_noise_std_km;

    let smoothed_x = kalman_smooth_1d(&xs, times_s, q, r);
    let smoothed_y = kalman_smooth_1d(&ys, times_s, q, r);

    let points: Vec<(f64, f64)> = smoothed_x.into_iter().zip(smoothed_y).collect();
    unproject_local_planar_km(&points, mean_lat_rad, cos_mean_lat)
}

type SmoothResult = Result<(Vec<f64>, Vec<f64>), String>;

fn is_valid_indexed_row(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    valid_rows: Option<&[bool]>,
    idx: usize,
) -> bool {
    valid_rows.is_none_or(|v| v[idx])
        && lats[idx].is_finite()
        && lngs[idx].is_finite()
        && times[idx].is_finite()
}

fn smooth_slice(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    config: &SmoothConfig,
) -> (Vec<f64>, Vec<f64>) {
    match config.method {
        SmoothMethod::KalmanCv => kalman_smooth_user_slice(lats, lngs, times, config),
    }
}

/// Batched, indexed entry point (one call covers every user, rows addressed
/// through `sorted_indices`/`ends`). Handles nulls natively per Rule 2:
/// invalid rows are excluded from the per-user chronological slice before
/// smoothing, and left as `NaN` in the output (row count never changes).
///
/// @usedBy `fastmob-py/src/trajectory/smooth_py.rs::smooth_trajectory_indexed`.
pub fn smooth_trajectory_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps_s: &[f64],
    sorted_indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
    config: &SmoothConfig,
) -> SmoothResult {
    crate::utils::validate_indexed_coord_ends(latitudes, longitudes, sorted_indices, ends)?;
    if timestamps_s.len() != latitudes.len() {
        return Err(
            "latitudes, longitudes, and timestamps_s must have the same length".to_string(),
        );
    }

    let n = latitudes.len();
    let per_user: Vec<(Vec<usize>, Vec<f64>, Vec<f64>)> = (0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let mut idxs = Vec::with_capacity(end - start);
            let mut lats = Vec::with_capacity(end - start);
            let mut lngs = Vec::with_capacity(end - start);
            let mut times = Vec::with_capacity(end - start);
            for &idx_u32 in &sorted_indices[start..end] {
                let idx = idx_u32;
                if is_valid_indexed_row(latitudes, longitudes, timestamps_s, valid_rows, idx) {
                    idxs.push(idx_u32);
                    lats.push(latitudes[idx]);
                    lngs.push(longitudes[idx]);
                    times.push(timestamps_s[idx]);
                }
            }
            let (smoothed_lats, smoothed_lngs) = smooth_slice(&lats, &lngs, &times, config);
            (idxs, smoothed_lats, smoothed_lngs)
        })
        .collect();

    let mut out_lats = vec![f64::NAN; n];
    let mut out_lngs = vec![f64::NAN; n];
    for (idxs, lats, lngs) in per_user {
        for (j, &idx) in idxs.iter().enumerate() {
            out_lats[idx] = lats[j];
            out_lngs[idx] = lngs[j];
        }
    }
    Ok((out_lats, out_lngs))
}

/// Batched, presorted-contiguous-ranges entry point. Assumes pre-cleaned
/// input (the `presorted=True` escape hatch); no null handling. Each range
/// is already the caller's original row order, so output arrays are simply
/// the per-user results concatenated in range order -- no scatter needed.
///
/// @usedBy `fastmob-py/src/trajectory/smooth_py.rs::smooth_trajectory_presorted`.
pub fn smooth_trajectory_presorted_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps_s: &[f64],
    ends: &[usize],
    config: &SmoothConfig,
) -> SmoothResult {
    validate_coord_ends(latitudes, longitudes, ends)?;
    if timestamps_s.len() != latitudes.len() {
        return Err(
            "latitudes, longitudes, and timestamps_s must have the same length".to_string(),
        );
    }

    let per_user: Vec<(Vec<f64>, Vec<f64>)> = (0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            smooth_slice(
                &latitudes[start..end],
                &longitudes[start..end],
                &timestamps_s[start..end],
                config,
            )
        })
        .collect();

    let mut out_lats = Vec::with_capacity(latitudes.len());
    let mut out_lngs = Vec::with_capacity(latitudes.len());
    for (lats, lngs) in per_user {
        out_lats.extend(lats);
        out_lngs.extend(lngs);
    }
    Ok((out_lats, out_lngs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_velocity_line_is_recovered_with_low_noise() {
        // A straight line traveled at constant speed with small measurement
        // noise should be smoothed back close to the true line.
        let n = 20;
        let times: Vec<f64> = (0..n).map(|i| i as f64 * 60.0).collect();
        let true_lats: Vec<f64> = (0..n).map(|i| 40.0 + i as f64 * 0.0005).collect();
        let true_lngs: Vec<f64> = vec![10.0; n];

        // Deterministic pseudo-noise via a simple LCG, no extra dependency.
        let mut seed: u64 = 42;
        let mut noisy_lats = Vec::with_capacity(n);
        for &lat in &true_lats {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = ((seed >> 33) as f64 / u32::MAX as f64 - 0.5) * 0.0002;
            noisy_lats.push(lat + noise);
        }

        let config = SmoothConfig::new(SmoothMethod::KalmanCv, 0.01, 0.05);
        let (smoothed_lats, smoothed_lngs) =
            kalman_smooth_user_slice(&noisy_lats, &true_lngs, &times, &config);

        assert_eq!(smoothed_lats.len(), n);
        assert_eq!(smoothed_lngs.len(), n);

        let raw_err: f64 = noisy_lats
            .iter()
            .zip(&true_lats)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / n as f64;
        let smoothed_err: f64 = smoothed_lats
            .iter()
            .zip(&true_lats)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / n as f64;
        assert!(
            smoothed_err < raw_err,
            "expected smoothing to reduce error: raw={raw_err}, smoothed={smoothed_err}"
        );
    }

    #[test]
    fn short_slice_is_returned_unchanged() {
        let config = SmoothConfig::new(SmoothMethod::KalmanCv, 0.01, 0.05);
        let (lats, lngs) = kalman_smooth_user_slice(&[1.0], &[2.0], &[0.0], &config);
        assert_eq!(lats, vec![1.0]);
        assert_eq!(lngs, vec![2.0]);

        let (lats, lngs) = kalman_smooth_user_slice(&[], &[], &[], &config);
        assert!(lats.is_empty());
        assert!(lngs.is_empty());
    }
}
