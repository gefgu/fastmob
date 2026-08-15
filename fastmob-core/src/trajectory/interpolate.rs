//! Named trajectory gap-filling algorithms.
//!
//! `fastmob.trajectory.interpolate` exposes one `method=` string that
//! resolves, on the Python side, to an [`InterpolationMethod`] variant
//! carried by [`InterpolationConfig`] -- mirrors
//! `fastmob-core::preprocessing::simplify`'s single-config-struct shape.
//!
//! Insertion policy (shared by all four methods, matching PTRAIL's
//! `Interpolation.interpolate_position`): for every gap between two
//! chronologically consecutive points whose time delta exceeds
//! `sampling_rate_s`, exactly **one** new point is inserted at
//! `t[i-1] + sampling_rate_s` -- the gap is not filled iteratively down to
//! `sampling_rate_s`-sized steps. Original points are always re-emitted
//! verbatim, so the output is a flat, expanded, still-chronological
//! trajectory (variable row count per user), not a per-user list column.

use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::str::FromStr;

use geo::{Destination, Haversine, Point};

use crate::utils::haversine::haversine_km;
use crate::utils::validate_coord_ends;

/// The four shipped named interpolation algorithms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolationMethod {
    Linear,
    CubicSpline,
    Kinematic,
    RandomWalk,
}

impl FromStr for InterpolationMethod {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "linear" => Ok(InterpolationMethod::Linear),
            "cubic_spline" => Ok(InterpolationMethod::CubicSpline),
            "kinematic" => Ok(InterpolationMethod::Kinematic),
            "random_walk" => Ok(InterpolationMethod::RandomWalk),
            other => Err(format!("unknown interpolate method: {other:?}")),
        }
    }
}

/// Parameters for every shipped interpolation method. Only the field(s)
/// relevant to `method` are used per call, mirroring `SimplifyConfig`'s flat
/// carry-every-parameter shape.
#[derive(Clone, Copy)]
pub struct InterpolationConfig {
    pub method: InterpolationMethod,
    /// Maximum time gap (seconds) allowed between consecutive points before a
    /// single interpolated point is inserted at `t[i-1] + sampling_rate_s`.
    pub sampling_rate_s: f64,
    /// Kinematic-only safety clamp: if the fitted kinematic position implies
    /// a speed from the preceding point above this bound, fall back to
    /// linear interpolation for that gap instead (guards against noisy
    /// velocity estimates producing wild extrapolation artifacts).
    pub max_speed_kmh: f64,
    /// Random-walk-only perturbation magnitude: the half-normal standard
    /// deviation (km) of the random displacement applied on top of the
    /// linear-interpolated base position.
    pub step_std_km: f64,
    /// Random-walk-only RNG seed. Combined with a per-user offset so
    /// different users in the same call draw independent sequences.
    pub seed: u64,
}

impl InterpolationConfig {
    pub fn new(
        method: InterpolationMethod,
        sampling_rate_s: f64,
        max_speed_kmh: f64,
        step_std_km: f64,
        seed: u64,
    ) -> Self {
        InterpolationConfig {
            method,
            sampling_rate_s,
            max_speed_kmh,
            step_std_km,
            seed,
        }
    }
}

type InterpolationResult = Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>), String>;

// ---------------------------------------------------------------------------
// Natural cubic spline (second-derivative-zero boundary conditions).
// ---------------------------------------------------------------------------

/// Keep only strictly-increasing `t` nodes (first occurrence wins), needed
/// because a natural cubic spline fit requires distinct x-nodes.
fn dedupe_by_time(times: &[f64], lats: &[f64], lngs: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut ts = Vec::with_capacity(times.len());
    let mut la = Vec::with_capacity(times.len());
    let mut lo = Vec::with_capacity(times.len());
    for i in 0..times.len() {
        if i == 0 || times[i] > *ts.last().unwrap() {
            ts.push(times[i]);
            la.push(lats[i]);
            lo.push(lngs[i]);
        }
    }
    (ts, la, lo)
}

/// Second derivatives at each node of a natural cubic spline through
/// strictly-increasing `t`, solved via the standard tridiagonal (Thomas
/// algorithm) system.
fn natural_cubic_spline_second_derivatives(t: &[f64], y: &[f64]) -> Vec<f64> {
    let n = t.len();
    let mut m = vec![0.0; n];
    if n < 3 {
        return m;
    }

    let mut h = vec![0.0; n - 1];
    for i in 0..n - 1 {
        h[i] = t[i + 1] - t[i];
    }

    let mut alpha = vec![0.0; n];
    for i in 1..n - 1 {
        alpha[i] = 3.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1]);
    }

    let mut l = vec![1.0; n];
    let mut mu = vec![0.0; n];
    let mut z = vec![0.0; n];
    for i in 1..n - 1 {
        l[i] = 2.0 * (t[i + 1] - t[i - 1]) - h[i - 1] * mu[i - 1];
        mu[i] = h[i] / l[i];
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
    }

    for j in (0..n - 1).rev() {
        m[j] = z[j] - mu[j] * m[j + 1];
    }
    m
}

fn evaluate_cubic_spline(t: &[f64], y: &[f64], m: &[f64], seg: usize, query_t: f64) -> f64 {
    let h = t[seg + 1] - t[seg];
    let a = (t[seg + 1] - query_t) / h;
    let b = (query_t - t[seg]) / h;
    a * y[seg]
        + b * y[seg + 1]
        + ((a.powi(3) - a) * m[seg] + (b.powi(3) - b) * m[seg + 1]) * (h * h) / 6.0
}

// ---------------------------------------------------------------------------
// Kinematic (velocity-continuous Hermite cubic) interpolation.
// ---------------------------------------------------------------------------

/// Solves `b, c` for `p(t) = x0 + v0*t + b*t^2/2 + c*t^3/6` such that
/// `p(dt) = x1` and `p'(dt) = (x1 - x0) / dt` (the gap's own secant slope,
/// used as the exit-velocity boundary condition -- matches PTRAIL's
/// `kinematic_help` coefficient derivation).
fn kinematic_solve(x0: f64, v0: f64, x1: f64, dt: f64) -> (f64, f64) {
    let v1 = (x1 - x0) / dt;
    let a11 = dt * dt / 2.0;
    let a12 = dt * dt * dt / 6.0;
    let a21 = dt;
    let a22 = dt * dt / 2.0;
    let b1 = x1 - x0 - v0 * dt;
    let b2 = v1 - v0;
    let det = a11 * a22 - a12 * a21;
    if det.abs() < 1e-12 {
        return (0.0, 0.0);
    }
    let b = (b1 * a22 - a12 * b2) / det;
    let c = (a11 * b2 - b1 * a21) / det;
    (b, c)
}

fn kinematic_eval(x0: f64, v0: f64, b: f64, c: f64, t: f64) -> f64 {
    x0 + v0 * t + b * t * t / 2.0 + c * t * t * t / 6.0
}

// ---------------------------------------------------------------------------
// Random-walk perturbation.
// ---------------------------------------------------------------------------

/// Standard-normal sample via the Box-Muller transform (avoids adding
/// `rand_distr` as a new dependency for a single distribution).
fn standard_normal_sample(rng: &mut Xoshiro256PlusPlus) -> f64 {
    let u1: f64 = rng.gen_range(0.0_f64..1.0).max(1e-12);
    let u2: f64 = rng.gen_range(0.0_f64..1.0);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// ---------------------------------------------------------------------------
// Per-user gap-fill.
// ---------------------------------------------------------------------------

type CubicSplineFit = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

fn fit_cubic_spline(times: &[f64], lats: &[f64], lngs: &[f64]) -> Option<CubicSplineFit> {
    let (dt_times, dt_lats, dt_lngs) = dedupe_by_time(times, lats, lngs);
    if dt_times.len() <= 3 {
        return None;
    }
    let lat_m = natural_cubic_spline_second_derivatives(&dt_times, &dt_lats);
    let lng_m = natural_cubic_spline_second_derivatives(&dt_times, &dt_lngs);
    Some((dt_times, lat_m, dt_lats, lng_m, dt_lngs))
}

fn cubic_spline_point(spline: &CubicSplineFit, t0: f64, t_new: f64) -> Option<(f64, f64)> {
    let (sp_times, lat_m, sp_lats, lng_m, sp_lngs) = spline;
    match sp_times.binary_search_by(|probe| probe.partial_cmp(&t0).unwrap()) {
        Ok(seg) if seg + 1 < sp_times.len() => Some((
            evaluate_cubic_spline(sp_times, sp_lats, lat_m, seg, t_new),
            evaluate_cubic_spline(sp_times, sp_lngs, lng_m, seg, t_new),
        )),
        _ => None,
    }
}

/// Runs the configured algorithm over one user's chronologically-sorted,
/// already-null-filtered slice. Returns the expanded `(lats, lngs, times)`
/// triple (original points re-emitted verbatim, plus at most one inserted
/// point per exceeded gap).
#[allow(clippy::too_many_arguments)]
fn interpolate_user_slice(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    config: &InterpolationConfig,
    rng: &mut Xoshiro256PlusPlus,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = lats.len();
    if n == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }
    if n == 1 {
        return (vec![lats[0]], vec![lngs[0]], vec![times[0]]);
    }

    let spline = if config.method == InterpolationMethod::CubicSpline {
        fit_cubic_spline(times, lats, lngs)
    } else {
        None
    };

    let mut out_lats = Vec::with_capacity(n);
    let mut out_lngs = Vec::with_capacity(n);
    let mut out_times = Vec::with_capacity(n);
    out_lats.push(lats[0]);
    out_lngs.push(lngs[0]);
    out_times.push(times[0]);

    for i in 1..n {
        let (lat0, lng0, t0) = (lats[i - 1], lngs[i - 1], times[i - 1]);
        let (lat1, lng1, t1) = (lats[i], lngs[i], times[i]);
        let dt = t1 - t0;

        if dt > config.sampling_rate_s && dt > 0.0 {
            let t_new = t0 + config.sampling_rate_s;
            let frac = config.sampling_rate_s / dt;
            let linear_point = (lat0 + (lat1 - lat0) * frac, lng0 + (lng1 - lng0) * frac);

            let point = match config.method {
                InterpolationMethod::Linear => linear_point,
                InterpolationMethod::CubicSpline => spline
                    .as_ref()
                    .and_then(|s| cubic_spline_point(s, t0, t_new))
                    .unwrap_or(linear_point),
                InterpolationMethod::Kinematic => {
                    if i >= 2 && times[i - 1] > times[i - 2] {
                        let prev_dt = times[i - 1] - times[i - 2];
                        let v0_lat = (lats[i - 1] - lats[i - 2]) / prev_dt;
                        let v0_lng = (lngs[i - 1] - lngs[i - 2]) / prev_dt;
                        let (b_lat, c_lat) = kinematic_solve(lat0, v0_lat, lat1, dt);
                        let (b_lng, c_lng) = kinematic_solve(lng0, v0_lng, lng1, dt);
                        let candidate = (
                            kinematic_eval(lat0, v0_lat, b_lat, c_lat, config.sampling_rate_s),
                            kinematic_eval(lng0, v0_lng, b_lng, c_lng, config.sampling_rate_s),
                        );
                        let implied_speed_kmh = haversine_km(lat0, lng0, candidate.0, candidate.1)
                            / (config.sampling_rate_s / 3600.0);
                        if implied_speed_kmh.is_finite()
                            && implied_speed_kmh <= config.max_speed_kmh
                        {
                            candidate
                        } else {
                            linear_point
                        }
                    } else {
                        linear_point
                    }
                }
                InterpolationMethod::RandomWalk => {
                    if config.step_std_km <= 0.0 {
                        linear_point
                    } else {
                        let magnitude_km = config.step_std_km * standard_normal_sample(rng).abs();
                        let bearing_deg = rng.gen_range(0.0_f64..360.0);
                        let origin = Point::new(linear_point.1, linear_point.0);
                        let destination =
                            Haversine.destination(origin, bearing_deg, magnitude_km * 1000.0);
                        (destination.y(), destination.x())
                    }
                }
            };

            out_lats.push(point.0);
            out_lngs.push(point.1);
            out_times.push(t_new);
        }

        out_lats.push(lat1);
        out_lngs.push(lng1);
        out_times.push(t1);
    }

    (out_lats, out_lngs, out_times)
}

fn flatten_per_user(
    per_user: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>) {
    let total: usize = per_user.iter().map(|(l, _, _)| l.len()).sum();
    let mut out_lats = Vec::with_capacity(total);
    let mut out_lngs = Vec::with_capacity(total);
    let mut out_times = Vec::with_capacity(total);
    let mut out_user_indices = Vec::with_capacity(total);
    for (user_idx, (lats, lngs, times)) in per_user.into_iter().enumerate() {
        let len = lats.len();
        out_lats.extend(lats);
        out_lngs.extend(lngs);
        out_times.extend(times);
        out_user_indices.extend(std::iter::repeat_n(user_idx, len));
    }
    (out_lats, out_lngs, out_times, out_user_indices)
}

fn is_valid_row(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps_s: &[f64],
    valid_rows: Option<&[bool]>,
    idx: usize,
) -> bool {
    valid_rows.is_none_or(|v| v[idx])
        && latitudes[idx].is_finite()
        && longitudes[idx].is_finite()
        && timestamps_s[idx].is_finite()
}

/// Batched, indexed entry point (one call covers every user, rows addressed
/// through `sorted_indices`/`ends`). Handles nulls natively: invalid rows are
/// excluded before gap detection, per Rule 2.
///
/// @usedBy `fastmob-py/src/trajectory/interpolate_py.rs::interpolate_trajectory_indexed`.
pub fn interpolate_trajectory_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps_s: &[f64],
    sorted_indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
    config: &InterpolationConfig,
) -> InterpolationResult {
    crate::utils::validate_indexed_coord_ends(latitudes, longitudes, sorted_indices, ends)?;
    if timestamps_s.len() != latitudes.len() {
        return Err(
            "latitudes, longitudes, and timestamps_s must have the same length".to_string(),
        );
    }

    let per_user: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let mut lats = Vec::with_capacity(end - start);
            let mut lngs = Vec::with_capacity(end - start);
            let mut times = Vec::with_capacity(end - start);
            for &idx_u32 in &sorted_indices[start..end] {
                let idx = idx_u32;
                if is_valid_row(latitudes, longitudes, timestamps_s, valid_rows, idx) {
                    lats.push(latitudes[idx]);
                    lngs.push(longitudes[idx]);
                    times.push(timestamps_s[idx]);
                }
            }
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(config.seed.wrapping_add(i as u64));
            interpolate_user_slice(&lats, &lngs, &times, config, &mut rng)
        })
        .collect();

    Ok(flatten_per_user(per_user))
}

/// Batched, presorted-contiguous-ranges entry point. Assumes pre-cleaned
/// input (the `presorted=True` escape hatch); no null handling.
///
/// @usedBy `fastmob-py/src/trajectory/interpolate_py.rs::interpolate_trajectory_presorted`.
pub fn interpolate_trajectory_presorted_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps_s: &[f64],
    ends: &[usize],
    config: &InterpolationConfig,
) -> InterpolationResult {
    validate_coord_ends(latitudes, longitudes, ends)?;
    if timestamps_s.len() != latitudes.len() {
        return Err(
            "latitudes, longitudes, and timestamps_s must have the same length".to_string(),
        );
    }

    let per_user: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(config.seed.wrapping_add(i as u64));
            interpolate_user_slice(
                &latitudes[start..end],
                &longitudes[start..end],
                &timestamps_s[start..end],
                config,
                &mut rng,
            )
        })
        .collect();

    Ok(flatten_per_user(per_user))
}
