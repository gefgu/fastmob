//! Point-in-time position queries.
//!
//! `fastmob.trajectory.interpolate_at` answers "where was this user at time
//! t?" for one or more query timestamps, independently per user. Unlike
//! [`super::interpolate`], this never changes the row count: output is
//! exactly `num_users * num_query_times` triples in user-major order, so no
//! per-user boundary arrays are needed.

use rayon::prelude::*;
use std::str::FromStr;

use crate::utils::validate_coord_ends;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PositionQueryMethod {
    Linear,
    Nearest,
}

impl FromStr for PositionQueryMethod {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "linear" => Ok(PositionQueryMethod::Linear),
            "nearest" => Ok(PositionQueryMethod::Nearest),
            other => Err(format!("unknown interpolate_at method: {other:?}")),
        }
    }
}

/// Queries one user's chronologically-sorted, already-null-filtered slice at
/// every timestamp in `query_times`. A query time outside
/// `[times[0], times[n-1]]` is marked invalid (`NaN`/`false`) rather than
/// raising, matching this codebase's validity-mask convention.
fn query_user_slice(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    query_times: &[f64],
    method: PositionQueryMethod,
) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
    let n = lats.len();
    let mut out_lats = Vec::with_capacity(query_times.len());
    let mut out_lngs = Vec::with_capacity(query_times.len());
    let mut out_valid = Vec::with_capacity(query_times.len());

    for &q in query_times {
        if n == 0 || q < times[0] || q > times[n - 1] {
            out_lats.push(f64::NAN);
            out_lngs.push(f64::NAN);
            out_valid.push(false);
            continue;
        }

        let idx = times.partition_point(|&t| t <= q);
        let (lat, lng) = if idx == 0 {
            (lats[0], lngs[0])
        } else if idx >= n {
            (lats[n - 1], lngs[n - 1])
        } else {
            let (t0, t1) = (times[idx - 1], times[idx]);
            match method {
                PositionQueryMethod::Nearest => {
                    if (q - t0).abs() <= (t1 - q).abs() {
                        (lats[idx - 1], lngs[idx - 1])
                    } else {
                        (lats[idx], lngs[idx])
                    }
                }
                PositionQueryMethod::Linear => {
                    if t1 > t0 {
                        let frac = (q - t0) / (t1 - t0);
                        (
                            lats[idx - 1] + (lats[idx] - lats[idx - 1]) * frac,
                            lngs[idx - 1] + (lngs[idx] - lngs[idx - 1]) * frac,
                        )
                    } else {
                        (lats[idx - 1], lngs[idx - 1])
                    }
                }
            }
        };
        out_lats.push(lat);
        out_lngs.push(lng);
        out_valid.push(true);
    }

    (out_lats, out_lngs, out_valid)
}

fn flatten_per_user(
    per_user: Vec<(Vec<f64>, Vec<f64>, Vec<bool>)>,
) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
    let total: usize = per_user.iter().map(|(l, _, _)| l.len()).sum();
    let mut out_lats = Vec::with_capacity(total);
    let mut out_lngs = Vec::with_capacity(total);
    let mut out_valid = Vec::with_capacity(total);
    for (lats, lngs, valid) in per_user {
        out_lats.extend(lats);
        out_lngs.extend(lngs);
        out_valid.extend(valid);
    }
    (out_lats, out_lngs, out_valid)
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

type PositionQueryResult = Result<(Vec<f64>, Vec<f64>, Vec<bool>), String>;

/// Batched, indexed entry point. Handles nulls natively per Rule 2.
///
/// @usedBy `fastmob-py/src/trajectory/interpolate_at_py.rs::interpolate_at_indexed`.
#[allow(clippy::too_many_arguments)]
pub fn interpolate_at_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps_s: &[f64],
    sorted_indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
    query_times_s: &[f64],
    method: PositionQueryMethod,
) -> PositionQueryResult {
    crate::utils::validate_indexed_coord_ends(latitudes, longitudes, sorted_indices, ends)?;
    if timestamps_s.len() != latitudes.len() {
        return Err(
            "latitudes, longitudes, and timestamps_s must have the same length".to_string(),
        );
    }

    let per_user: Vec<(Vec<f64>, Vec<f64>, Vec<bool>)> = (0..ends.len())
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
            query_user_slice(&lats, &lngs, &times, query_times_s, method)
        })
        .collect();

    Ok(flatten_per_user(per_user))
}

/// Batched, presorted-contiguous-ranges entry point. Assumes pre-cleaned
/// input (the `presorted=True` escape hatch); no null handling.
///
/// @usedBy `fastmob-py/src/trajectory/interpolate_at_py.rs::interpolate_at_presorted`.
pub fn interpolate_at_presorted_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps_s: &[f64],
    ends: &[usize],
    query_times_s: &[f64],
    method: PositionQueryMethod,
) -> PositionQueryResult {
    validate_coord_ends(latitudes, longitudes, ends)?;
    if timestamps_s.len() != latitudes.len() {
        return Err(
            "latitudes, longitudes, and timestamps_s must have the same length".to_string(),
        );
    }

    let per_user: Vec<(Vec<f64>, Vec<f64>, Vec<bool>)> = (0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            query_user_slice(
                &latitudes[start..end],
                &longitudes[start..end],
                &timestamps_s[start..end],
                query_times_s,
                method,
            )
        })
        .collect();

    Ok(flatten_per_user(per_user))
}
