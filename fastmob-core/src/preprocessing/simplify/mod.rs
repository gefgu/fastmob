//! Named trajectory simplification algorithms.
//!
//! `fastmob.preprocessing.simplify` exposes one `method=` string that
//! resolves, on the Python side, to a [`SimplifyMethod`] variant carried by
//! [`SimplifyConfig`]. Each per-user algorithm returns the 0-based local
//! indices it keeps; [`simplify_trajectory_impl`] /
//! [`simplify_trajectory_indexed_impl`] convert those per-user index lists
//! into a single boolean keep-mask aligned with the input rows, matching
//! `filter_traj`'s keep-mask output convention.
//!
//! Agarwal simplification is intentionally out of scope (see the project
//! plan): it needs a parametric feasibility-region search materially more
//! complex than the Wedge/corridor primitive built here for Chan-Chin and
//! Imai-Iri, with low value-add over Chan-Chin's tighter approximation
//! bound alone.

pub mod chan_chin;
pub mod corridor;
pub mod distance_time_threshold;
pub mod douglas_peucker;
pub mod imai_iri;
pub mod top_down_time_ratio;

use std::str::FromStr;

use rayon::prelude::*;

/// The seven shipped named simplification algorithms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimplifyMethod {
    DouglasPeucker,
    TopDownTimeRatio,
    MinDistance,
    MinTimeDelta,
    MaxDistance,
    ChanChin,
    ImaiIri,
}

impl FromStr for SimplifyMethod {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "douglas_peucker" => Ok(SimplifyMethod::DouglasPeucker),
            "top_down_time_ratio" => Ok(SimplifyMethod::TopDownTimeRatio),
            "min_distance" => Ok(SimplifyMethod::MinDistance),
            "min_time_delta" => Ok(SimplifyMethod::MinTimeDelta),
            "max_distance" => Ok(SimplifyMethod::MaxDistance),
            "chan_chin" => Ok(SimplifyMethod::ChanChin),
            "imai_iri" => Ok(SimplifyMethod::ImaiIri),
            other => Err(format!("unknown simplify method: {other:?}")),
        }
    }
}

/// Parameters for every shipped simplification method. Only the field(s)
/// relevant to `method` are used per call, mirroring `FilterConfig`'s flat
/// carry-every-parameter shape.
#[derive(Clone, Copy)]
pub struct SimplifyConfig {
    pub method: SimplifyMethod,
    /// Perpendicular/spatiotemporal distance tolerance in km, used by
    /// `DouglasPeucker`, `TopDownTimeRatio`, `MaxDistance`, `ChanChin`, and
    /// `ImaiIri`.
    pub epsilon_km: f64,
    /// Minimum distance between consecutive kept points in km, used by
    /// `MinDistance`.
    pub min_distance_km: f64,
    /// Minimum time delta between consecutive kept points in seconds, used
    /// by `MinTimeDelta`.
    pub min_time_delta_s: f64,
}

impl SimplifyConfig {
    pub fn new(
        method: SimplifyMethod,
        epsilon_km: f64,
        min_distance_km: f64,
        min_time_delta_s: f64,
    ) -> Self {
        SimplifyConfig {
            method,
            epsilon_km,
            min_distance_km,
            min_time_delta_s,
        }
    }
}

/// Run the configured algorithm over a single user's slice and return a
/// keep-mask the same length as the slice.
///
/// Trajectories with 2 or fewer points always keep every point: every
/// shipped algorithm keeps both endpoints unconditionally, so there is
/// nothing to simplify.
fn simplify_user_slice(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    config: &SimplifyConfig,
) -> Vec<bool> {
    let n = lats.len();
    if n <= 2 {
        return vec![true; n];
    }

    let kept_indices: Vec<usize> = match config.method {
        SimplifyMethod::DouglasPeucker => {
            douglas_peucker::douglas_peucker_indices(lats, lngs, config.epsilon_km)
        }
        SimplifyMethod::TopDownTimeRatio => {
            top_down_time_ratio::top_down_time_ratio_indices(lats, lngs, times, config.epsilon_km)
        }
        SimplifyMethod::MinDistance => {
            distance_time_threshold::min_distance_indices(lats, lngs, config.min_distance_km)
        }
        SimplifyMethod::MinTimeDelta => {
            distance_time_threshold::min_time_delta_indices(times, config.min_time_delta_s)
        }
        SimplifyMethod::MaxDistance => {
            distance_time_threshold::max_distance_indices(lats, lngs, config.epsilon_km)
        }
        SimplifyMethod::ChanChin => chan_chin::chan_chin_indices(lats, lngs, config.epsilon_km),
        SimplifyMethod::ImaiIri => imai_iri::imai_iri_indices(lats, lngs, config.epsilon_km),
    };

    let mut keep = vec![false; n];
    for idx in kept_indices {
        keep[idx] = true;
    }
    keep
}

fn is_valid_simplify_row(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    valid_rows: Option<&[bool]>,
    idx: usize,
) -> bool {
    valid_rows.is_none_or(|valid| valid[idx])
        && lats[idx].is_finite()
        && lngs[idx].is_finite()
        && times[idx].is_finite()
}

/// Batched, presorted-contiguous-ranges entry point (one call covers every
/// user). Returns a boolean keep-mask the same length as `latitudes`.
///
/// @usedBy `fastmob-py/src/preprocessing/simplify_traj_py.rs::simplify_trajectory_{numpy,arrow}`.
pub fn simplify_trajectory_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps_s: &[f64],
    ranges: &[(usize, usize)],
    config: &SimplifyConfig,
) -> Vec<bool> {
    let n = latitudes.len();
    let mut mask = vec![true; n];

    let mask_addr = mask.as_mut_ptr() as usize;
    ranges.par_iter().for_each(|&(start, end)| {
        let user_len = end - start;
        let user_mask = unsafe {
            std::slice::from_raw_parts_mut((mask_addr as *mut bool).add(start), user_len)
        };
        user_mask.copy_from_slice(&simplify_user_slice(
            &latitudes[start..end],
            &longitudes[start..end],
            &timestamps_s[start..end],
            config,
        ));
    });

    mask
}

/// Batched, indexed entry point (one call covers every user, rows addressed
/// through `sorted_indices`/`ends` rather than contiguous ranges). Handles
/// nulls natively: rows failing [`is_valid_simplify_row`] are marked `false`
/// in the output mask and excluded from the per-user algorithm input.
///
/// @usedBy `fastmob-py/src/preprocessing/simplify_traj_py.rs::simplify_trajectory_indexed_{numpy,arrow}`.
pub fn simplify_trajectory_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps_s: &[f64],
    sorted_indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
    config: &SimplifyConfig,
) -> Vec<bool> {
    let n = latitudes.len();
    debug_assert_eq!(longitudes.len(), n);
    debug_assert_eq!(timestamps_s.len(), n);
    debug_assert!(valid_rows.is_none_or(|valid| valid.len() == n));

    let mut global_mask = vec![true; n];

    let mask_addr = global_mask.as_mut_ptr() as usize;
    (0..ends.len()).into_par_iter().for_each_init(
        || {
            (
                Vec::<usize>::new(),
                Vec::<f64>::new(),
                Vec::<f64>::new(),
                Vec::<f64>::new(),
            )
        },
        |(valid_indices_buf, lat_buf, lng_buf, time_buf), i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            if start >= end {
                return;
            }

            let user_indices = &sorted_indices[start..end];
            valid_indices_buf.clear();
            lat_buf.clear();
            lng_buf.clear();
            time_buf.clear();
            let mask_ptr = mask_addr as *mut bool;

            for &idx in user_indices {
                if is_valid_simplify_row(latitudes, longitudes, timestamps_s, valid_rows, idx) {
                    valid_indices_buf.push(idx);
                    lat_buf.push(latitudes[idx]);
                    lng_buf.push(longitudes[idx]);
                    time_buf.push(timestamps_s[idx]);
                } else {
                    unsafe {
                        *mask_ptr.add(idx) = false;
                    }
                }
            }

            let keep = simplify_user_slice(lat_buf, lng_buf, time_buf, config);
            for (local_idx, &keep_value) in keep.iter().enumerate() {
                let absolute_original_idx = valid_indices_buf[local_idx];
                unsafe {
                    *mask_ptr.add(absolute_original_idx) = keep_value;
                }
            }
        },
    );

    global_mask
}
