//! Named trajectory outlier-detection algorithms.
//!
//! `fastmob.preprocessing.filter` exposes a `method=` string; the
//! pre-existing `"speed"` method (with optional loop detection) keeps using
//! `filter_traj.rs`'s standalone `FilterConfig` / `filter_trajectory_impl`
//! pipeline completely unchanged, for exact backward compatibility. This
//! module adds four *additional* named methods — `hampel`, `greedy`,
//! `smart_greedy`, `zheng` — behind their own [`OutlierConfig`], dispatched
//! the same way `simplify/mod.rs` dispatches its seven methods: one
//! `method` enum, one flat config struct carrying every method's
//! parameters, one per-user dispatch function, and batched
//! contiguous-range / indexed entry points that return a boolean keep-mask.
//!
//! Output-Sensitive outlier detection is explicitly out of scope (see the
//! project plan): it is MoveTK's largest/most complex detector, and
//! Greedy + SmartGreedy + Zheng already cover the "fast heuristic" and
//! "physics-based" niches this phase targets.

pub mod greedy;
pub mod hampel;
pub mod zheng;

use std::str::FromStr;

use rayon::prelude::*;

/// The four shipped named outlier-detection algorithms, in addition to the
/// pre-existing `"speed"` method (which stays on `filter_traj.rs`'s own
/// `FilterConfig` pipeline).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OutlierMethod {
    Hampel,
    Greedy,
    SmartGreedy,
    Zheng,
}

impl FromStr for OutlierMethod {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "hampel" => Ok(OutlierMethod::Hampel),
            "greedy" => Ok(OutlierMethod::Greedy),
            "smart_greedy" => Ok(OutlierMethod::SmartGreedy),
            "zheng" => Ok(OutlierMethod::Zheng),
            other => Err(format!("unknown outlier method: {other:?}")),
        }
    }
}

/// Parameters for every shipped outlier method. Only the field(s) relevant
/// to `method` are used per call, mirroring `SimplifyConfig`'s flat
/// carry-every-parameter shape.
#[derive(Clone, Copy)]
pub struct OutlierConfig {
    pub method: OutlierMethod,
    /// Centered rolling window length (points), used by `Hampel`.
    pub window_size: usize,
    /// Standard-deviation multiplier (scaled by 1.4826x MAD), used by
    /// `Hampel`.
    pub n_sigma: f64,
    /// Maximum physically-consistent speed in km/h, used by `Greedy`,
    /// `SmartGreedy`, and `Zheng`.
    pub max_speed_kmh: f64,
    /// Minimum consistent-run length required to keep a `Zheng` segment.
    pub min_seg_size: usize,
}

impl OutlierConfig {
    pub fn new(
        method: OutlierMethod,
        window_size: usize,
        n_sigma: f64,
        max_speed_kmh: f64,
        min_seg_size: usize,
    ) -> Self {
        OutlierConfig {
            method,
            window_size,
            n_sigma,
            max_speed_kmh,
            min_seg_size,
        }
    }
}

/// Run the configured algorithm over a single user's slice and return a
/// keep-mask the same length as the slice.
fn outlier_user_slice(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    config: &OutlierConfig,
) -> Vec<bool> {
    match config.method {
        OutlierMethod::Hampel => {
            hampel::hampel_keep_mask(lats, lngs, times, config.window_size, config.n_sigma)
        }
        OutlierMethod::Greedy => greedy::greedy_keep_mask(lats, lngs, times, config.max_speed_kmh),
        OutlierMethod::SmartGreedy => {
            greedy::smart_greedy_keep_mask(lats, lngs, times, config.max_speed_kmh)
        }
        OutlierMethod::Zheng => {
            zheng::zheng_keep_mask(lats, lngs, times, config.max_speed_kmh, config.min_seg_size)
        }
    }
}

fn is_valid_outlier_row(
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
/// @usedBy `fastmob-py/src/preprocessing/outliers_traj_py.rs::outlier_trajectory_{numpy,arrow}`.
pub fn outlier_trajectory_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps_s: &[f64],
    ranges: &[(usize, usize)],
    config: &OutlierConfig,
) -> Vec<bool> {
    let n = latitudes.len();
    let mut mask = vec![true; n];

    let mask_addr = mask.as_mut_ptr() as usize;
    ranges.par_iter().for_each(|&(start, end)| {
        let user_len = end - start;
        let user_mask = unsafe {
            std::slice::from_raw_parts_mut((mask_addr as *mut bool).add(start), user_len)
        };
        user_mask.copy_from_slice(&outlier_user_slice(
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
/// nulls natively: rows failing [`is_valid_outlier_row`] are marked `false`
/// in the output mask and excluded from the per-user algorithm input.
///
/// @usedBy `fastmob-py/src/preprocessing/outliers_traj_py.rs::outlier_trajectory_indexed_{numpy,arrow}`.
pub fn outlier_trajectory_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps_s: &[f64],
    sorted_indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
    config: &OutlierConfig,
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
                if is_valid_outlier_row(latitudes, longitudes, timestamps_s, valid_rows, idx) {
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

            let keep = outlier_user_slice(lat_buf, lng_buf, time_buf, config);
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
