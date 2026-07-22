//! Named trajectory segmentation algorithms.
//!
//! `fastmob.preprocessing.segment` exposes a `method=` string that resolves
//! to a [`SegmentMethod`] variant carried by [`SegmentConfig`]. Unlike
//! `simplify`/`filter` (which both produce a boolean keep-mask, i.e. a row
//! *subset*), segmentation partitions rows without dropping any of them: the
//! output is a per-row `segment_id` (`u32`) column aligned 1:1 with the
//! input, restarting at `0` per user (see the project plan's "Segmentation
//! output shape" decision). This module's dispatch functions mirror
//! `simplify/mod.rs`/`outliers/mod.rs`'s batched contiguous-range / indexed
//! entry-point shape, but return `Vec<u32>` segment ids instead of
//! `Vec<bool>` keep-masks.
//!
//! Six named algorithms ship, all ported from MovingPandas'
//! `trajectory_splitter.py`: `AngleChange`, `ObservationGap`, `Speed`,
//! `Stop`, `ValueChange`, and `Temporal` (the last is not a distinct Rust
//! method — `fastmob/preprocessing/_segment.py::_prepare_temporal` builds an
//! integer bucket-id array from a truncated datetime column and delegates to
//! this module's `ValueChange` method with that array in place of an
//! arbitrary column's values).
//!
//! MoveTK's Monotone/Model-based/Brownian-bridge segmentation strategies are
//! intentionally out of scope (see the project plan): Monotone is a
//! generic criterion-family rather than one algorithm, Model-based needs a
//! real change-point/model-selection framework, and Brownian-bridge is a
//! full probabilistic movement model — each a separate research-grade
//! feature, while MovingPandas' 6 splitters already fully satisfy "a menu of
//! segmentation criteria".

pub mod angle_change;
pub mod observation_gap;
pub mod speed;
pub mod stop;
pub mod value_change;

use std::str::FromStr;

use rayon::prelude::*;

/// The five shipped named segmentation algorithms (`Temporal` routes through
/// `ValueChange`; see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentMethod {
    AngleChange,
    ObservationGap,
    Speed,
    Stop,
    ValueChange,
}

impl FromStr for SegmentMethod {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "angle_change" => Ok(SegmentMethod::AngleChange),
            "observation_gap" => Ok(SegmentMethod::ObservationGap),
            "speed" => Ok(SegmentMethod::Speed),
            "stop" => Ok(SegmentMethod::Stop),
            "value_change" => Ok(SegmentMethod::ValueChange),
            other => Err(format!("unknown segment method: {other:?}")),
        }
    }
}

/// Parameters for every shipped segmentation method. Only the field(s)
/// relevant to `method` are used per call, mirroring `SimplifyConfig`'s /
/// `OutlierConfig`'s flat carry-every-parameter shape.
#[derive(Clone, Copy)]
pub struct SegmentConfig {
    pub method: SegmentMethod,
    /// Minimum bearing change in degrees, used by `AngleChange`.
    pub min_angle_deg: f64,
    /// Minimum incoming speed (km/h) at which a point is considered for
    /// angle-change evaluation, used by `AngleChange`.
    pub angle_min_speed_kmh: f64,
    /// Time-gap threshold in seconds, used by `ObservationGap`.
    pub gap_s: f64,
    /// Minimum "moving" speed in km/h, used by `Speed`.
    pub speed_min_kmh: f64,
    /// Maximum "moving" speed in km/h, used by `Speed`.
    pub speed_max_kmh: f64,
    /// Minimum non-moving run duration in seconds required to bracket a new
    /// segment, used by `Speed`.
    pub duration_s: f64,
    /// Stop radius in km, used by `Stop` (forwarded to
    /// `detect_stops_for_user`).
    pub stop_radius_km: f64,
    /// Minimum stop duration in minutes, used by `Stop`.
    pub stop_minutes_for_a_stop: f64,
    /// Gap threshold in minutes above which data is treated as missing,
    /// used by `Stop`.
    pub stop_no_data_for_minutes: f64,
    /// Minimum speed (km/h) used to trim trailing high-speed points from a
    /// detected stop's end, used by `Stop`. `f64::INFINITY` disables
    /// trimming.
    pub stop_min_speed_kmh: f64,
}

impl SegmentConfig {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        method: SegmentMethod,
        min_angle_deg: f64,
        angle_min_speed_kmh: f64,
        gap_s: f64,
        speed_min_kmh: f64,
        speed_max_kmh: f64,
        duration_s: f64,
        stop_radius_km: f64,
        stop_minutes_for_a_stop: f64,
        stop_no_data_for_minutes: f64,
        stop_min_speed_kmh: f64,
    ) -> Self {
        SegmentConfig {
            method,
            min_angle_deg,
            angle_min_speed_kmh,
            gap_s,
            speed_min_kmh,
            speed_max_kmh,
            duration_s,
            stop_radius_km,
            stop_minutes_for_a_stop,
            stop_no_data_for_minutes,
            stop_min_speed_kmh,
        }
    }
}

/// Run the configured algorithm over a single user's slice and return a
/// `segment_id` array the same length as the slice, restarting at `0`.
///
/// `bucket_ids`, when present, is only consulted by `ValueChange` (and, by
/// extension, `Temporal`, which is handled entirely in Python by building a
/// bucket-id array and dispatching through `ValueChange`); if `ValueChange`
/// is selected without a `bucket_ids` array, every point is assigned to
/// segment `0` rather than panicking.
fn segment_user_slice(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    bucket_ids: Option<&[i64]>,
    config: &SegmentConfig,
) -> Vec<u32> {
    match config.method {
        SegmentMethod::AngleChange => angle_change::angle_change_segment_ids(
            lats,
            lngs,
            times,
            config.min_angle_deg,
            config.angle_min_speed_kmh,
        ),
        SegmentMethod::ObservationGap => {
            observation_gap::observation_gap_segment_ids(times, config.gap_s)
        }
        SegmentMethod::Speed => speed::speed_segment_ids(
            lats,
            lngs,
            times,
            config.speed_min_kmh,
            config.speed_max_kmh,
            config.duration_s,
        ),
        SegmentMethod::Stop => stop::stop_segment_ids(
            lats,
            lngs,
            times,
            config.stop_radius_km,
            config.stop_minutes_for_a_stop,
            config.stop_no_data_for_minutes,
            config.stop_min_speed_kmh,
        ),
        SegmentMethod::ValueChange => match bucket_ids {
            Some(ids) => value_change::value_change_segment_ids(ids),
            None => vec![0u32; lats.len()],
        },
    }
}

fn is_valid_segment_row(
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
/// user). Returns a `segment_id` array the same length as `latitudes`.
///
/// @usedBy `fastmob-py/src/preprocessing/segment_traj_py.rs::segment_trajectory_{numpy,arrow}`.
pub fn segment_trajectory_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps_s: &[f64],
    ranges: &[(usize, usize)],
    bucket_ids: Option<&[i64]>,
    config: &SegmentConfig,
) -> Vec<u32> {
    let n = latitudes.len();
    let mut out = vec![0u32; n];

    let out_addr = out.as_mut_ptr() as usize;
    ranges.par_iter().for_each(|&(start, end)| {
        let user_len = end - start;
        let user_out =
            unsafe { std::slice::from_raw_parts_mut((out_addr as *mut u32).add(start), user_len) };
        let user_bucket_ids = bucket_ids.map(|ids| &ids[start..end]);
        user_out.copy_from_slice(&segment_user_slice(
            &latitudes[start..end],
            &longitudes[start..end],
            &timestamps_s[start..end],
            user_bucket_ids,
            config,
        ));
    });

    out
}

/// Batched, indexed entry point (one call covers every user, rows addressed
/// through `sorted_indices`/`ends` rather than contiguous ranges). Handles
/// nulls natively: rows failing [`is_valid_segment_row`] are excluded from
/// the per-user algorithm input, but — unlike `simplify`/`outliers`'
/// keep-mask outputs — every row still receives a `segment_id` in the
/// output, since segmentation can never drop rows. An invalid row is
/// assigned the most recently computed segment id for its user (carrying
/// forward across the gap), defaulting to `0` if no valid row precedes it.
///
/// @usedBy `fastmob-py/src/preprocessing/segment_traj_py.rs::segment_trajectory_indexed_{numpy,arrow}`.
#[allow(clippy::too_many_arguments)]
pub fn segment_trajectory_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps_s: &[f64],
    sorted_indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
    bucket_ids: Option<&[i64]>,
    config: &SegmentConfig,
) -> Vec<u32> {
    let n = latitudes.len();
    debug_assert_eq!(longitudes.len(), n);
    debug_assert_eq!(timestamps_s.len(), n);
    debug_assert!(valid_rows.is_none_or(|valid| valid.len() == n));
    debug_assert!(bucket_ids.is_none_or(|ids| ids.len() == n));

    let mut global_out = vec![0u32; n];

    let out_addr = global_out.as_mut_ptr() as usize;
    (0..ends.len()).into_par_iter().for_each_init(
        || {
            (
                Vec::<bool>::new(),
                Vec::<f64>::new(),
                Vec::<f64>::new(),
                Vec::<f64>::new(),
                Vec::<i64>::new(),
            )
        },
        |(valid_flags_buf, lat_buf, lng_buf, time_buf, bucket_buf), i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            if start >= end {
                return;
            }

            let user_indices = &sorted_indices[start..end];
            valid_flags_buf.clear();
            lat_buf.clear();
            lng_buf.clear();
            time_buf.clear();
            bucket_buf.clear();
            let out_ptr = out_addr as *mut u32;

            for &idx in user_indices {
                let is_valid =
                    is_valid_segment_row(latitudes, longitudes, timestamps_s, valid_rows, idx);
                valid_flags_buf.push(is_valid);
                if is_valid {
                    lat_buf.push(latitudes[idx]);
                    lng_buf.push(longitudes[idx]);
                    time_buf.push(timestamps_s[idx]);
                    if let Some(ids) = bucket_ids {
                        bucket_buf.push(ids[idx]);
                    }
                }
            }

            let bucket_slice = bucket_ids.map(|_| bucket_buf.as_slice());
            let valid_ids = segment_user_slice(lat_buf, lng_buf, time_buf, bucket_slice, config);

            let mut valid_pos = 0usize;
            let mut last_id = 0u32;
            for (&idx, &is_valid) in user_indices.iter().zip(valid_flags_buf.iter()) {
                if is_valid {
                    last_id = valid_ids[valid_pos];
                    valid_pos += 1;
                }
                unsafe {
                    *out_ptr.add(idx) = last_id;
                }
            }
        },
    );

    global_out
}
