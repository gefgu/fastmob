use rayon::prelude::*;

use crate::utils::haversine::haversine_km;
use crate::utils::helpers::median_slice_in_place;

pub type StayLocationsBatchResult = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>);

pub struct Stop {
    pub lat: f64,
    pub lng: f64,
    pub entry_time_s: f64,
    pub leaving_time_s: f64,
}

pub fn sub_cmp_std(a: &[f64], b: &[f64], c: f64) -> Vec<bool> {
    assert_eq!(a.len(), b.len(), "Input slices must have the same length");

    a.iter()
        .zip(b.iter())
        .map(|(&a_val, &b_val)| (a_val - b_val) > c)
        .collect()
}

/// Drops rows that are exact duplicates (same lat, lng, and time) of the row
/// immediately before them, mirroring trackintel's `exclude_duplicate_pfs=True`
/// default. Assumes rows are already sorted by time within a user; duplicate
/// timestamps put duplicate rows adjacent, so a consecutive check is enough.
///
/// Three flat elementwise-equality passes (same shape as `sub_cmp_std`)
/// rather than one branchy fused loop or a "skip entirely if nothing to
/// remove" prescan: measured head-to-head on full-scale GeoLife, both
/// alternatives were *slower* than this, because duplicate positionfixes are
/// common enough per user in real GPS logs that a prescan rarely gets to
/// skip anything, and a fused per-element branch defeats autovectorization
/// that these flat compares get for free.
fn dedup_consecutive(lats: &[f64], lngs: &[f64], times: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n = lats.len();
    if n <= 1 {
        return (lats.to_vec(), lngs.to_vec(), times.to_vec());
    }

    let eq_prev = |values: &[f64]| -> Vec<bool> {
        values[1..]
            .iter()
            .zip(values[..n - 1].iter())
            .map(|(&a, &b)| a == b)
            .collect()
    };
    let lat_dup = eq_prev(lats);
    let lng_dup = eq_prev(lngs);
    let time_dup = eq_prev(times);

    let mut out_lat = Vec::with_capacity(n);
    let mut out_lng = Vec::with_capacity(n);
    let mut out_t = Vec::with_capacity(n);
    out_lat.push(lats[0]);
    out_lng.push(lngs[0]);
    out_t.push(times[0]);

    for i in 0..n - 1 {
        if !(lat_dup[i] && lng_dup[i] && time_dup[i]) {
            out_lat.push(lats[i + 1]);
            out_lng.push(lngs[i + 1]);
            out_t.push(times[i + 1]);
        }
    }

    (out_lat, out_lng, out_t)
}

/// Sliding-window stop detection (Li et al. 2008 / trackintel's algorithm)
/// over already-deduplicated, chronologically-sorted rows for one user.
/// Callers are responsible for deduplication -- both `detect_stops_for_user`
/// and `detect_stops_for_user_indexed` call `dedup_consecutive` before this.
#[allow(clippy::too_many_arguments)]
fn detect_stops_for_user_core(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    stop_radius_km: f64,
    minutes_for_a_stop: f64,
    no_data_for_minutes: f64,
    min_speed_kmh: f64,
    include_last: bool,
) -> Vec<Stop> {
    let n = lats.len();
    if n <= 1 {
        return Vec::new();
    }

    let mut stops: Vec<Stop> = Vec::new();

    let mut lat_0 = lats[0];
    let mut lon_0 = lngs[0];
    let mut t_0 = times[0];
    let mut segment_start = 0usize;

    let mut scratch_lat: Vec<f64> = Vec::new();
    let mut scratch_lon: Vec<f64> = Vec::new();
    let mut speeds_kmh: Vec<f64> = Vec::new();

    let minutes_for_a_stop = minutes_for_a_stop * 60.0; // convert to seconds
    let no_data_for_minutes = no_data_for_minutes * 60.0; // convert to seconds

    let time_gap_mask = sub_cmp_std(&times[1..], &times[..n - 1], no_data_for_minutes);

    let lendata = n - 1;

    for i in 0..lendata {
        let lat = lats[i + 1];
        let lon = lngs[i + 1];
        let t = times[i + 1];

        if time_gap_mask[i] {
            lat_0 = lat;
            lon_0 = lon;
            t_0 = t;
            segment_start = i + 1;
            speeds_kmh.clear();
            continue;
        }

        let dt_min = t - t_0;
        let dr = haversine_km(lat_0, lon_0, lat, lon);

        let speed = if dt_min > 0.0 {
            dr / dt_min * 3600.0
        } else {
            0.0
        };
        speeds_kmh.push(speed);

        let is_last = i == lendata - 1;
        let force_close = include_last && is_last;

        if dr >= stop_radius_km || force_close {
            if dt_min >= minutes_for_a_stop || force_close {
                let mut final_t = t;
                let mut lat_end = i + 1 - segment_start;

                if min_speed_kmh.is_finite() && !speeds_kmh.is_empty() {
                    if let Some(pos) = speeds_kmh.iter().rev().position(|&s| s < min_speed_kmh) {
                        let j = pos + 1;
                        let trim_idx = lat_end.saturating_sub(j - 1);
                        if trim_idx > 0 && trim_idx < lat_end {
                            final_t = times[segment_start + trim_idx];
                            lat_end = trim_idx.saturating_sub(1);
                        }
                    }
                }

                let dur_min = final_t - t_0;
                if lat_end > 0 && dur_min >= minutes_for_a_stop {
                    scratch_lat.clear();
                    scratch_lon.clear();
                    scratch_lat.extend_from_slice(&lats[segment_start..segment_start + lat_end]);
                    scratch_lon.extend_from_slice(&lngs[segment_start..segment_start + lat_end]);
                    let stop_lat = median_slice_in_place(&mut scratch_lat);
                    let stop_lon = median_slice_in_place(&mut scratch_lon);
                    stops.push(Stop {
                        lat: stop_lat,
                        lng: stop_lon,
                        entry_time_s: t_0,
                        leaving_time_s: final_t,
                    });
                }
            }

            lat_0 = lat;
            lon_0 = lon;
            t_0 = t;
            segment_start = i + 1;
            speeds_kmh.clear();
        }
    }

    stops
}

#[allow(clippy::too_many_arguments)]
pub fn detect_stops_for_user(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    stop_radius_km: f64,
    minutes_for_a_stop: f64,
    no_data_for_minutes: f64,
    min_speed_kmh: f64,
    include_last: bool,
) -> Vec<Stop> {
    let (lats, lngs, times) = dedup_consecutive(lats, lngs, times);
    detect_stops_for_user_core(
        &lats,
        &lngs,
        &times,
        stop_radius_km,
        minutes_for_a_stop,
        no_data_for_minutes,
        min_speed_kmh,
        include_last,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn min_speed_is_interpreted_in_kilometres_per_hour() {
        // The first four points form a 30-minute stop. The final two points
        // move at roughly 33 km/h, so a 5 km/h threshold must trim the stop
        // at the last stationary point rather than treating the moving tail
        // as below the threshold.
        let lats = [0.0, 0.0, 0.0, 0.0, 0.05, 0.10];
        let lngs = [0.0; 6];
        let times = [0.0, 600.0, 1200.0, 1800.0, 2400.0, 3000.0];

        let stops = detect_stops_for_user(&lats, &lngs, &times, 0.2, 20.0, 1e12, 5.0, true);

        assert_eq!(stops.len(), 1);
        assert_eq!(stops[0].entry_time_s, 0.0);
        assert_eq!(stops[0].leaving_time_s, 1800.0);
    }

    #[test]
    fn consecutive_duplicate_positionfixes_are_dropped() {
        // Row at t=300 repeats t=0's fix exactly; it must not be treated as a
        // second, distinct sample when checking the stop's duration.
        let lats = [0.0, 0.0, 0.0, 0.0, 0.0];
        let lngs = [0.0; 5];
        let times = [0.0, 0.0, 300.0, 300.0, 1500.0];

        let stops = detect_stops_for_user(&lats, &lngs, &times, 0.2, 20.0, 1e12, f64::INFINITY, true);

        assert_eq!(stops.len(), 1);
        assert_eq!(stops[0].entry_time_s, 0.0);
        assert_eq!(stops[0].leaving_time_s, 1500.0);
    }

    #[test]
    fn include_last_controls_the_trailing_open_stay() {
        // The user is still stationary when tracking ends -- with
        // include_last=false (trackintel's default) that trailing stay must
        // be omitted; with include_last=true it must be emitted.
        let lats = [0.0, 0.0, 0.0];
        let lngs = [0.0; 3];
        let times = [0.0, 600.0, 1500.0];

        let omitted = detect_stops_for_user(&lats, &lngs, &times, 0.2, 20.0, 1e12, f64::INFINITY, false);
        assert!(omitted.is_empty());

        let included = detect_stops_for_user(&lats, &lngs, &times, 0.2, 20.0, 1e12, f64::INFINITY, true);
        assert_eq!(included.len(), 1);
        assert_eq!(included[0].entry_time_s, 0.0);
        assert_eq!(included[0].leaving_time_s, 1500.0);
    }
}

#[allow(clippy::too_many_arguments)]
fn detect_stops_for_user_indexed(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    user_indices: &[usize],
    valid_rows: Option<&[bool]>,
    stop_radius_km: f64,
    minutes_for_a_stop: f64,
    no_data_for_minutes: f64,
    min_speed_kmh: f64,
    include_last: bool,
) -> Vec<Stop> {
    // Fuses the valid-row filter and the gather-by-index copy into one pass
    // (the previous shape was filter -> Vec<usize>, then 3 separate maps off
    // that index vec: 4 passes/allocations for what's really one job).
    // Dedup is a separate, deliberately non-fused bulk pass -- see
    // `dedup_consecutive`'s docs for why fusing it in here as a per-element
    // "compare to the last kept row" check measured slower, not faster.
    let mut lats_u: Vec<f64> = Vec::with_capacity(user_indices.len());
    let mut lngs_u: Vec<f64> = Vec::with_capacity(user_indices.len());
    let mut times_u: Vec<f64> = Vec::with_capacity(user_indices.len());

    for &idx in user_indices {
        let valid = valid_rows.is_none_or(|v| v[idx])
            && lats[idx].is_finite()
            && lngs[idx].is_finite()
            && times[idx].is_finite();
        if valid {
            lats_u.push(lats[idx]);
            lngs_u.push(lngs[idx]);
            times_u.push(times[idx]);
        }
    }

    let (lats_u, lngs_u, times_u) = dedup_consecutive(&lats_u, &lngs_u, &times_u);

    detect_stops_for_user_core(
        &lats_u,
        &lngs_u,
        &times_u,
        stop_radius_km,
        minutes_for_a_stop,
        no_data_for_minutes,
        min_speed_kmh,
        include_last,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn detect_stay_locations_batch_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps_s: &[f64],
    ranges: &[(usize, usize)],
    stop_radius_km: f64,
    minutes_for_a_stop: f64,
    no_data_for_minutes: f64,
    min_speed_kmh: f64,
    include_last: bool,
) -> StayLocationsBatchResult {
    let per_user_stops: Vec<Vec<Stop>> = ranges
        .par_iter()
        .map(|&(start, end)| {
            detect_stops_for_user(
                &latitudes[start..end],
                &longitudes[start..end],
                &timestamps_s[start..end],
                stop_radius_km,
                minutes_for_a_stop,
                no_data_for_minutes,
                min_speed_kmh,
                include_last,
            )
        })
        .collect();

    flatten_stops(per_user_stops)
}

#[allow(clippy::too_many_arguments)]
pub fn detect_stay_locations_batch_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps_s: &[f64],
    sorted_indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
    stop_radius_km: f64,
    minutes_for_a_stop: f64,
    no_data_for_minutes: f64,
    min_speed_kmh: f64,
    include_last: bool,
) -> StayLocationsBatchResult {
    let per_user_stops: Vec<Vec<Stop>> = (0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            detect_stops_for_user_indexed(
                latitudes,
                longitudes,
                timestamps_s,
                &sorted_indices[start..end],
                valid_rows,
                stop_radius_km,
                minutes_for_a_stop,
                no_data_for_minutes,
                min_speed_kmh,
                include_last,
            )
        })
        .collect();

    flatten_stops(per_user_stops)
}

fn flatten_stops(per_user_stops: Vec<Vec<Stop>>) -> StayLocationsBatchResult {
    let n_stops: usize = per_user_stops.iter().map(|s| s.len()).sum();
    let mut out_lats = Vec::with_capacity(n_stops);
    let mut out_lngs = Vec::with_capacity(n_stops);
    let mut entry_times = Vec::with_capacity(n_stops);
    let mut leaving_times = Vec::with_capacity(n_stops);
    let mut user_range_idx = Vec::with_capacity(n_stops);

    for (range_idx, stops) in per_user_stops.iter().enumerate() {
        for stop in stops {
            out_lats.push(stop.lat);
            out_lngs.push(stop.lng);
            entry_times.push(stop.entry_time_s);
            leaving_times.push(stop.leaving_time_s);
            user_range_idx.push(range_idx);
        }
    }

    (
        out_lats,
        out_lngs,
        entry_times,
        leaving_times,
        user_range_idx,
    )
}
