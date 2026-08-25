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

pub fn detect_stops_for_user(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    stop_radius_km: f64,
    minutes_for_a_stop: f64,
    no_data_for_minutes: f64,
    min_speed_kmh: f64,
) -> Vec<Stop> {
    let n = lats.len();
    if n == 0 {
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

        if dr > stop_radius_km || is_last {
            if dt_min > minutes_for_a_stop || is_last {
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
                if lat_end > 0 && dur_min > minutes_for_a_stop {
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

        let stops = detect_stops_for_user(&lats, &lngs, &times, 0.2, 20.0, 1e12, 5.0);

        assert_eq!(stops.len(), 1);
        assert_eq!(stops[0].entry_time_s, 0.0);
        assert_eq!(stops[0].leaving_time_s, 1800.0);
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
) -> Vec<Stop> {
    let valid_user_indices: Vec<usize> = user_indices
        .iter()
        .copied()
        .filter(|&idx| {
            valid_rows.is_none_or(|v| v[idx])
                && lats[idx].is_finite()
                && lngs[idx].is_finite()
                && times[idx].is_finite()
        })
        .collect();
    if valid_user_indices.is_empty() {
        return Vec::new();
    }
    let lats_u: Vec<f64> = valid_user_indices.iter().map(|&i| lats[i]).collect();
    let lngs_u: Vec<f64> = valid_user_indices.iter().map(|&i| lngs[i]).collect();
    let times_u: Vec<f64> = valid_user_indices.iter().map(|&i| times[i]).collect();
    detect_stops_for_user(
        &lats_u,
        &lngs_u,
        &times_u,
        stop_radius_km,
        minutes_for_a_stop,
        no_data_for_minutes,
        min_speed_kmh,
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
