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

    let mut sum_lat: Vec<f64> = vec![lat_0];
    let mut sum_lon: Vec<f64> = vec![lon_0];
    let mut speeds_kmh: Vec<f64> = Vec::new();

    let lendata = n - 1;

    for i in 0..lendata {
        let lat = lats[i + 1];
        let lon = lngs[i + 1];
        let t = times[i + 1];

        let gap_min = (times[i + 1] - times[i]) / 60.0;
        if gap_min > no_data_for_minutes {
            lat_0 = lat;
            lon_0 = lon;
            t_0 = t;
            segment_start = i + 1;
            sum_lat.clear();
            sum_lon.clear();
            speeds_kmh.clear();

            sum_lat.push(lat);
            sum_lon.push(lon);
            continue;
        }

        let dt_min = (t - t_0) / 60.0;
        let dr = haversine_km(lat_0, lon_0, lat, lon);

        let speed = if dt_min > 0.0 {
            dr / dt_min * 60.0
        } else {
            0.0
        };
        speeds_kmh.push(speed);

        let is_last = i == lendata - 1;

        if dr > stop_radius_km || is_last {
            if dt_min > minutes_for_a_stop || is_last {
                let mut final_t = t;
                let mut lat_end = sum_lat.len();

                if min_speed_kmh.is_finite() && !speeds_kmh.is_empty() {
                    let mut j = 0usize;
                    for k in 1..speeds_kmh.len() {
                        if speeds_kmh[speeds_kmh.len() - k] < min_speed_kmh {
                            j = k;
                            break;
                        }
                    }
                    if j > 1 {
                        let trim_idx = sum_lat.len().saturating_sub(j - 1);
                        if trim_idx > 0 && trim_idx < sum_lat.len() {
                            final_t = times[segment_start + trim_idx];
                            lat_end = trim_idx.saturating_sub(1);
                        }
                    }
                }

                let dur_min = (final_t - t_0) / 60.0;
                if lat_end > 0 && dur_min > minutes_for_a_stop {
                    let stop_lat = median_slice_in_place(&mut sum_lat[..lat_end]);
                    let stop_lon = median_slice_in_place(&mut sum_lon[..lat_end]);
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
            sum_lat.clear();
            sum_lon.clear();
            speeds_kmh.clear();
        }

        sum_lat.push(lat);
        sum_lon.push(lon);
    }

    stops
}

#[allow(clippy::too_many_arguments)]
fn detect_stops_for_user_indexed(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    user_indices: &[usize],
    stop_radius_km: f64,
    minutes_for_a_stop: f64,
    no_data_for_minutes: f64,
    min_speed_kmh: f64,
) -> Vec<Stop> {
    if user_indices.is_empty() {
        return Vec::new();
    }
    let lats_u: Vec<f64> = user_indices.iter().map(|&i| lats[i]).collect();
    let lngs_u: Vec<f64> = user_indices.iter().map(|&i| lngs[i]).collect();
    let times_u: Vec<f64> = user_indices.iter().map(|&i| times[i]).collect();
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
    ranges: &[(usize, usize)],
    stop_radius_km: f64,
    minutes_for_a_stop: f64,
    no_data_for_minutes: f64,
    min_speed_kmh: f64,
) -> StayLocationsBatchResult {
    let per_user_stops: Vec<Vec<Stop>> = ranges
        .par_iter()
        .map(|&(start, end)| {
            detect_stops_for_user_indexed(
                latitudes,
                longitudes,
                timestamps_s,
                &sorted_indices[start..end],
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
