use pyo3::prelude::*;

use crate::haversine::haversine_km;

type StayLocationsBatchResult = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<usize>);

struct Stop {
    lat: f64,
    lng: f64,
    entry_time_s: f64,
    leaving_time_s: f64,
}

fn detect_stops_for_user(
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

    let mut sum_lat: Vec<f64> = vec![lat_0];
    let mut sum_lon: Vec<f64> = vec![lon_0];
    let mut sum_t: Vec<f64> = vec![t_0];
    let mut speeds_kmh: Vec<f64> = Vec::new();

    let lendata = n - 1;

    for i in 0..lendata {
        let lat = lats[i + 1];
        let lon = lngs[i + 1];
        let t = times[i + 1];

        // Check for missing data gap
        let gap_min = (times[i + 1] - times[i]) / 60.0;
        if gap_min > no_data_for_minutes {
            lat_0 = lat;
            lon_0 = lon;
            t_0 = t;
            sum_lat = Vec::new();
            sum_lon = Vec::new();
            sum_t = Vec::new();
            speeds_kmh = Vec::new();

            sum_lat.push(lat);
            sum_lon.push(lon);
            sum_t.push(t);
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
                let mut effective_sum_lat = sum_lat.clone();
                let mut effective_sum_lon = sum_lon.clone();

                if min_speed_kmh.is_finite() && !speeds_kmh.is_empty() {
                    // Find how many trailing points exceed min_speed_kmh
                    let mut j = 0usize;
                    for k in 1..speeds_kmh.len() {
                        if speeds_kmh[speeds_kmh.len() - k] < min_speed_kmh {
                            j = k;
                            break;
                        }
                    }
                    if j > 1 {
                        // estimated_final_t = sum_t[-j + 1]
                        let trim_idx = sum_t.len().saturating_sub(j - 1);
                        if trim_idx > 0 && trim_idx < sum_t.len() {
                            final_t = sum_t[trim_idx];
                            effective_sum_lat = effective_sum_lat[..trim_idx - 1].to_vec();
                            effective_sum_lon = effective_sum_lon[..trim_idx - 1].to_vec();
                        }
                    }
                }

                let dur_min = (final_t - t_0) / 60.0;
                if !effective_sum_lat.is_empty() && dur_min > minutes_for_a_stop {
                    let stop_lat = median(&effective_sum_lat);
                    let stop_lon = median(&effective_sum_lon);
                    stops.push(Stop {
                        lat: stop_lat,
                        lng: stop_lon,
                        entry_time_s: t_0,
                        leaving_time_s: final_t,
                    });
                }
            }

            // Reset accumulator
            lat_0 = lat;
            lon_0 = lon;
            t_0 = t;
            sum_lat = Vec::new();
            sum_lon = Vec::new();
            sum_t = Vec::new();
            speeds_kmh = Vec::new();
        }

        sum_lat.push(lat);
        sum_lon.push(lon);
        sum_t.push(t);
    }

    stops
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let mut sorted = v.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn detect_stay_locations_batch(
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    timestamps_s: Vec<f64>,
    ranges: Vec<(usize, usize)>,
    stop_radius_km: f64,
    minutes_for_a_stop: f64,
    no_data_for_minutes: f64,
    min_speed_kmh: f64,
) -> PyResult<StayLocationsBatchResult> {
    let mut out_lats: Vec<f64> = Vec::new();
    let mut out_lngs: Vec<f64> = Vec::new();
    let mut entry_times: Vec<f64> = Vec::new();
    let mut leaving_times: Vec<f64> = Vec::new();
    let mut user_range_idx: Vec<usize> = Vec::new();

    for (range_idx, &(start, end)) in ranges.iter().enumerate() {
        let stops = detect_stops_for_user(
            &latitudes[start..end],
            &longitudes[start..end],
            &timestamps_s[start..end],
            stop_radius_km,
            minutes_for_a_stop,
            no_data_for_minutes,
            min_speed_kmh,
        );
        for stop in stops {
            out_lats.push(stop.lat);
            out_lngs.push(stop.lng);
            entry_times.push(stop.entry_time_s);
            leaving_times.push(stop.leaving_time_s);
            user_range_idx.push(range_idx);
        }
    }

    Ok((
        out_lats,
        out_lngs,
        entry_times,
        leaving_times,
        user_range_idx,
    ))
}
