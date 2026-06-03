use rayon::prelude::*;

use crate::utils::haversine::haversine_km;
use crate::utils::helpers::validate_ranges;

pub type VisitationBatchResult = (Vec<usize>, Vec<usize>, Vec<f64>, Vec<bool>);
pub type TripBatchResult = (Vec<usize>, Vec<usize>);

pub fn cdr_approx_travel_minutes_impl(
    origin_lats: &[f64],
    origin_lons: &[f64],
    destination_lats: &[f64],
    destination_lons: &[f64],
    avg_speed_kmh: f64,
    circuity: f64,
) -> Result<Vec<f64>, String> {
    let n = origin_lats.len();
    if origin_lons.len() != n || destination_lats.len() != n || destination_lons.len() != n {
        return Err(
            "origin and destination coordinate arrays must have the same length".to_string(),
        );
    }
    if !avg_speed_kmh.is_finite() || avg_speed_kmh <= 0.0 {
        return Err("avg_speed_kmh must be a positive finite value".to_string());
    }
    if !circuity.is_finite() || circuity < 0.0 {
        return Err("circuity must be a non-negative finite value".to_string());
    }

    let travel_minutes = (0..n)
        .into_par_iter()
        .map(|i| {
            let distance_km = haversine_km(
                origin_lats[i],
                origin_lons[i],
                destination_lats[i],
                destination_lons[i],
            );
            (distance_km * circuity / avg_speed_kmh) * 60.0
        })
        .collect();

    Ok(travel_minutes)
}

pub fn cdr_visitation_stays_impl(
    venue_codes: &[i64],
    timestamps_s: &[f64],
    ranges: &[(usize, usize)],
) -> Result<VisitationBatchResult, String> {
    if venue_codes.len() != timestamps_s.len() {
        return Err("venue_codes and timestamps_s must have the same length".to_string());
    }
    validate_ranges(venue_codes.len(), ranges)?;

    let mut stay_starts: Vec<usize> = Vec::new();
    let mut stay_ends: Vec<usize> = Vec::new();
    let mut end_timestamps_s: Vec<f64> = Vec::new();
    let mut has_end_timestamp: Vec<bool> = Vec::new();

    for &(user_start, user_end) in ranges {
        if user_start == user_end {
            continue;
        }

        let mut stay_start = user_start;
        let mut i = user_start + 1;

        while i <= user_end {
            let at_user_end = i == user_end;
            let venue_changed = !at_user_end && venue_codes[i] != venue_codes[i - 1];

            if at_user_end || venue_changed {
                stay_starts.push(stay_start);
                stay_ends.push(i);

                if at_user_end {
                    end_timestamps_s.push(0.0);
                    has_end_timestamp.push(false);
                } else {
                    let last_ts = timestamps_s[i - 1];
                    let next_ts = timestamps_s[i];
                    end_timestamps_s.push(last_ts + (next_ts - last_ts) / 2.0);
                    has_end_timestamp.push(true);
                }

                stay_start = i;
            }

            i += 1;
        }
    }

    Ok((stay_starts, stay_ends, end_timestamps_s, has_end_timestamp))
}

pub fn cdr_trip_indices_impl(
    has_departure_timestamp: &[bool],
    ranges: &[(usize, usize)],
) -> Result<TripBatchResult, String> {
    validate_ranges(has_departure_timestamp.len(), ranges)?;

    let mut origins: Vec<usize> = Vec::new();
    let mut destinations: Vec<usize> = Vec::new();

    for &(user_start, user_end) in ranges {
        if user_end.saturating_sub(user_start) < 2 {
            continue;
        }

        for (origin_idx, has_departure) in has_departure_timestamp
            .iter()
            .enumerate()
            .take(user_end - 1)
            .skip(user_start)
        {
            if *has_departure {
                origins.push(origin_idx);
                destinations.push(origin_idx + 1);
            }
        }
    }

    Ok((origins, destinations))
}
