use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::haversine::haversine_km;
use crate::utils::validate_ranges;

type VisitationBatchResult = (Vec<usize>, Vec<usize>, Vec<f64>, Vec<bool>);
type TripBatchResult = (Vec<usize>, Vec<usize>);

#[pyfunction]
pub(crate) fn cdr_approx_travel_minutes(
    origin_lats: Vec<f64>,
    origin_lons: Vec<f64>,
    destination_lats: Vec<f64>,
    destination_lons: Vec<f64>,
    avg_speed_kmh: f64,
    circuity: f64,
) -> PyResult<Vec<f64>> {
    let n = origin_lats.len();
    if origin_lons.len() != n || destination_lats.len() != n || destination_lons.len() != n {
        return Err(PyValueError::new_err(
            "origin and destination coordinate arrays must have the same length",
        ));
    }
    if !avg_speed_kmh.is_finite() || avg_speed_kmh <= 0.0 {
        return Err(PyValueError::new_err(
            "avg_speed_kmh must be a positive finite value",
        ));
    }
    if !circuity.is_finite() || circuity < 0.0 {
        return Err(PyValueError::new_err(
            "circuity must be a non-negative finite value",
        ));
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

#[pyfunction]
pub(crate) fn cdr_visitation_stays(
    venue_codes: Vec<i64>,
    timestamps_s: Vec<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<VisitationBatchResult> {
    if venue_codes.len() != timestamps_s.len() {
        return Err(PyValueError::new_err(
            "venue_codes and timestamps_s must have the same length",
        ));
    }
    validate_ranges(venue_codes.len(), &ranges)?;

    let mut stay_starts: Vec<usize> = Vec::new();
    let mut stay_ends: Vec<usize> = Vec::new();
    let mut end_timestamps_s: Vec<f64> = Vec::new();
    let mut has_end_timestamp: Vec<bool> = Vec::new();

    for &(user_start, user_end) in &ranges {
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

#[pyfunction]
pub(crate) fn cdr_trip_indices(
    has_departure_timestamp: Vec<bool>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<TripBatchResult> {
    validate_ranges(has_departure_timestamp.len(), &ranges)?;

    let mut origins: Vec<usize> = Vec::new();
    let mut destinations: Vec<usize> = Vec::new();

    for &(user_start, user_end) in &ranges {
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
