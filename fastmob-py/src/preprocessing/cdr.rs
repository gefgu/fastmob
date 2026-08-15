use fastmob_core::preprocessing::cdr::{
    cdr_approx_travel_minutes_impl, cdr_trip_indices_impl, cdr_visitation_stays_impl,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::ranges_from_ends;

type VisitationBatchResult = (Vec<u64>, Vec<u64>, Vec<f64>, Vec<bool>);
type TripBatchResult = (Vec<u64>, Vec<u64>);

fn u64_offsets(values: Vec<usize>) -> Vec<u64> {
    values.into_iter().map(|value| value as u64).collect()
}

#[pyfunction]
pub fn cdr_approx_travel_minutes(
    origin_lats: Vec<f64>,
    origin_lons: Vec<f64>,
    destination_lats: Vec<f64>,
    destination_lons: Vec<f64>,
    avg_speed_kmh: f64,
    circuity: f64,
) -> PyResult<Vec<f64>> {
    cdr_approx_travel_minutes_impl(
        &origin_lats,
        &origin_lons,
        &destination_lats,
        &destination_lons,
        avg_speed_kmh,
        circuity,
    )
    .map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn cdr_visitation_stays(
    venue_codes: Vec<i64>,
    timestamps_s: Vec<f64>,
    ends: ArrowPyArray,
) -> PyResult<VisitationBatchResult> {
    let ranges = ranges_from_ends(ends, venue_codes.len())?;
    cdr_visitation_stays_impl(&venue_codes, &timestamps_s, &ranges)
        .map(|(starts, ends, values, valid)| {
            (u64_offsets(starts), u64_offsets(ends), values, valid)
        })
        .map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn cdr_trip_indices(
    has_departure_timestamp: Vec<bool>,
    ends: ArrowPyArray,
) -> PyResult<TripBatchResult> {
    let ranges = ranges_from_ends(ends, has_departure_timestamp.len())?;
    cdr_trip_indices_impl(&has_departure_timestamp, &ranges)
        .map(|(starts, ends)| (u64_offsets(starts), u64_offsets(ends)))
        .map_err(PyValueError::new_err)
}
