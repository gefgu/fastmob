use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use skmob2_core::preprocessing::cdr::{
    cdr_approx_travel_minutes_impl, cdr_trip_indices_impl, cdr_visitation_stays_impl,
};

type VisitationBatchResult = (Vec<usize>, Vec<usize>, Vec<f64>, Vec<bool>);
type TripBatchResult = (Vec<usize>, Vec<usize>);

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
    ranges: Vec<(usize, usize)>,
) -> PyResult<VisitationBatchResult> {
    cdr_visitation_stays_impl(&venue_codes, &timestamps_s, &ranges)
        .map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn cdr_trip_indices(
    has_departure_timestamp: Vec<bool>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<TripBatchResult> {
    cdr_trip_indices_impl(&has_departure_timestamp, &ranges).map_err(PyValueError::new_err)
}
