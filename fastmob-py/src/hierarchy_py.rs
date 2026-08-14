use fastmob_core::hierarchy::trips::{
    ToursResult, TriplegLengthsResult, TripsResult, tours_from_trips_impl,
    tripleg_lengths_attributed_impl, trips_from_timeline_impl,
};
use numpy::IntoPyArray;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_bool_values, arrow_i64_values, arrow_u8_values, arrow_u32_values, arrow_values,
    as_bool_array, as_f64_array, as_i64_array, as_u8_array, as_u32_array, f64_results_into_arrow,
    i64_results_into_arrow, u32_results_into_arrow,
};

type TriplegLengthsPyResult = (Py<PyAny>, Py<PyAny>, Py<PyAny>);
type TripsPyResult<'py> = (
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Bound<'py, numpy::PyArray1<usize>>,
    Py<PyAny>,
);
type ToursPyResult<'py> = (
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Bound<'py, numpy::PyArray1<usize>>,
    Py<PyAny>,
);

fn arrow_u32_output(py: Python<'_>, values: Vec<u32>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, u32_results_into_arrow(values))?.into_any())
}

fn arrow_i64_output(py: Python<'_>, values: Vec<i64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, i64_results_into_arrow(values))?.into_any())
}

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

#[pyfunction]
pub fn tripleg_lengths_attributed(
    py: Python<'_>,
    uid_codes: ArrowPyArray,
    segment_ids: ArrowPyArray,
    is_stop: ArrowPyArray,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
) -> PyResult<TriplegLengthsPyResult> {
    let uid_codes = as_u32_array(uid_codes, "uid_codes")?;
    let segment_ids = as_i64_array(segment_ids, "segment_ids")?;
    let is_stop = as_bool_array(is_stop, "is_stop")?;
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let is_stop_values = arrow_bool_values(&is_stop);
    let result: TriplegLengthsResult = py
        .detach(|| {
            tripleg_lengths_attributed_impl(
                arrow_u32_values(&uid_codes),
                arrow_i64_values(&segment_ids),
                &is_stop_values,
                arrow_values(&latitudes),
                arrow_values(&longitudes),
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok((
        arrow_u32_output(py, result.uid_codes)?,
        arrow_i64_output(py, result.segment_ids)?,
        arrow_f64_output(py, result.lengths_km)?,
    ))
}

fn trips_output_arrow<'py>(py: Python<'py>, result: TripsResult) -> PyResult<TripsPyResult<'py>> {
    Ok((
        arrow_u32_output(py, result.uid_codes)?,
        arrow_i64_output(py, result.started_at_us)?,
        arrow_i64_output(py, result.finished_at_us)?,
        arrow_i64_output(py, result.origin_staypoint_ids)?,
        arrow_i64_output(py, result.destination_staypoint_ids)?,
        result.tripleg_offsets.into_pyarray(py),
        arrow_i64_output(py, result.tripleg_ids)?,
    ))
}

#[pyfunction]
pub fn trips_from_timeline<'py>(
    py: Python<'py>,
    uid_codes: ArrowPyArray,
    kind_codes: ArrowPyArray,
    activity: ArrowPyArray,
    staypoint_ids: ArrowPyArray,
    tripleg_ids: ArrowPyArray,
    started_at_us: ArrowPyArray,
    finished_at_us: ArrowPyArray,
) -> PyResult<TripsPyResult<'py>> {
    let uid_codes = as_u32_array(uid_codes, "uid_codes")?;
    let kind_codes = as_u8_array(kind_codes, "kind_codes")?;
    let activity = as_bool_array(activity, "activity")?;
    let staypoint_ids = as_i64_array(staypoint_ids, "staypoint_ids")?;
    let tripleg_ids = as_i64_array(tripleg_ids, "tripleg_ids")?;
    let started_at_us = as_i64_array(started_at_us, "started_at_us")?;
    let finished_at_us = as_i64_array(finished_at_us, "finished_at_us")?;
    let activity_values = arrow_bool_values(&activity);
    let result = py
        .detach(|| {
            trips_from_timeline_impl(
                arrow_u32_values(&uid_codes),
                arrow_u8_values(&kind_codes),
                &activity_values,
                arrow_i64_values(&staypoint_ids),
                arrow_i64_values(&tripleg_ids),
                arrow_i64_values(&started_at_us),
                arrow_i64_values(&finished_at_us),
            )
        })
        .map_err(PyValueError::new_err)?;
    trips_output_arrow(py, result)
}

fn tours_output_arrow<'py>(py: Python<'py>, result: ToursResult) -> PyResult<ToursPyResult<'py>> {
    Ok((
        arrow_u32_output(py, result.uid_codes)?,
        arrow_i64_output(py, result.started_at_us)?,
        arrow_i64_output(py, result.finished_at_us)?,
        arrow_i64_output(py, result.location_ids)?,
        result.journey_offsets.into_pyarray(py),
        arrow_i64_output(py, result.journey_trip_ids)?,
    ))
}

#[pyfunction]
pub fn tours_from_trips<'py>(
    py: Python<'py>,
    uid_codes: ArrowPyArray,
    trip_ids: ArrowPyArray,
    started_at_us: ArrowPyArray,
    finished_at_us: ArrowPyArray,
    origin_location_ids: ArrowPyArray,
    destination_location_ids: ArrowPyArray,
) -> PyResult<ToursPyResult<'py>> {
    let uid_codes = as_u32_array(uid_codes, "uid_codes")?;
    let trip_ids = as_i64_array(trip_ids, "trip_ids")?;
    let started_at_us = as_i64_array(started_at_us, "started_at_us")?;
    let finished_at_us = as_i64_array(finished_at_us, "finished_at_us")?;
    let origin_location_ids = as_i64_array(origin_location_ids, "origin_location_ids")?;
    let destination_location_ids =
        as_i64_array(destination_location_ids, "destination_location_ids")?;
    let result = py
        .detach(|| {
            tours_from_trips_impl(
                arrow_u32_values(&uid_codes),
                arrow_i64_values(&trip_ids),
                arrow_i64_values(&started_at_us),
                arrow_i64_values(&finished_at_us),
                arrow_i64_values(&origin_location_ids),
                arrow_i64_values(&destination_location_ids),
            )
        })
        .map_err(PyValueError::new_err)?;
    tours_output_arrow(py, result)
}
