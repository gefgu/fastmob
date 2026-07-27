use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, BooleanArray, Int64Array, PrimitiveArray, UInt8Array, UInt64Array,
    types::{Int64Type, UInt8Type, UInt64Type},
};
use fastmob_core::hierarchy::trips::{
    ToursResult, TriplegLengthsResult, TripsResult, tours_from_trips_impl,
    tripleg_lengths_attributed_impl, trips_from_timeline_impl,
};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

type TriplegLengthsPyResult<'py> = (Py<PyAny>, Py<PyAny>, Py<PyAny>);
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

fn is_arrow_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.hasattr("__arrow_c_array__")
}

fn as_u64_array(arr: PyArray, name: &str) -> PyResult<PrimitiveArray<UInt64Type>> {
    let (array_ref, _field) = arr.into_inner();
    let array = array_ref
        .as_any()
        .downcast_ref::<UInt64Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected uint64 Arrow array for {name}")))?;
    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }
    Ok(array)
}

fn as_i64_array(arr: PyArray, name: &str) -> PyResult<PrimitiveArray<Int64Type>> {
    let (array_ref, _field) = arr.into_inner();
    let array = array_ref
        .as_any()
        .downcast_ref::<Int64Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected int64 Arrow array for {name}")))?;
    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }
    Ok(array)
}

fn as_u8_array(arr: PyArray, name: &str) -> PyResult<PrimitiveArray<UInt8Type>> {
    let (array_ref, _field) = arr.into_inner();
    let array = array_ref
        .as_any()
        .downcast_ref::<UInt8Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected uint8 Arrow array for {name}")))?;
    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }
    Ok(array)
}

fn as_bool_array(arr: PyArray, name: &str) -> PyResult<BooleanArray> {
    let (array_ref, _field) = arr.into_inner();
    let array = array_ref
        .as_any()
        .downcast_ref::<BooleanArray>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected bool Arrow array for {name}")))?;
    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }
    Ok(array)
}

fn arrow_u64_values(array: &PrimitiveArray<UInt64Type>) -> &[u64] {
    let start = array.offset();
    let end = start + array.len();
    &array.values()[start..end]
}

fn arrow_i64_values(array: &PrimitiveArray<Int64Type>) -> &[i64] {
    let start = array.offset();
    let end = start + array.len();
    &array.values()[start..end]
}

fn arrow_u8_values(array: &PrimitiveArray<UInt8Type>) -> &[u8] {
    let start = array.offset();
    let end = start + array.len();
    &array.values()[start..end]
}

fn arrow_bool_values(array: &BooleanArray) -> Vec<bool> {
    (0..array.len()).map(|i| array.value(i)).collect()
}

fn numpy_u64_output(py: Python<'_>, values: Vec<u64>) -> Py<PyAny> {
    values.into_pyarray(py).into_any().unbind()
}

fn numpy_i64_output(py: Python<'_>, values: Vec<i64>) -> Py<PyAny> {
    values.into_pyarray(py).into_any().unbind()
}

fn numpy_f64_output(py: Python<'_>, values: Vec<f64>) -> Py<PyAny> {
    values.into_pyarray(py).into_any().unbind()
}

fn arrow_u64_output(py: Python<'_>, values: Vec<u64>) -> PyResult<Py<PyAny>> {
    let array: ArrayRef = Arc::new(UInt64Array::from(values));
    Ok(Py::new(py, PyArray::from_array_ref(array))?.into_any())
}

fn arrow_i64_output(py: Python<'_>, values: Vec<i64>) -> PyResult<Py<PyAny>> {
    let array: ArrayRef = Arc::new(Int64Array::from(values));
    Ok(Py::new(py, PyArray::from_array_ref(array))?.into_any())
}

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    let array: ArrayRef = Arc::new(arrow_array::Float64Array::from(values));
    Ok(Py::new(py, PyArray::from_array_ref(array))?.into_any())
}

fn tripleg_lengths_numpy<'py>(
    py: Python<'py>,
    uid_codes: PyReadonlyArray1<'py, u64>,
    segment_ids: PyReadonlyArray1<'py, i64>,
    is_stop: PyReadonlyArray1<'py, bool>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
) -> PyResult<TriplegLengthsPyResult<'py>> {
    let uid_codes = uid_codes.as_slice()?;
    let segment_ids = segment_ids.as_slice()?;
    let is_stop = is_stop.as_slice()?;
    let latitudes = latitudes.as_slice()?;
    let longitudes = longitudes.as_slice()?;
    let result = py
        .detach(|| {
            tripleg_lengths_attributed_impl(uid_codes, segment_ids, is_stop, latitudes, longitudes)
        })
        .map_err(PyValueError::new_err)?;
    Ok((
        numpy_u64_output(py, result.uid_codes),
        numpy_i64_output(py, result.segment_ids),
        numpy_f64_output(py, result.lengths_km),
    ))
}

fn tripleg_lengths_arrow<'py>(
    py: Python<'py>,
    uid_codes: PyArray,
    segment_ids: PyArray,
    is_stop: PyArray,
    latitudes: PyArray,
    longitudes: PyArray,
) -> PyResult<TriplegLengthsPyResult<'py>> {
    let uid_codes = as_u64_array(uid_codes, "uid_codes")?;
    let segment_ids = as_i64_array(segment_ids, "segment_ids")?;
    let is_stop = as_bool_array(is_stop, "is_stop")?;
    let latitudes = crate::utils::as_f64_array(latitudes, "latitudes")?;
    let longitudes = crate::utils::as_f64_array(longitudes, "longitudes")?;
    let is_stop_values = arrow_bool_values(&is_stop);
    let result: TriplegLengthsResult = py
        .detach(|| {
            tripleg_lengths_attributed_impl(
                arrow_u64_values(&uid_codes),
                arrow_i64_values(&segment_ids),
                &is_stop_values,
                crate::utils::arrow_values(&latitudes),
                crate::utils::arrow_values(&longitudes),
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok((
        arrow_u64_output(py, result.uid_codes)?,
        arrow_i64_output(py, result.segment_ids)?,
        arrow_f64_output(py, result.lengths_km)?,
    ))
}

#[pyfunction]
pub fn tripleg_lengths_attributed<'py>(
    py: Python<'py>,
    uid_codes: &Bound<'py, PyAny>,
    segment_ids: &Bound<'py, PyAny>,
    is_stop: &Bound<'py, PyAny>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
) -> PyResult<TriplegLengthsPyResult<'py>> {
    if let (Ok(uid_codes), Ok(segment_ids), Ok(is_stop), Ok(latitudes), Ok(longitudes)) = (
        uid_codes.extract::<PyReadonlyArray1<u64>>(),
        segment_ids.extract::<PyReadonlyArray1<i64>>(),
        is_stop.extract::<PyReadonlyArray1<bool>>(),
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
    ) {
        return tripleg_lengths_numpy(py, uid_codes, segment_ids, is_stop, latitudes, longitudes);
    }

    if is_arrow_array(uid_codes)?
        && is_arrow_array(segment_ids)?
        && is_arrow_array(is_stop)?
        && is_arrow_array(latitudes)?
        && is_arrow_array(longitudes)?
    {
        return tripleg_lengths_arrow(
            py,
            uid_codes.extract::<PyArray>()?,
            segment_ids.extract::<PyArray>()?,
            is_stop.extract::<PyArray>()?,
            latitudes.extract::<PyArray>()?,
            longitudes.extract::<PyArray>()?,
        );
    }

    Err(PyTypeError::new_err(
        "tripleg length inputs must be all NumPy arrays or all Arrow arrays",
    ))
}

fn trips_output_numpy<'py>(py: Python<'py>, result: TripsResult) -> TripsPyResult<'py> {
    (
        numpy_u64_output(py, result.uid_codes),
        numpy_i64_output(py, result.started_at_us),
        numpy_i64_output(py, result.finished_at_us),
        numpy_i64_output(py, result.origin_staypoint_ids),
        numpy_i64_output(py, result.destination_staypoint_ids),
        result.tripleg_offsets.into_pyarray(py),
        numpy_i64_output(py, result.tripleg_ids),
    )
}

fn trips_output_arrow<'py>(py: Python<'py>, result: TripsResult) -> PyResult<TripsPyResult<'py>> {
    Ok((
        arrow_u64_output(py, result.uid_codes)?,
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
    uid_codes: &Bound<'py, PyAny>,
    kind_codes: &Bound<'py, PyAny>,
    activity: &Bound<'py, PyAny>,
    staypoint_ids: &Bound<'py, PyAny>,
    tripleg_ids: &Bound<'py, PyAny>,
    started_at_us: &Bound<'py, PyAny>,
    finished_at_us: &Bound<'py, PyAny>,
) -> PyResult<TripsPyResult<'py>> {
    if let (
        Ok(uid_codes),
        Ok(kind_codes),
        Ok(activity),
        Ok(staypoint_ids),
        Ok(tripleg_ids),
        Ok(started_at_us),
        Ok(finished_at_us),
    ) = (
        uid_codes.extract::<PyReadonlyArray1<u64>>(),
        kind_codes.extract::<PyReadonlyArray1<u8>>(),
        activity.extract::<PyReadonlyArray1<bool>>(),
        staypoint_ids.extract::<PyReadonlyArray1<i64>>(),
        tripleg_ids.extract::<PyReadonlyArray1<i64>>(),
        started_at_us.extract::<PyReadonlyArray1<i64>>(),
        finished_at_us.extract::<PyReadonlyArray1<i64>>(),
    ) {
        let uid_codes = uid_codes.as_slice()?;
        let kind_codes = kind_codes.as_slice()?;
        let activity = activity.as_slice()?;
        let staypoint_ids = staypoint_ids.as_slice()?;
        let tripleg_ids = tripleg_ids.as_slice()?;
        let started_at_us = started_at_us.as_slice()?;
        let finished_at_us = finished_at_us.as_slice()?;
        let result = py
            .detach(|| {
                trips_from_timeline_impl(
                    uid_codes,
                    kind_codes,
                    activity,
                    staypoint_ids,
                    tripleg_ids,
                    started_at_us,
                    finished_at_us,
                )
            })
            .map_err(PyValueError::new_err)?;
        return Ok(trips_output_numpy(py, result));
    }

    if is_arrow_array(uid_codes)?
        && is_arrow_array(kind_codes)?
        && is_arrow_array(activity)?
        && is_arrow_array(staypoint_ids)?
        && is_arrow_array(tripleg_ids)?
        && is_arrow_array(started_at_us)?
        && is_arrow_array(finished_at_us)?
    {
        let uid_codes = as_u64_array(uid_codes.extract::<PyArray>()?, "uid_codes")?;
        let kind_codes = as_u8_array(kind_codes.extract::<PyArray>()?, "kind_codes")?;
        let activity = as_bool_array(activity.extract::<PyArray>()?, "activity")?;
        let staypoint_ids = as_i64_array(staypoint_ids.extract::<PyArray>()?, "staypoint_ids")?;
        let tripleg_ids = as_i64_array(tripleg_ids.extract::<PyArray>()?, "tripleg_ids")?;
        let started_at_us = as_i64_array(started_at_us.extract::<PyArray>()?, "started_at_us")?;
        let finished_at_us = as_i64_array(finished_at_us.extract::<PyArray>()?, "finished_at_us")?;
        let activity_values = arrow_bool_values(&activity);
        let result = py
            .detach(|| {
                trips_from_timeline_impl(
                    arrow_u64_values(&uid_codes),
                    arrow_u8_values(&kind_codes),
                    &activity_values,
                    arrow_i64_values(&staypoint_ids),
                    arrow_i64_values(&tripleg_ids),
                    arrow_i64_values(&started_at_us),
                    arrow_i64_values(&finished_at_us),
                )
            })
            .map_err(PyValueError::new_err)?;
        return trips_output_arrow(py, result);
    }

    Err(PyTypeError::new_err(
        "trip timeline inputs must be all NumPy arrays or all Arrow arrays",
    ))
}

fn tours_output_numpy<'py>(py: Python<'py>, result: ToursResult) -> ToursPyResult<'py> {
    (
        numpy_u64_output(py, result.uid_codes),
        numpy_i64_output(py, result.started_at_us),
        numpy_i64_output(py, result.finished_at_us),
        numpy_i64_output(py, result.location_ids),
        result.journey_offsets.into_pyarray(py),
        numpy_i64_output(py, result.journey_trip_ids),
    )
}

fn tours_output_arrow<'py>(py: Python<'py>, result: ToursResult) -> PyResult<ToursPyResult<'py>> {
    Ok((
        arrow_u64_output(py, result.uid_codes)?,
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
    uid_codes: &Bound<'py, PyAny>,
    trip_ids: &Bound<'py, PyAny>,
    started_at_us: &Bound<'py, PyAny>,
    finished_at_us: &Bound<'py, PyAny>,
    origin_location_ids: &Bound<'py, PyAny>,
    destination_location_ids: &Bound<'py, PyAny>,
) -> PyResult<ToursPyResult<'py>> {
    if let (
        Ok(uid_codes),
        Ok(trip_ids),
        Ok(started_at_us),
        Ok(finished_at_us),
        Ok(origin_location_ids),
        Ok(destination_location_ids),
    ) = (
        uid_codes.extract::<PyReadonlyArray1<u64>>(),
        trip_ids.extract::<PyReadonlyArray1<i64>>(),
        started_at_us.extract::<PyReadonlyArray1<i64>>(),
        finished_at_us.extract::<PyReadonlyArray1<i64>>(),
        origin_location_ids.extract::<PyReadonlyArray1<i64>>(),
        destination_location_ids.extract::<PyReadonlyArray1<i64>>(),
    ) {
        let uid_codes = uid_codes.as_slice()?;
        let trip_ids = trip_ids.as_slice()?;
        let started_at_us = started_at_us.as_slice()?;
        let finished_at_us = finished_at_us.as_slice()?;
        let origin_location_ids = origin_location_ids.as_slice()?;
        let destination_location_ids = destination_location_ids.as_slice()?;
        let result = py
            .detach(|| {
                tours_from_trips_impl(
                    uid_codes,
                    trip_ids,
                    started_at_us,
                    finished_at_us,
                    origin_location_ids,
                    destination_location_ids,
                )
            })
            .map_err(PyValueError::new_err)?;
        return Ok(tours_output_numpy(py, result));
    }

    if is_arrow_array(uid_codes)?
        && is_arrow_array(trip_ids)?
        && is_arrow_array(started_at_us)?
        && is_arrow_array(finished_at_us)?
        && is_arrow_array(origin_location_ids)?
        && is_arrow_array(destination_location_ids)?
    {
        let uid_codes = as_u64_array(uid_codes.extract::<PyArray>()?, "uid_codes")?;
        let trip_ids = as_i64_array(trip_ids.extract::<PyArray>()?, "trip_ids")?;
        let started_at_us = as_i64_array(started_at_us.extract::<PyArray>()?, "started_at_us")?;
        let finished_at_us = as_i64_array(finished_at_us.extract::<PyArray>()?, "finished_at_us")?;
        let origin_location_ids = as_i64_array(
            origin_location_ids.extract::<PyArray>()?,
            "origin_location_ids",
        )?;
        let destination_location_ids = as_i64_array(
            destination_location_ids.extract::<PyArray>()?,
            "destination_location_ids",
        )?;
        let result = py
            .detach(|| {
                tours_from_trips_impl(
                    arrow_u64_values(&uid_codes),
                    arrow_i64_values(&trip_ids),
                    arrow_i64_values(&started_at_us),
                    arrow_i64_values(&finished_at_us),
                    arrow_i64_values(&origin_location_ids),
                    arrow_i64_values(&destination_location_ids),
                )
            })
            .map_err(PyValueError::new_err)?;
        return tours_output_arrow(py, result);
    }

    Err(PyTypeError::new_err(
        "tour inputs must be all NumPy arrays or all Arrow arrays",
    ))
}
