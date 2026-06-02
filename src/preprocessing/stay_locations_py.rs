use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use super::stay_locations::{
    detect_stay_locations_batch_impl, detect_stay_locations_batch_indexed_impl,
};
use crate::utils::{
    arrow_values, as_f64_array, f64_results_into_arrow, ranges_from_starts_ends,
    u64_results_into_arrow,
};

type StayLocationsBatchNumpyResult<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<usize>>,
);
type StayLocationsBatchArrowResult = (
    ArrowPyArray,
    ArrowPyArray,
    ArrowPyArray,
    ArrowPyArray,
    ArrowPyArray,
);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn detect_stay_locations_batch_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    timestamps_s: PyReadonlyArray1<'py, f64>,
    ranges: Vec<(usize, usize)>,
    stop_radius_km: f64,
    minutes_for_a_stop: f64,
    no_data_for_minutes: f64,
    min_speed_kmh: f64,
) -> PyResult<StayLocationsBatchNumpyResult<'py>> {
    let (out_lats, out_lngs, entry_times, leaving_times, user_range_idx) =
        detect_stay_locations_batch_impl(
            latitudes.as_slice()?,
            longitudes.as_slice()?,
            timestamps_s.as_slice()?,
            &ranges,
            stop_radius_km,
            minutes_for_a_stop,
            no_data_for_minutes,
            min_speed_kmh,
        );
    Ok((
        PyArray1::from_vec(py, out_lats),
        PyArray1::from_vec(py, out_lngs),
        PyArray1::from_vec(py, entry_times),
        PyArray1::from_vec(py, leaving_times),
        PyArray1::from_vec(py, user_range_idx),
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn detect_stay_locations_batch_arrow(
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    ranges: Vec<(usize, usize)>,
    stop_radius_km: f64,
    minutes_for_a_stop: f64,
    no_data_for_minutes: f64,
    min_speed_kmh: f64,
) -> PyResult<StayLocationsBatchArrowResult> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;
    let (out_lats, out_lngs, entry_times, leaving_times, user_range_idx) =
        detect_stay_locations_batch_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            arrow_values(&timestamps_s),
            &ranges,
            stop_radius_km,
            minutes_for_a_stop,
            no_data_for_minutes,
            min_speed_kmh,
        );
    Ok((
        f64_results_into_arrow(out_lats),
        f64_results_into_arrow(out_lngs),
        f64_results_into_arrow(entry_times),
        f64_results_into_arrow(leaving_times),
        u64_results_into_arrow(user_range_idx.into_iter().map(|i| i as u64).collect()),
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn detect_stay_locations_batch_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    timestamps_s: PyReadonlyArray1<'py, f64>,
    sorted_indices: PyReadonlyArray1<'py, usize>,
    starts: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    stop_radius_km: f64,
    minutes_for_a_stop: f64,
    no_data_for_minutes: f64,
    min_speed_kmh: f64,
) -> PyResult<StayLocationsBatchNumpyResult<'py>> {
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    let (out_lats, out_lngs, entry_times, leaving_times, user_range_idx) =
        detect_stay_locations_batch_indexed_impl(
            latitudes.as_slice()?,
            longitudes.as_slice()?,
            timestamps_s.as_slice()?,
            sorted_indices.as_slice()?,
            &ranges,
            stop_radius_km,
            minutes_for_a_stop,
            no_data_for_minutes,
            min_speed_kmh,
        );
    Ok((
        PyArray1::from_vec(py, out_lats),
        PyArray1::from_vec(py, out_lngs),
        PyArray1::from_vec(py, entry_times),
        PyArray1::from_vec(py, leaving_times),
        PyArray1::from_vec(py, user_range_idx),
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn detect_stay_locations_batch_indexed_arrow(
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    sorted_indices: PyReadonlyArray1<usize>,
    starts: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
    stop_radius_km: f64,
    minutes_for_a_stop: f64,
    no_data_for_minutes: f64,
    min_speed_kmh: f64,
) -> PyResult<StayLocationsBatchArrowResult> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    let (out_lats, out_lngs, entry_times, leaving_times, user_range_idx) =
        detect_stay_locations_batch_indexed_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            arrow_values(&timestamps_s),
            sorted_indices.as_slice()?,
            &ranges,
            stop_radius_km,
            minutes_for_a_stop,
            no_data_for_minutes,
            min_speed_kmh,
        );
    Ok((
        f64_results_into_arrow(out_lats),
        f64_results_into_arrow(out_lngs),
        f64_results_into_arrow(entry_times),
        f64_results_into_arrow(leaving_times),
        u64_results_into_arrow(user_range_idx.into_iter().map(|i| i as u64).collect()),
    ))
}
