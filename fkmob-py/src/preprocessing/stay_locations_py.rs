use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;
use fkmob_core::preprocessing::stay_locations::{
    detect_stay_locations_batch_impl, detect_stay_locations_batch_indexed_impl,
};

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, f64_results_into_arrow,
    u64_results_into_arrow, validate_indexed_ends,
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
pub fn detect_stay_locations_batch_numpy<'py>(
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
    let lats = latitudes.as_slice()?;
    let lngs = longitudes.as_slice()?;
    let times = timestamps_s.as_slice()?;
    let (out_lats, out_lngs, entry_times, leaving_times, user_range_idx) = py.detach(|| {
        detect_stay_locations_batch_impl(
            lats,
            lngs,
            times,
            &ranges,
            stop_radius_km,
            minutes_for_a_stop,
            no_data_for_minutes,
            min_speed_kmh,
        )
    });
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
pub fn detect_stay_locations_batch_arrow(
    py: Python<'_>,
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
    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    let times = arrow_values(&timestamps_s);
    let (out_lats, out_lngs, entry_times, leaving_times, user_range_idx) = py.detach(|| {
        detect_stay_locations_batch_impl(
            lats,
            lngs,
            times,
            &ranges,
            stop_radius_km,
            minutes_for_a_stop,
            no_data_for_minutes,
            min_speed_kmh,
        )
    });
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
pub fn detect_stay_locations_batch_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    timestamps_s: PyReadonlyArray1<'py, f64>,
    sorted_indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    stop_radius_km: f64,
    minutes_for_a_stop: f64,
    no_data_for_minutes: f64,
    min_speed_kmh: f64,
) -> PyResult<StayLocationsBatchNumpyResult<'py>> {
    let lats = latitudes.as_slice()?;
    let lngs = longitudes.as_slice()?;
    let times = timestamps_s.as_slice()?;
    let idxs = sorted_indices.as_slice()?;
    let ends = ends.as_slice()?;
    validate_indexed_ends(lats.len(), idxs, ends)?;
    let (out_lats, out_lngs, entry_times, leaving_times, user_range_idx) = py.detach(|| {
        detect_stay_locations_batch_indexed_impl(
            lats,
            lngs,
            times,
            idxs,
            ends,
            None,
            stop_radius_km,
            minutes_for_a_stop,
            no_data_for_minutes,
            min_speed_kmh,
        )
    });
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
pub fn detect_stay_locations_batch_indexed_arrow(
    py: Python<'_>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    sorted_indices: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
    stop_radius_km: f64,
    minutes_for_a_stop: f64,
    no_data_for_minutes: f64,
    min_speed_kmh: f64,
) -> PyResult<StayLocationsBatchArrowResult> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let timestamps_s = as_nullable_f64_array(timestamps_s, "timestamps_s")?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes, &timestamps_s]);
    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    let times = arrow_values(&timestamps_s);
    let idxs = sorted_indices.as_slice()?;
    let ends = ends.as_slice()?;
    validate_indexed_ends(lats.len(), idxs, ends)?;
    let (out_lats, out_lngs, entry_times, leaving_times, user_range_idx) = py.detach(|| {
        detect_stay_locations_batch_indexed_impl(
            lats,
            lngs,
            times,
            idxs,
            ends,
            valid_rows.as_deref(),
            stop_radius_km,
            minutes_for_a_stop,
            no_data_for_minutes,
            min_speed_kmh,
        )
    });
    Ok((
        f64_results_into_arrow(out_lats),
        f64_results_into_arrow(out_lngs),
        f64_results_into_arrow(entry_times),
        f64_results_into_arrow(leaving_times),
        u64_results_into_arrow(user_range_idx.into_iter().map(|i| i as u64).collect()),
    ))
}
