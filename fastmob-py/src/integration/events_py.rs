use fastmob_core::integration::events::{SortedReferenceEvents, nearest_event_within_window};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

/// Batch binding for `nearest_event_within_window`: for each trajectory
/// point, finds the nearest (Haversine) reference event whose timestamp
/// lies within `[time - window_seconds, time + window_seconds]`. The
/// Python wrapper is responsible for sorting `ref_*` by time ascending and
/// supplying `ref_original_index` (the pre-sort row position) so it can
/// gather id/category columns after the fact.
#[pyfunction]
#[pyo3(name = "nearest_event_within_window")]
#[allow(clippy::too_many_arguments)]
pub fn nearest_event_within_window_numpy<'py>(
    py: Python<'py>,
    query_lat: PyReadonlyArray1<'py, f64>,
    query_lng: PyReadonlyArray1<'py, f64>,
    query_time: PyReadonlyArray1<'py, i64>,
    ref_lat: PyReadonlyArray1<'py, f64>,
    ref_lng: PyReadonlyArray1<'py, f64>,
    ref_time_sorted: PyReadonlyArray1<'py, i64>,
    ref_original_index: PyReadonlyArray1<'py, i64>,
    window_seconds: i64,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<f64>>)> {
    let query_lat = query_lat.as_slice()?;
    let query_lng = query_lng.as_slice()?;
    let query_time = query_time.as_slice()?;
    let ref_lat = ref_lat.as_slice()?;
    let ref_lng = ref_lng.as_slice()?;
    let ref_time_sorted = ref_time_sorted.as_slice()?;
    let ref_original_index: Vec<usize> =
        ref_original_index.as_slice()?.iter().map(|&x| x as usize).collect();

    let reference = SortedReferenceEvents {
        lat: ref_lat,
        lng: ref_lng,
        time_sorted: ref_time_sorted,
        original_index: &ref_original_index,
    };
    let (idx, dist) = py.detach(|| nearest_event_within_window(query_lat, query_lng, query_time, &reference, window_seconds));
    Ok((idx.into_pyarray(py), dist.into_pyarray(py)))
}
