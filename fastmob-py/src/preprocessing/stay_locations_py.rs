use fastmob_core::preprocessing::stay_locations::{
    detect_stay_locations_batch_impl, detect_stay_locations_batch_indexed_impl,
};
use numpy::{IntoPyArray, PyArray1};
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::adapters::trajectory::run_indexed_timed_coordinate_arrow;
use crate::utils::{arrow_values, as_f64_array, f64_results_into_arrow};

type StayLocationsBatchResult<'py> = (
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Py<PyAny>,
    Bound<'py, PyArray1<usize>>,
);

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn detect_stay_locations_batch<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    ranges: Vec<(usize, usize)>,
    stop_radius_km: f64,
    minutes_for_a_stop: f64,
    no_data_for_minutes: f64,
    min_speed_kmh: f64,
) -> PyResult<StayLocationsBatchResult<'py>> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;
    let (out_lats, out_lngs, entry_times, leaving_times, user_range_idx) = py.detach(|| {
        detect_stay_locations_batch_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            arrow_values(&timestamps_s),
            &ranges,
            stop_radius_km,
            minutes_for_a_stop,
            no_data_for_minutes,
            min_speed_kmh,
        )
    });
    Ok((
        arrow_f64_output(py, out_lats)?,
        arrow_f64_output(py, out_lngs)?,
        arrow_f64_output(py, entry_times)?,
        arrow_f64_output(py, leaving_times)?,
        user_range_idx.into_pyarray(py),
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn detect_stay_locations_batch_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    sorted_indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
    stop_radius_km: f64,
    minutes_for_a_stop: f64,
    no_data_for_minutes: f64,
    min_speed_kmh: f64,
) -> PyResult<StayLocationsBatchResult<'py>> {
    let (out_lats, out_lngs, entry_times, leaving_times, user_range_idx) =
        run_indexed_timed_coordinate_arrow(
            py,
            latitudes,
            longitudes,
            timestamps_s,
            sorted_indices,
            ends,
            |view| {
                detect_stay_locations_batch_indexed_impl(
                    view.coordinates.latitudes,
                    view.coordinates.longitudes,
                    view.coordinates.times,
                    view.indices,
                    view.ends,
                    view.valid_rows,
                    stop_radius_km,
                    minutes_for_a_stop,
                    no_data_for_minutes,
                    min_speed_kmh,
                )
            },
        )?;
    Ok((
        arrow_f64_output(py, out_lats)?,
        arrow_f64_output(py, out_lngs)?,
        arrow_f64_output(py, entry_times)?,
        arrow_f64_output(py, leaving_times)?,
        user_range_idx.into_pyarray(py),
    ))
}
