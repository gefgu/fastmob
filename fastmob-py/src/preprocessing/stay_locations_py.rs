use fastmob_core::preprocessing::stay_locations::{
    detect_stay_locations_batch_impl, detect_stay_locations_batch_indexed_impl,
};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, f64_results_into_arrow,
    u64_results_into_arrow, validate_indexed_ends,
};

type StayLocationsBatchResult<'py> = (Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>);

fn is_arrow_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.hasattr("__arrow_c_array__")
}

fn numpy_f64_output(py: Python<'_>, values: Vec<f64>) -> Py<PyAny> {
    values.into_pyarray(py).into_any().unbind()
}

fn numpy_usize_output(py: Python<'_>, values: Vec<usize>) -> Py<PyAny> {
    values.into_pyarray(py).into_any().unbind()
}

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

fn arrow_usize_output(py: Python<'_>, values: Vec<usize>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(
        py,
        u64_results_into_arrow(values.into_iter().map(|i| i as u64).collect()),
    )?
    .into_any())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn detect_stay_locations_batch<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    timestamps_s: &Bound<'py, PyAny>,
    ranges: Vec<(usize, usize)>,
    stop_radius_km: f64,
    minutes_for_a_stop: f64,
    no_data_for_minutes: f64,
    min_speed_kmh: f64,
) -> PyResult<StayLocationsBatchResult<'py>> {
    if let (Ok(latitudes), Ok(longitudes), Ok(timestamps_s)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
        timestamps_s.extract::<PyReadonlyArray1<f64>>(),
    ) {
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
        return Ok((
            numpy_f64_output(py, out_lats),
            numpy_f64_output(py, out_lngs),
            numpy_f64_output(py, entry_times),
            numpy_f64_output(py, leaving_times),
            numpy_usize_output(py, user_range_idx),
        ));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? && is_arrow_array(timestamps_s)? {
        let latitudes = as_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes = as_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        let timestamps_s = as_f64_array(timestamps_s.extract::<ArrowPyArray>()?, "timestamps_s")?;
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
        return Ok((
            arrow_f64_output(py, out_lats)?,
            arrow_f64_output(py, out_lngs)?,
            arrow_f64_output(py, entry_times)?,
            arrow_f64_output(py, leaving_times)?,
            arrow_usize_output(py, user_range_idx)?,
        ));
    }

    Err(PyTypeError::new_err(
        "latitudes, longitudes, and timestamps_s must all be NumPy arrays or all be Arrow arrays",
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn detect_stay_locations_batch_indexed<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    timestamps_s: &Bound<'py, PyAny>,
    sorted_indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    stop_radius_km: f64,
    minutes_for_a_stop: f64,
    no_data_for_minutes: f64,
    min_speed_kmh: f64,
) -> PyResult<StayLocationsBatchResult<'py>> {
    let sorted_indices = sorted_indices.as_slice()?;
    let ends = ends.as_slice()?;

    if let (Ok(latitudes), Ok(longitudes), Ok(timestamps_s)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
        timestamps_s.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let lats = latitudes.as_slice()?;
        let lngs = longitudes.as_slice()?;
        let times = timestamps_s.as_slice()?;
        validate_indexed_ends(lats.len(), sorted_indices, ends)?;
        let (out_lats, out_lngs, entry_times, leaving_times, user_range_idx) = py.detach(|| {
            detect_stay_locations_batch_indexed_impl(
                lats,
                lngs,
                times,
                sorted_indices,
                ends,
                None,
                stop_radius_km,
                minutes_for_a_stop,
                no_data_for_minutes,
                min_speed_kmh,
            )
        });
        return Ok((
            numpy_f64_output(py, out_lats),
            numpy_f64_output(py, out_lngs),
            numpy_f64_output(py, entry_times),
            numpy_f64_output(py, leaving_times),
            numpy_usize_output(py, user_range_idx),
        ));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? && is_arrow_array(timestamps_s)? {
        let latitudes = as_nullable_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes =
            as_nullable_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        let timestamps_s =
            as_nullable_f64_array(timestamps_s.extract::<ArrowPyArray>()?, "timestamps_s")?;
        validate_indexed_ends(latitudes.len(), sorted_indices, ends)?;
        let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes, &timestamps_s]);
        let (out_lats, out_lngs, entry_times, leaving_times, user_range_idx) = py.detach(|| {
            detect_stay_locations_batch_indexed_impl(
                arrow_values(&latitudes),
                arrow_values(&longitudes),
                arrow_values(&timestamps_s),
                sorted_indices,
                ends,
                valid_rows.as_deref(),
                stop_radius_km,
                minutes_for_a_stop,
                no_data_for_minutes,
                min_speed_kmh,
            )
        });
        return Ok((
            arrow_f64_output(py, out_lats)?,
            arrow_f64_output(py, out_lngs)?,
            arrow_f64_output(py, entry_times)?,
            arrow_f64_output(py, leaving_times)?,
            arrow_usize_output(py, user_range_idx)?,
        ));
    }

    Err(PyTypeError::new_err(
        "latitudes, longitudes, and timestamps_s must all be NumPy arrays or all be Arrow arrays",
    ))
}
