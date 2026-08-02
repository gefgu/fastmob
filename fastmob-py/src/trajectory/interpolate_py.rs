use std::str::FromStr;

use fastmob_core::trajectory::interpolate::{
    interpolate_trajectory_indexed_impl, interpolate_trajectory_presorted_impl,
    InterpolationConfig as CoreInterpolationConfig, InterpolationMethod,
};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::adapters::trajectory::{
    run_indexed_timed_coordinate_arrow_ms, run_presorted_timed_coordinate_arrow_ms,
};
use crate::utils::{f64_results_into_arrow, i64_results_into_arrow};

#[pyclass(name = "InterpolationConfig", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyInterpolationConfig(pub CoreInterpolationConfig);

#[pymethods]
impl PyInterpolationConfig {
    #[new]
    #[pyo3(signature = (method="linear", sampling_rate_s=3600.0, max_speed_kmh=300.0, step_std_km=0.01, seed=0))]
    fn new(
        method: &str,
        sampling_rate_s: f64,
        max_speed_kmh: f64,
        step_std_km: f64,
        seed: u64,
    ) -> PyResult<Self> {
        let method = InterpolationMethod::from_str(method).map_err(PyValueError::new_err)?;
        Ok(PyInterpolationConfig(CoreInterpolationConfig::new(
            method,
            sampling_rate_s,
            max_speed_kmh,
            step_std_km,
            seed,
        )))
    }
}

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

/// Converts kernel-internal seconds (`f64`) back to milliseconds (`i64`) at the
/// FFI boundary, matching `fastmob.utils._common._extract_timestamps`'s default unit.
fn arrow_ms_output(py: Python<'_>, seconds: Vec<f64>) -> PyResult<Py<PyAny>> {
    let ms: Vec<i64> = seconds.into_iter().map(|s| (s * 1000.0).round() as i64).collect();
    Ok(Py::new(py, i64_results_into_arrow(ms))?.into_any())
}

type InterpolateOutput<'py> = (Py<PyAny>, Py<PyAny>, Py<PyAny>, Bound<'py, PyArray1<usize>>);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn interpolate_trajectory_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_ms: ArrowPyArray,
    sorted_indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
    config: PyInterpolationConfig,
) -> PyResult<InterpolateOutput<'py>> {
    let (out_lats, out_lngs, out_times, out_user_indices) = run_indexed_timed_coordinate_arrow_ms(
        py,
        latitudes,
        longitudes,
        timestamps_ms,
        sorted_indices,
        ends,
        |view| {
            interpolate_trajectory_indexed_impl(
                view.coordinates.latitudes,
                view.coordinates.longitudes,
                view.coordinates.times,
                view.indices,
                view.ends,
                view.valid_rows,
                &config.0,
            )
        },
    )?
    .map_err(PyValueError::new_err)?;
    Ok((
        arrow_f64_output(py, out_lats)?,
        arrow_f64_output(py, out_lngs)?,
        arrow_ms_output(py, out_times)?,
        out_user_indices.into_pyarray(py),
    ))
}

#[pyfunction]
pub fn interpolate_trajectory_presorted<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_ms: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
    config: PyInterpolationConfig,
) -> PyResult<InterpolateOutput<'py>> {
    let (out_lats, out_lngs, out_times, out_user_indices) = run_presorted_timed_coordinate_arrow_ms(
        py,
        latitudes,
        longitudes,
        timestamps_ms,
        ends,
        |view, ends| {
            interpolate_trajectory_presorted_impl(
                view.latitudes,
                view.longitudes,
                view.times,
                ends,
                &config.0,
            )
        },
    )?
    .map_err(PyValueError::new_err)?;
    Ok((
        arrow_f64_output(py, out_lats)?,
        arrow_f64_output(py, out_lngs)?,
        arrow_ms_output(py, out_times)?,
        out_user_indices.into_pyarray(py),
    ))
}
