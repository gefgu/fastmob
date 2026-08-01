use std::str::FromStr;

use fastmob_core::trajectory::smooth::{
    smooth_trajectory_indexed_impl, smooth_trajectory_presorted_impl,
    SmoothConfig as CoreSmoothConfig, SmoothMethod,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::adapters::trajectory::{
    run_indexed_timed_coordinate_arrow, run_presorted_timed_coordinate_arrow,
};
use crate::utils::f64_results_into_arrow;

#[pyclass(name = "SmoothConfig", from_py_object)]
#[derive(Clone, Copy)]
pub struct PySmoothConfig(pub CoreSmoothConfig);

#[pymethods]
impl PySmoothConfig {
    #[new]
    #[pyo3(signature = (method="kalman_cv", process_noise_std_km=0.05, measurement_noise_std_km=0.1))]
    fn new(
        method: &str,
        process_noise_std_km: f64,
        measurement_noise_std_km: f64,
    ) -> PyResult<Self> {
        let method = SmoothMethod::from_str(method).map_err(PyValueError::new_err)?;
        Ok(PySmoothConfig(CoreSmoothConfig::new(
            method,
            process_noise_std_km,
            measurement_noise_std_km,
        )))
    }
}

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

type SmoothOutput = (Py<PyAny>, Py<PyAny>);

#[pyfunction]
pub fn smooth_trajectory_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    sorted_indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
    config: PySmoothConfig,
) -> PyResult<SmoothOutput> {
    let (out_lats, out_lngs) = run_indexed_timed_coordinate_arrow(
        py,
        latitudes,
        longitudes,
        timestamps_s,
        sorted_indices,
        ends,
        |view| {
            smooth_trajectory_indexed_impl(
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
    ))
}

#[pyfunction]
pub fn smooth_trajectory_presorted<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
    config: PySmoothConfig,
) -> PyResult<SmoothOutput> {
    let (out_lats, out_lngs) = run_presorted_timed_coordinate_arrow(
        py,
        latitudes,
        longitudes,
        timestamps_s,
        ends,
        |view, ends| {
            smooth_trajectory_presorted_impl(
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
    ))
}
