use std::str::FromStr;

use fastmob_core::preprocessing::simplify::{
    SimplifyConfig as CoreSimplifyConfig, SimplifyMethod, simplify_trajectory_impl,
    simplify_trajectory_indexed_impl,
};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, bool_results_into_arrow,
    validate_indexed_ends,
};

#[pyclass(name = "SimplifyConfig", from_py_object)]
#[derive(Clone, Copy)]
pub struct PySimplifyConfig(pub CoreSimplifyConfig);

#[pymethods]
impl PySimplifyConfig {
    #[new]
    #[pyo3(signature = (method="douglas_peucker", epsilon_km=0.001, min_distance_km=0.01, min_time_delta_s=60.0))]
    fn new(
        method: &str,
        epsilon_km: f64,
        min_distance_km: f64,
        min_time_delta_s: f64,
    ) -> PyResult<Self> {
        let method = SimplifyMethod::from_str(method).map_err(PyValueError::new_err)?;
        Ok(PySimplifyConfig(CoreSimplifyConfig::new(
            method,
            epsilon_km,
            min_distance_km,
            min_time_delta_s,
        )))
    }

    #[getter]
    fn epsilon_km(&self) -> f64 {
        self.0.epsilon_km
    }
    #[setter]
    fn set_epsilon_km(&mut self, v: f64) {
        self.0.epsilon_km = v;
    }
    #[getter]
    fn min_distance_km(&self) -> f64 {
        self.0.min_distance_km
    }
    #[setter]
    fn set_min_distance_km(&mut self, v: f64) {
        self.0.min_distance_km = v;
    }
    #[getter]
    fn min_time_delta_s(&self) -> f64 {
        self.0.min_time_delta_s
    }
    #[setter]
    fn set_min_time_delta_s(&mut self, v: f64) {
        self.0.min_time_delta_s = v;
    }
}

#[pyfunction]
pub fn simplify_trajectory_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    timestamps_s: PyReadonlyArray1<'py, f64>,
    ranges: Vec<(usize, usize)>,
    config: PySimplifyConfig,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let lats = latitudes.as_slice()?;
    let lngs = longitudes.as_slice()?;
    let times = timestamps_s.as_slice()?;

    let keep_mask = py.detach(|| simplify_trajectory_impl(lats, lngs, times, &ranges, &config.0));

    Ok(PyArray1::from_vec(py, keep_mask))
}

#[pyfunction]
pub fn simplify_trajectory_arrow(
    py: Python<'_>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    ranges: Vec<(usize, usize)>,
    config: PySimplifyConfig,
) -> PyResult<ArrowPyArray> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;

    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    let times = arrow_values(&timestamps_s);

    let keep_mask = py.detach(|| simplify_trajectory_impl(lats, lngs, times, &ranges, &config.0));

    Ok(bool_results_into_arrow(keep_mask))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn simplify_trajectory_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    timestamps_s: PyReadonlyArray1<'py, f64>,
    sorted_indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    config: PySimplifyConfig,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let lats = latitudes.as_slice()?;
    let lngs = longitudes.as_slice()?;
    let times = timestamps_s.as_slice()?;
    let indices = sorted_indices.as_slice()?;
    let ends = ends.as_slice()?;
    validate_indexed_ends(lats.len(), indices, ends)?;

    let keep_mask = py.detach(|| {
        simplify_trajectory_indexed_impl(lats, lngs, times, indices, ends, None, &config.0)
    });

    Ok(PyArray1::from_vec(py, keep_mask))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn simplify_trajectory_indexed_arrow(
    py: Python<'_>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    sorted_indices: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
    config: PySimplifyConfig,
) -> PyResult<ArrowPyArray> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let timestamps_s = as_nullable_f64_array(timestamps_s, "timestamps_s")?;

    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    let times = arrow_values(&timestamps_s);
    let indices = sorted_indices.as_slice()?;
    let ends = ends.as_slice()?;
    validate_indexed_ends(lats.len(), indices, ends)?;

    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes, &timestamps_s]);

    let keep_mask = py.detach(|| {
        simplify_trajectory_indexed_impl(
            lats,
            lngs,
            times,
            indices,
            ends,
            valid_rows.as_deref(),
            &config.0,
        )
    });

    Ok(bool_results_into_arrow(keep_mask))
}
