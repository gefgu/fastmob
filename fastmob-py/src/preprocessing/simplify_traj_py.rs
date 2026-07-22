use std::str::FromStr;

use fastmob_core::preprocessing::simplify::{
    SimplifyConfig as CoreSimplifyConfig, SimplifyMethod, simplify_trajectory_impl,
    simplify_trajectory_indexed_impl,
};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
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

fn is_arrow_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.hasattr("__arrow_c_array__")
}

fn numpy_bool_output(py: Python<'_>, values: Vec<bool>) -> Py<PyAny> {
    values.into_pyarray(py).into_any().unbind()
}

fn arrow_bool_output(py: Python<'_>, values: Vec<bool>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, bool_results_into_arrow(values))?.into_any())
}

#[pyfunction]
pub fn simplify_trajectory_sorted<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    timestamps_s: &Bound<'py, PyAny>,
    ranges: Vec<(usize, usize)>,
    config: PySimplifyConfig,
) -> PyResult<Py<PyAny>> {
    if let (Ok(latitudes), Ok(longitudes), Ok(timestamps_s)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
        timestamps_s.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let lats = latitudes.as_slice()?;
        let lngs = longitudes.as_slice()?;
        let times = timestamps_s.as_slice()?;
        let keep_mask =
            py.detach(|| simplify_trajectory_impl(lats, lngs, times, &ranges, &config.0));
        return Ok(numpy_bool_output(py, keep_mask));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? && is_arrow_array(timestamps_s)? {
        let latitudes = as_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes = as_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        let timestamps_s = as_f64_array(timestamps_s.extract::<ArrowPyArray>()?, "timestamps_s")?;
        let keep_mask = py.detach(|| {
            simplify_trajectory_impl(
                arrow_values(&latitudes),
                arrow_values(&longitudes),
                arrow_values(&timestamps_s),
                &ranges,
                &config.0,
            )
        });
        return arrow_bool_output(py, keep_mask);
    }

    Err(PyTypeError::new_err(
        "latitudes, longitudes, and timestamps_s must all be NumPy arrays or all be Arrow arrays",
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn simplify_trajectory_indexed<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    timestamps_s: &Bound<'py, PyAny>,
    sorted_indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    config: PySimplifyConfig,
) -> PyResult<Py<PyAny>> {
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
        let keep_mask = py.detach(|| {
            simplify_trajectory_indexed_impl(
                lats,
                lngs,
                times,
                sorted_indices,
                ends,
                None,
                &config.0,
            )
        });
        return Ok(numpy_bool_output(py, keep_mask));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? && is_arrow_array(timestamps_s)? {
        let latitudes = as_nullable_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes =
            as_nullable_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        let timestamps_s =
            as_nullable_f64_array(timestamps_s.extract::<ArrowPyArray>()?, "timestamps_s")?;
        validate_indexed_ends(latitudes.len(), sorted_indices, ends)?;
        let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes, &timestamps_s]);
        let keep_mask = py.detach(|| {
            simplify_trajectory_indexed_impl(
                arrow_values(&latitudes),
                arrow_values(&longitudes),
                arrow_values(&timestamps_s),
                sorted_indices,
                ends,
                valid_rows.as_deref(),
                &config.0,
            )
        });
        return arrow_bool_output(py, keep_mask);
    }

    Err(PyTypeError::new_err(
        "latitudes, longitudes, and timestamps_s must all be NumPy arrays or all be Arrow arrays",
    ))
}
