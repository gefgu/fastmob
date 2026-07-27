use std::str::FromStr;

use fastmob_core::preprocessing::simplify::{
    SimplifyConfig as CoreSimplifyConfig, SimplifyMethod, simplify_trajectory_impl,
    simplify_trajectory_indexed_impl,
};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::adapters::trajectory::run_indexed_timed_coordinate_arrow;
use crate::utils::{arrow_values, as_f64_array, bool_results_into_arrow};

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

fn arrow_bool_output(py: Python<'_>, values: Vec<bool>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, bool_results_into_arrow(values))?.into_any())
}

#[pyfunction]
pub fn simplify_trajectory_sorted(
    py: Python<'_>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    ranges: Vec<(usize, usize)>,
    config: PySimplifyConfig,
) -> PyResult<Py<PyAny>> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;
    let keep_mask = py.detach(|| {
        simplify_trajectory_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            arrow_values(&timestamps_s),
            &ranges,
            &config.0,
        )
    });
    arrow_bool_output(py, keep_mask)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn simplify_trajectory_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    sorted_indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    config: PySimplifyConfig,
) -> PyResult<Py<PyAny>> {
    let keep_mask = run_indexed_timed_coordinate_arrow(
        py,
        latitudes,
        longitudes,
        timestamps_s,
        sorted_indices,
        ends,
        |view| {
            simplify_trajectory_indexed_impl(
                view.coordinates.latitudes,
                view.coordinates.longitudes,
                view.coordinates.times,
                view.indices,
                view.ends,
                view.valid_rows,
                &config.0,
            )
        },
    )?;
    arrow_bool_output(py, keep_mask)
}
