use std::str::FromStr;

use fastmob_core::preprocessing::outliers::{
    OutlierConfig as CoreOutlierConfig, OutlierMethod, outlier_trajectory_impl,
    outlier_trajectory_indexed_impl,
};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::adapters::trajectory::run_indexed_timed_coordinate_arrow;
use crate::utils::{arrow_values, as_f64_array, bool_results_into_arrow};

#[pyclass(name = "OutlierConfig", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyOutlierConfig(pub CoreOutlierConfig);

#[pymethods]
impl PyOutlierConfig {
    #[new]
    #[pyo3(signature = (method="hampel", window_size=5, n_sigma=3.0, max_speed_kmh=100.0, min_seg_size=1))]
    fn new(
        method: &str,
        window_size: usize,
        n_sigma: f64,
        max_speed_kmh: f64,
        min_seg_size: usize,
    ) -> PyResult<Self> {
        let method = OutlierMethod::from_str(method).map_err(PyValueError::new_err)?;
        Ok(PyOutlierConfig(CoreOutlierConfig::new(
            method,
            window_size,
            n_sigma,
            max_speed_kmh,
            min_seg_size,
        )))
    }

    #[getter]
    fn window_size(&self) -> usize {
        self.0.window_size
    }
    #[setter]
    fn set_window_size(&mut self, v: usize) {
        self.0.window_size = v;
    }
    #[getter]
    fn n_sigma(&self) -> f64 {
        self.0.n_sigma
    }
    #[setter]
    fn set_n_sigma(&mut self, v: f64) {
        self.0.n_sigma = v;
    }
    #[getter]
    fn max_speed_kmh(&self) -> f64 {
        self.0.max_speed_kmh
    }
    #[setter]
    fn set_max_speed_kmh(&mut self, v: f64) {
        self.0.max_speed_kmh = v;
    }
    #[getter]
    fn min_seg_size(&self) -> usize {
        self.0.min_seg_size
    }
    #[setter]
    fn set_min_seg_size(&mut self, v: usize) {
        self.0.min_seg_size = v;
    }
}

fn arrow_bool_output(py: Python<'_>, values: Vec<bool>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, bool_results_into_arrow(values))?.into_any())
}

#[pyfunction]
pub fn outlier_trajectory_sorted(
    py: Python<'_>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    ranges: Vec<(usize, usize)>,
    config: PyOutlierConfig,
) -> PyResult<Py<PyAny>> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;
    let keep_mask = py.detach(|| {
        outlier_trajectory_impl(
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
pub fn outlier_trajectory_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    sorted_indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    config: PyOutlierConfig,
) -> PyResult<Py<PyAny>> {
    let keep_mask = run_indexed_timed_coordinate_arrow(
        py,
        latitudes,
        longitudes,
        timestamps_s,
        sorted_indices,
        ends,
        |view| {
            outlier_trajectory_indexed_impl(
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
