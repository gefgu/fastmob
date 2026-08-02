use fastmob_core::preprocessing::filter_traj::{
    filter_trajectory_impl, filter_trajectory_indexed_impl, FilterConfig as CoreFilterConfig,
};
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::adapters::trajectory::run_indexed_timed_coordinate_arrow_ms;
use crate::utils::{arrow_values, as_f64_array, bool_results_into_arrow};

#[pyclass(name = "FilterConfig", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyFilterConfig(pub CoreFilterConfig);

#[pymethods]
impl PyFilterConfig {
    #[new]
    #[pyo3(signature = (max_speed_kmh=500.0, include_loops=false, speed_kmh=5.0, max_loop=6, ratio_max=0.25))]
    fn new(
        max_speed_kmh: f64,
        include_loops: bool,
        speed_kmh: f64,
        max_loop: usize,
        ratio_max: f64,
    ) -> Self {
        PyFilterConfig(CoreFilterConfig::new(
            max_speed_kmh,
            include_loops,
            speed_kmh,
            max_loop,
            ratio_max,
        ))
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
    fn include_loops(&self) -> bool {
        self.0.include_loops
    }
    #[setter]
    fn set_include_loops(&mut self, v: bool) {
        self.0.include_loops = v;
    }
    #[getter]
    fn speed_kmh(&self) -> f64 {
        self.0.speed_kmh
    }
    #[setter]
    fn set_speed_kmh(&mut self, v: f64) {
        self.0.speed_kmh = v;
    }
    #[getter]
    fn max_loop(&self) -> usize {
        self.0.max_loop
    }
    #[setter]
    fn set_max_loop(&mut self, v: usize) {
        self.0.max_loop = v;
    }
    #[getter]
    fn ratio_max(&self) -> f64 {
        self.0.ratio_max
    }
    #[setter]
    fn set_ratio_max(&mut self, v: f64) {
        self.0.ratio_max = v;
    }
}

fn arrow_bool_output(py: Python<'_>, values: Vec<bool>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, bool_results_into_arrow(values))?.into_any())
}

#[pyfunction]
pub fn filter_trajectory(
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    timestamps_s: Vec<f64>,
    ranges: Vec<(usize, usize)>,
    config: PyFilterConfig,
) -> PyResult<Vec<bool>> {
    Ok(filter_trajectory_impl(
        &latitudes,
        &longitudes,
        &timestamps_s,
        &ranges,
        &config.0,
    ))
}

#[pyfunction]
pub fn filter_trajectory_sorted(
    py: Python<'_>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    ranges: Vec<(usize, usize)>,
    config: PyFilterConfig,
) -> PyResult<Py<PyAny>> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;
    let keep_mask = py.detach(|| {
        filter_trajectory_impl(
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
pub fn filter_trajectory_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_ms: ArrowPyArray,
    sorted_indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
    config: PyFilterConfig,
) -> PyResult<Py<PyAny>> {
    let keep_mask = run_indexed_timed_coordinate_arrow_ms(
        py,
        latitudes,
        longitudes,
        timestamps_ms,
        sorted_indices,
        ends,
        |view| {
            filter_trajectory_indexed_impl(
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
