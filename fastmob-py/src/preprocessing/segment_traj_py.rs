use std::str::FromStr;

use fastmob_core::preprocessing::segment::{
    segment_trajectory_impl, segment_trajectory_indexed_impl, SegmentConfig as CoreSegmentConfig,
    SegmentMethod,
};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::adapters::trajectory::run_indexed_timed_coordinate_arrow;
use crate::utils::{arrow_values, as_f64_array, u32_results_into_arrow};

#[pyclass(name = "SegmentConfig", from_py_object)]
#[derive(Clone, Copy)]
pub struct PySegmentConfig(pub CoreSegmentConfig);

#[pymethods]
impl PySegmentConfig {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        method="value_change",
        min_angle_deg=45.0,
        angle_min_speed_kmh=0.0,
        gap_s=3600.0,
        speed_min_kmh=0.0,
        speed_max_kmh=f64::INFINITY,
        duration_s=300.0,
        stop_radius_km=0.2,
        stop_minutes_for_a_stop=20.0,
        stop_no_data_for_minutes=1e12,
        stop_min_speed_kmh=f64::INFINITY,
    ))]
    fn new(
        method: &str,
        min_angle_deg: f64,
        angle_min_speed_kmh: f64,
        gap_s: f64,
        speed_min_kmh: f64,
        speed_max_kmh: f64,
        duration_s: f64,
        stop_radius_km: f64,
        stop_minutes_for_a_stop: f64,
        stop_no_data_for_minutes: f64,
        stop_min_speed_kmh: f64,
    ) -> PyResult<Self> {
        let method = SegmentMethod::from_str(method).map_err(PyValueError::new_err)?;
        Ok(PySegmentConfig(CoreSegmentConfig::new(
            method,
            min_angle_deg,
            angle_min_speed_kmh,
            gap_s,
            speed_min_kmh,
            speed_max_kmh,
            duration_s,
            stop_radius_km,
            stop_minutes_for_a_stop,
            stop_no_data_for_minutes,
            stop_min_speed_kmh,
        )))
    }

    #[getter]
    fn min_angle_deg(&self) -> f64 {
        self.0.min_angle_deg
    }
    #[setter]
    fn set_min_angle_deg(&mut self, v: f64) {
        self.0.min_angle_deg = v;
    }
    #[getter]
    fn angle_min_speed_kmh(&self) -> f64 {
        self.0.angle_min_speed_kmh
    }
    #[setter]
    fn set_angle_min_speed_kmh(&mut self, v: f64) {
        self.0.angle_min_speed_kmh = v;
    }
    #[getter]
    fn gap_s(&self) -> f64 {
        self.0.gap_s
    }
    #[setter]
    fn set_gap_s(&mut self, v: f64) {
        self.0.gap_s = v;
    }
    #[getter]
    fn speed_min_kmh(&self) -> f64 {
        self.0.speed_min_kmh
    }
    #[setter]
    fn set_speed_min_kmh(&mut self, v: f64) {
        self.0.speed_min_kmh = v;
    }
    #[getter]
    fn speed_max_kmh(&self) -> f64 {
        self.0.speed_max_kmh
    }
    #[setter]
    fn set_speed_max_kmh(&mut self, v: f64) {
        self.0.speed_max_kmh = v;
    }
    #[getter]
    fn duration_s(&self) -> f64 {
        self.0.duration_s
    }
    #[setter]
    fn set_duration_s(&mut self, v: f64) {
        self.0.duration_s = v;
    }
    #[getter]
    fn stop_radius_km(&self) -> f64 {
        self.0.stop_radius_km
    }
    #[setter]
    fn set_stop_radius_km(&mut self, v: f64) {
        self.0.stop_radius_km = v;
    }
    #[getter]
    fn stop_minutes_for_a_stop(&self) -> f64 {
        self.0.stop_minutes_for_a_stop
    }
    #[setter]
    fn set_stop_minutes_for_a_stop(&mut self, v: f64) {
        self.0.stop_minutes_for_a_stop = v;
    }
    #[getter]
    fn stop_no_data_for_minutes(&self) -> f64 {
        self.0.stop_no_data_for_minutes
    }
    #[setter]
    fn set_stop_no_data_for_minutes(&mut self, v: f64) {
        self.0.stop_no_data_for_minutes = v;
    }
    #[getter]
    fn stop_min_speed_kmh(&self) -> f64 {
        self.0.stop_min_speed_kmh
    }
    #[setter]
    fn set_stop_min_speed_kmh(&mut self, v: f64) {
        self.0.stop_min_speed_kmh = v;
    }
}

fn arrow_u32_output(py: Python<'_>, values: Vec<u32>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, u32_results_into_arrow(values))?.into_any())
}

#[pyfunction]
#[pyo3(signature = (latitudes, longitudes, timestamps_s, ranges, config, bucket_ids=None))]
pub fn segment_trajectory_sorted<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    ranges: Vec<(usize, usize)>,
    config: PySegmentConfig,
    bucket_ids: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<Py<PyAny>> {
    let bucket_slice = bucket_ids.as_ref().map(|b| b.as_slice()).transpose()?;
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;
    let segment_ids = py.detach(|| {
        segment_trajectory_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            arrow_values(&timestamps_s),
            &ranges,
            bucket_slice,
            &config.0,
        )
    });
    arrow_u32_output(py, segment_ids)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (latitudes, longitudes, timestamps_s, sorted_indices, ends, config, bucket_ids=None))]
pub fn segment_trajectory_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    sorted_indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
    config: PySegmentConfig,
    bucket_ids: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<Py<PyAny>> {
    let bucket_slice = bucket_ids.as_ref().map(|b| b.as_slice()).transpose()?;
    let segment_ids = run_indexed_timed_coordinate_arrow(
        py,
        latitudes,
        longitudes,
        timestamps_s,
        sorted_indices,
        ends,
        |view| {
            segment_trajectory_indexed_impl(
                view.coordinates.latitudes,
                view.coordinates.longitudes,
                view.coordinates.times,
                view.indices,
                view.ends,
                view.valid_rows,
                bucket_slice,
                &config.0,
            )
        },
    )?;
    arrow_u32_output(py, segment_ids)
}
