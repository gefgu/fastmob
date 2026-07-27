use std::str::FromStr;

use fastmob_core::preprocessing::segment::{
    segment_trajectory_impl, segment_trajectory_indexed_impl, SegmentConfig as CoreSegmentConfig,
    SegmentMethod,
};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, u32_results_into_arrow,
    validate_indexed_ends,
};

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

fn is_arrow_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.hasattr("__arrow_c_array__")
}

fn numpy_u32_output(py: Python<'_>, values: Vec<u32>) -> Py<PyAny> {
    values.into_pyarray(py).into_any().unbind()
}

fn arrow_u32_output(py: Python<'_>, values: Vec<u32>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, u32_results_into_arrow(values))?.into_any())
}

#[pyfunction]
#[pyo3(signature = (latitudes, longitudes, timestamps_s, ranges, config, bucket_ids=None))]
pub fn segment_trajectory_sorted<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    timestamps_s: &Bound<'py, PyAny>,
    ranges: Vec<(usize, usize)>,
    config: PySegmentConfig,
    bucket_ids: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<Py<PyAny>> {
    let bucket_slice = bucket_ids.as_ref().map(|b| b.as_slice()).transpose()?;

    if let (Ok(latitudes), Ok(longitudes), Ok(timestamps_s)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
        timestamps_s.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let lats = latitudes.as_slice()?;
        let lngs = longitudes.as_slice()?;
        let times = timestamps_s.as_slice()?;
        let segment_ids = py.detach(|| {
            segment_trajectory_impl(lats, lngs, times, &ranges, bucket_slice, &config.0)
        });
        return Ok(numpy_u32_output(py, segment_ids));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? && is_arrow_array(timestamps_s)? {
        let latitudes = as_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes = as_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        let timestamps_s = as_f64_array(timestamps_s.extract::<ArrowPyArray>()?, "timestamps_s")?;
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
        return arrow_u32_output(py, segment_ids);
    }

    Err(PyTypeError::new_err(
        "latitudes, longitudes, and timestamps_s must all be NumPy arrays or all be Arrow arrays",
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (latitudes, longitudes, timestamps_s, sorted_indices, ends, config, bucket_ids=None))]
pub fn segment_trajectory_indexed<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    timestamps_s: &Bound<'py, PyAny>,
    sorted_indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    config: PySegmentConfig,
    bucket_ids: Option<PyReadonlyArray1<'py, i64>>,
) -> PyResult<Py<PyAny>> {
    let sorted_indices = sorted_indices.as_slice()?;
    let ends = ends.as_slice()?;
    let bucket_slice = bucket_ids.as_ref().map(|b| b.as_slice()).transpose()?;

    if let (Ok(latitudes), Ok(longitudes), Ok(timestamps_s)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
        timestamps_s.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let lats = latitudes.as_slice()?;
        let lngs = longitudes.as_slice()?;
        let times = timestamps_s.as_slice()?;
        validate_indexed_ends(lats.len(), sorted_indices, ends)?;
        let segment_ids = py.detach(|| {
            segment_trajectory_indexed_impl(
                lats,
                lngs,
                times,
                sorted_indices,
                ends,
                None,
                bucket_slice,
                &config.0,
            )
        });
        return Ok(numpy_u32_output(py, segment_ids));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? && is_arrow_array(timestamps_s)? {
        let latitudes = as_nullable_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes =
            as_nullable_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        let timestamps_s =
            as_nullable_f64_array(timestamps_s.extract::<ArrowPyArray>()?, "timestamps_s")?;
        validate_indexed_ends(latitudes.len(), sorted_indices, ends)?;
        let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes, &timestamps_s]);
        let segment_ids = py.detach(|| {
            segment_trajectory_indexed_impl(
                arrow_values(&latitudes),
                arrow_values(&longitudes),
                arrow_values(&timestamps_s),
                sorted_indices,
                ends,
                valid_rows.as_deref(),
                bucket_slice,
                &config.0,
            )
        });
        return arrow_u32_output(py, segment_ids);
    }

    Err(PyTypeError::new_err(
        "latitudes, longitudes, and timestamps_s must all be NumPy arrays or all be Arrow arrays",
    ))
}
