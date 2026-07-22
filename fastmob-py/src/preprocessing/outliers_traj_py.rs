use std::str::FromStr;

use fastmob_core::preprocessing::outliers::{
    OutlierConfig as CoreOutlierConfig, OutlierMethod, outlier_trajectory_impl,
    outlier_trajectory_indexed_impl,
};
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, bool_results_into_arrow,
    validate_indexed_ends,
};

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

#[pyfunction]
pub fn outlier_trajectory_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    timestamps_s: PyReadonlyArray1<'py, f64>,
    ranges: Vec<(usize, usize)>,
    config: PyOutlierConfig,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let lats = latitudes.as_slice()?;
    let lngs = longitudes.as_slice()?;
    let times = timestamps_s.as_slice()?;

    let keep_mask = py.detach(|| outlier_trajectory_impl(lats, lngs, times, &ranges, &config.0));

    Ok(PyArray1::from_vec(py, keep_mask))
}

#[pyfunction]
pub fn outlier_trajectory_arrow(
    py: Python<'_>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    ranges: Vec<(usize, usize)>,
    config: PyOutlierConfig,
) -> PyResult<ArrowPyArray> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;

    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    let times = arrow_values(&timestamps_s);

    let keep_mask = py.detach(|| outlier_trajectory_impl(lats, lngs, times, &ranges, &config.0));

    Ok(bool_results_into_arrow(keep_mask))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn outlier_trajectory_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    timestamps_s: PyReadonlyArray1<'py, f64>,
    sorted_indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    config: PyOutlierConfig,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let lats = latitudes.as_slice()?;
    let lngs = longitudes.as_slice()?;
    let times = timestamps_s.as_slice()?;
    let indices = sorted_indices.as_slice()?;
    let ends = ends.as_slice()?;
    validate_indexed_ends(lats.len(), indices, ends)?;

    let keep_mask = py.detach(|| {
        outlier_trajectory_indexed_impl(lats, lngs, times, indices, ends, None, &config.0)
    });

    Ok(PyArray1::from_vec(py, keep_mask))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn outlier_trajectory_indexed_arrow(
    py: Python<'_>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    timestamps_s: ArrowPyArray,
    sorted_indices: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
    config: PyOutlierConfig,
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
        outlier_trajectory_indexed_impl(
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
