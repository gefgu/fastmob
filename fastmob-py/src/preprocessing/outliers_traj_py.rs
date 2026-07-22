use std::str::FromStr;

use fastmob_core::preprocessing::outliers::{
    OutlierConfig as CoreOutlierConfig, OutlierMethod, outlier_trajectory_impl,
    outlier_trajectory_indexed_impl,
};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
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
pub fn outlier_trajectory_sorted<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    timestamps_s: &Bound<'py, PyAny>,
    ranges: Vec<(usize, usize)>,
    config: PyOutlierConfig,
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
            py.detach(|| outlier_trajectory_impl(lats, lngs, times, &ranges, &config.0));
        return Ok(numpy_bool_output(py, keep_mask));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? && is_arrow_array(timestamps_s)? {
        let latitudes = as_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes = as_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        let timestamps_s = as_f64_array(timestamps_s.extract::<ArrowPyArray>()?, "timestamps_s")?;
        let keep_mask = py.detach(|| {
            outlier_trajectory_impl(
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
pub fn outlier_trajectory_indexed<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    timestamps_s: &Bound<'py, PyAny>,
    sorted_indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    config: PyOutlierConfig,
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
            outlier_trajectory_indexed_impl(
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
            outlier_trajectory_indexed_impl(
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
