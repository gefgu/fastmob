use fastmob_core::preprocessing::filter_traj::{
    FilterConfig as CoreFilterConfig, filter_trajectory_impl, filter_trajectory_indexed_impl,
};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, bool_results_into_arrow,
    validate_indexed_ends,
};

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
pub fn filter_trajectory_sorted<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    timestamps_s: &Bound<'py, PyAny>,
    ranges: Vec<(usize, usize)>,
    config: PyFilterConfig,
) -> PyResult<Py<PyAny>> {
    if let (Ok(latitudes), Ok(longitudes), Ok(timestamps_s)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
        timestamps_s.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let lats = latitudes.as_slice()?;
        let lngs = longitudes.as_slice()?;
        let times = timestamps_s.as_slice()?;
        let keep_mask = py.detach(|| filter_trajectory_impl(lats, lngs, times, &ranges, &config.0));
        return Ok(numpy_bool_output(py, keep_mask));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? && is_arrow_array(timestamps_s)? {
        let latitudes = as_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes = as_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        let timestamps_s = as_f64_array(timestamps_s.extract::<ArrowPyArray>()?, "timestamps_s")?;
        let keep_mask = py.detach(|| {
            filter_trajectory_impl(
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
pub fn filter_trajectory_indexed<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    timestamps_s: &Bound<'py, PyAny>,
    sorted_indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    config: PyFilterConfig,
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
            filter_trajectory_indexed_impl(lats, lngs, times, sorted_indices, ends, None, &config.0)
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
            filter_trajectory_indexed_impl(
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
