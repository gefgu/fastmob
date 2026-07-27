use std::str::FromStr;

use fastmob_core::trajectory::interpolate::{
    interpolate_trajectory_indexed_impl, interpolate_trajectory_presorted_impl,
    InterpolationConfig as CoreInterpolationConfig, InterpolationMethod,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, f64_results_into_arrow,
    validate_indexed_ends,
};

#[pyclass(name = "InterpolationConfig", from_py_object)]
#[derive(Clone, Copy)]
pub struct PyInterpolationConfig(pub CoreInterpolationConfig);

#[pymethods]
impl PyInterpolationConfig {
    #[new]
    #[pyo3(signature = (method="linear", sampling_rate_s=3600.0, max_speed_kmh=300.0, step_std_km=0.01, seed=0))]
    fn new(
        method: &str,
        sampling_rate_s: f64,
        max_speed_kmh: f64,
        step_std_km: f64,
        seed: u64,
    ) -> PyResult<Self> {
        let method = InterpolationMethod::from_str(method).map_err(PyValueError::new_err)?;
        Ok(PyInterpolationConfig(CoreInterpolationConfig::new(
            method,
            sampling_rate_s,
            max_speed_kmh,
            step_std_km,
            seed,
        )))
    }
}

fn is_arrow_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.hasattr("__arrow_c_array__")
}

fn numpy_f64_output(py: Python<'_>, values: Vec<f64>) -> Py<PyAny> {
    values.into_pyarray(py).into_any().unbind()
}

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

type InterpolateOutput<'py> = (Py<PyAny>, Py<PyAny>, Py<PyAny>, Bound<'py, PyArray1<usize>>);

const BACKEND_ERROR: &str =
    "latitudes, longitudes, and timestamps_s must all be NumPy arrays or all be Arrow arrays";

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn interpolate_trajectory_indexed<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    timestamps_s: &Bound<'py, PyAny>,
    sorted_indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    config: PyInterpolationConfig,
) -> PyResult<InterpolateOutput<'py>> {
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
        let (out_lats, out_lngs, out_times, out_user_indices) = py
            .detach(|| {
                interpolate_trajectory_indexed_impl(
                    lats,
                    lngs,
                    times,
                    sorted_indices,
                    ends,
                    None,
                    &config.0,
                )
            })
            .map_err(PyValueError::new_err)?;
        return Ok((
            numpy_f64_output(py, out_lats),
            numpy_f64_output(py, out_lngs),
            numpy_f64_output(py, out_times),
            out_user_indices.into_pyarray(py),
        ));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? && is_arrow_array(timestamps_s)? {
        let latitudes = as_nullable_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes =
            as_nullable_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        let timestamps_s =
            as_nullable_f64_array(timestamps_s.extract::<ArrowPyArray>()?, "timestamps_s")?;
        validate_indexed_ends(latitudes.len(), sorted_indices, ends)?;
        let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes, &timestamps_s]);
        let (out_lats, out_lngs, out_times, out_user_indices) = py
            .detach(|| {
                interpolate_trajectory_indexed_impl(
                    arrow_values(&latitudes),
                    arrow_values(&longitudes),
                    arrow_values(&timestamps_s),
                    sorted_indices,
                    ends,
                    valid_rows.as_deref(),
                    &config.0,
                )
            })
            .map_err(PyValueError::new_err)?;
        return Ok((
            arrow_f64_output(py, out_lats)?,
            arrow_f64_output(py, out_lngs)?,
            arrow_f64_output(py, out_times)?,
            out_user_indices.into_pyarray(py),
        ));
    }

    Err(PyTypeError::new_err(BACKEND_ERROR))
}

#[pyfunction]
pub fn interpolate_trajectory_presorted<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    timestamps_s: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
    config: PyInterpolationConfig,
) -> PyResult<InterpolateOutput<'py>> {
    let ends = ends.as_slice()?;

    if let (Ok(latitudes), Ok(longitudes), Ok(timestamps_s)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
        timestamps_s.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let lats = latitudes.as_slice()?;
        let lngs = longitudes.as_slice()?;
        let times = timestamps_s.as_slice()?;
        let (out_lats, out_lngs, out_times, out_user_indices) = py
            .detach(|| interpolate_trajectory_presorted_impl(lats, lngs, times, ends, &config.0))
            .map_err(PyValueError::new_err)?;
        return Ok((
            numpy_f64_output(py, out_lats),
            numpy_f64_output(py, out_lngs),
            numpy_f64_output(py, out_times),
            out_user_indices.into_pyarray(py),
        ));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? && is_arrow_array(timestamps_s)? {
        let latitudes = as_f64_array(latitudes.extract::<ArrowPyArray>()?, "latitudes")?;
        let longitudes = as_f64_array(longitudes.extract::<ArrowPyArray>()?, "longitudes")?;
        let timestamps_s = as_f64_array(timestamps_s.extract::<ArrowPyArray>()?, "timestamps_s")?;
        let (out_lats, out_lngs, out_times, out_user_indices) = py
            .detach(|| {
                interpolate_trajectory_presorted_impl(
                    arrow_values(&latitudes),
                    arrow_values(&longitudes),
                    arrow_values(&timestamps_s),
                    ends,
                    &config.0,
                )
            })
            .map_err(PyValueError::new_err)?;
        return Ok((
            arrow_f64_output(py, out_lats)?,
            arrow_f64_output(py, out_lngs)?,
            arrow_f64_output(py, out_times)?,
            out_user_indices.into_pyarray(py),
        ));
    }

    Err(PyTypeError::new_err(BACKEND_ERROR))
}
