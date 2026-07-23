use fastmob_core::trajectory::distance::{
    dtw_distance_impl, frechet_distance_impl, hausdorff_distance_impl, lcss_similarity_impl,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::adapters::trajectory::run_two_sequence_f64;

#[pyclass(name = "DistanceConfig", from_py_object)]
#[derive(Clone)]
pub struct PyDistanceConfig {
    pub method: String,
    pub epsilon_km: f64,
}

#[pymethods]
impl PyDistanceConfig {
    #[new]
    #[pyo3(signature = (method="dtw", epsilon_km=0.1))]
    fn new(method: &str, epsilon_km: f64) -> Self {
        PyDistanceConfig {
            method: method.to_string(),
            epsilon_km,
        }
    }
}

#[pyfunction]
pub fn trajectory_distance<'py>(
    py: Python<'py>,
    latitudes_a: &Bound<'py, PyAny>,
    longitudes_a: &Bound<'py, PyAny>,
    latitudes_b: &Bound<'py, PyAny>,
    longitudes_b: &Bound<'py, PyAny>,
    config: PyDistanceConfig,
) -> PyResult<f64> {
    let epsilon_km = config.epsilon_km;
    match config.method.as_str() {
        "dtw" => run_two_sequence_f64(
            py,
            latitudes_a,
            longitudes_a,
            latitudes_b,
            longitudes_b,
            |a, b| dtw_distance_impl(a.latitudes, a.longitudes, b.latitudes, b.longitudes),
        ),
        "frechet" => run_two_sequence_f64(
            py,
            latitudes_a,
            longitudes_a,
            latitudes_b,
            longitudes_b,
            |a, b| frechet_distance_impl(a.latitudes, a.longitudes, b.latitudes, b.longitudes),
        ),
        "hausdorff" => run_two_sequence_f64(
            py,
            latitudes_a,
            longitudes_a,
            latitudes_b,
            longitudes_b,
            |a, b| hausdorff_distance_impl(a.latitudes, a.longitudes, b.latitudes, b.longitudes),
        ),
        "lcss" => run_two_sequence_f64(
            py,
            latitudes_a,
            longitudes_a,
            latitudes_b,
            longitudes_b,
            |a, b| {
                lcss_similarity_impl(
                    a.latitudes,
                    a.longitudes,
                    b.latitudes,
                    b.longitudes,
                    epsilon_km,
                )
            },
        ),
        other => Err(PyValueError::new_err(format!(
            "unknown distance method: {other:?}"
        ))),
    }
}
