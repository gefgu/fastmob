use fastmob_core::measures::individual::entropy::{
    real_entropy_batch as core_real_entropy_batch,
    real_entropy_indexed_impl as core_real_entropy_indexed_impl,
    trajectory_entropy_batch as core_trajectory_entropy_batch,
    trajectory_predictability_batch as core_trajectory_predictability_batch,
};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::adapters::trajectory::run_indexed_coordinate_arrow;
use crate::utils::f64_results_into_arrow;

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

type PredictabilityBatchResult = (Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>);

#[pyfunction]
pub fn real_entropy_batch(
    py: Python<'_>,
    tokens: Vec<String>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    let result = py.detach(move || core_real_entropy_batch(tokens, ranges));
    result.map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn trajectory_entropy_batch(
    py: Python<'_>,
    tokens: Vec<String>,
    ranges: Vec<(usize, usize)>,
    normalized: bool,
) -> PyResult<Vec<f64>> {
    let result = py.detach(move || core_trajectory_entropy_batch(tokens, ranges, normalized));
    result.map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn trajectory_predictability_batch(
    py: Python<'_>,
    tokens: Vec<String>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<PredictabilityBatchResult> {
    let result = py.detach(move || core_trajectory_predictability_batch(tokens, ranges));
    result.map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn real_entropy_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<Py<PyAny>> {
    let values = run_indexed_coordinate_arrow(py, latitudes, longitudes, indices, ends, |view| {
        core_real_entropy_indexed_impl(
            view.coordinates.latitudes,
            view.coordinates.longitudes,
            view.indices,
            view.ends,
            view.valid_rows,
        )
    })?
    .map_err(PyValueError::new_err)?;
    arrow_f64_output(py, values)
}
