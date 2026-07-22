use fastmob_core::measures::individual::entropy::{
    real_entropy_batch as core_real_entropy_batch,
    real_entropy_indexed_impl as core_real_entropy_indexed_impl,
    trajectory_entropy_batch as core_trajectory_entropy_batch,
    trajectory_predictability_batch as core_trajectory_predictability_batch,
};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::adapters::trajectory::{PyF64Result, run_indexed_coordinate_f64};

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
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyF64Result> {
    run_indexed_coordinate_f64(py, latitudes, longitudes, indices, ends, |view| {
        core_real_entropy_indexed_impl(
            view.coordinates.latitudes,
            view.coordinates.longitudes,
            view.indices,
            view.ends,
            view.valid_rows,
        )
    })
}
