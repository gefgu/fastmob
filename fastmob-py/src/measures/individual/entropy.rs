use fastmob_core::measures::individual::entropy::{
    real_entropy_batch as core_real_entropy_batch,
    real_entropy_indexed_impl as core_real_entropy_indexed_impl,
    trajectory_entropy_batch as core_trajectory_entropy_batch,
    trajectory_predictability_batch as core_trajectory_predictability_batch,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{arrow_valid_rows, arrow_values, as_nullable_f64_array, f64_results_into_arrow};

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
pub fn real_entropy_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(core_real_entropy_indexed_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        indices.as_slice()?,
        ends.as_slice()?,
        None,
    )
    .map_err(PyValueError::new_err)?
    .into_pyarray(py))
}

#[pyfunction]
pub fn real_entropy_indexed_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    indices: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<PyArray> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    Ok(f64_results_into_arrow(
        core_real_entropy_indexed_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            indices.as_slice()?,
            ends.as_slice()?,
            valid_rows.as_deref(),
        )
        .map_err(PyValueError::new_err)?,
    ))
}
