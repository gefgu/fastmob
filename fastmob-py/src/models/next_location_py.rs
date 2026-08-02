use crate::utils::ArrowUsizeArrayExt;
use fastmob_core::models::next_location::{
    MarkovLocationModel, NextLocationConfig as CoreNextLocationConfig, markov_fit_indexed,
    markov_predict_batch,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// An order-k (+ backoff) Markov next-location model, fit once per user and
/// reused for many `predict_batch` calls -- fitting (building the
/// transition-count tables) is the expensive step, not prediction, so one
/// handle serves an entire evaluation pass instead of re-fitting per query
/// (same "prepare once, query many" shape as `RoadNetworkHandle`).
#[pyclass(name = "NextLocationModels")]
pub struct PyNextLocationModels {
    models: Vec<MarkovLocationModel>,
}

#[pymethods]
impl PyNextLocationModels {
    #[new]
    #[pyo3(signature = (location_codes, sorted_indices, ends, order=1, backoff=true))]
    fn new(
        location_codes: PyReadonlyArray1<'_, u64>,
        sorted_indices: pyo3_arrow::PyArray,
        ends: pyo3_arrow::PyArray,
        order: usize,
        backoff: bool,
    ) -> PyResult<Self> {
        let codes = location_codes.as_slice()?;
        let indices = sorted_indices.as_slice()?;
        let ends = ends.as_slice()?;
        let config = CoreNextLocationConfig::new(order, backoff);
        let models =
            markov_fit_indexed(codes, indices, ends, &config).map_err(PyValueError::new_err)?;
        Ok(Self { models })
    }

    /// One context vector per user (same user order as `fit`, i.e. index
    /// `i` predicts for the same user whose training sequence was
    /// `location_codes[sorted_indices[start_i..end_i]]` at fit time).
    /// Returns `(codes, probs, out_user_starts, out_user_ends)`: flat
    /// top-`top_k` candidates per user (fewer than `top_k` when a user has
    /// fewer distinct next-location candidates, or backoff is disabled and
    /// the context was never seen), ordered by descending observed count.
    #[pyo3(signature = (context_codes, context_starts, context_ends, top_k=1))]
    fn predict_batch<'py>(
        &self,
        py: Python<'py>,
        context_codes: PyReadonlyArray1<'py, u64>,
        context_starts: pyo3_arrow::PyArray,
        context_ends: pyo3_arrow::PyArray,
        top_k: usize,
    ) -> PyResult<(
        Bound<'py, PyArray1<u64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<usize>>,
        Bound<'py, PyArray1<usize>>,
    )> {
        let codes = context_codes.as_slice()?;
        let starts = context_starts.as_slice()?;
        let ends = context_ends.as_slice()?;
        let (out_codes, out_probs, out_starts, out_ends) =
            markov_predict_batch(&self.models, codes, starts, ends, top_k)
                .map_err(PyValueError::new_err)?;
        Ok((
            out_codes.into_pyarray(py),
            out_probs.into_pyarray(py),
            out_starts.into_pyarray(py),
            out_ends.into_pyarray(py),
        ))
    }

    fn __len__(&self) -> usize {
        self.models.len()
    }
}
