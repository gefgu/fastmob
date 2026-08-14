use crate::utils::{
    ArrowUsizeArrayExt, arrow_u32_values, as_u32_array, f64_results_into_arrow,
    u32_results_into_arrow, u64_results_into_arrow,
};
use fastmob_core::models::next_location::{
    MarkovLocationModel, NextLocationConfig as CoreNextLocationConfig, markov_fit_indexed,
    markov_predict_batch,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

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
        location_codes: ArrowPyArray,
        sorted_indices: ArrowPyArray,
        ends: ArrowPyArray,
        order: usize,
        backoff: bool,
    ) -> PyResult<Self> {
        let codes = as_u32_array(location_codes, "location_codes")?;
        let indices = sorted_indices.as_slice()?;
        let ends = ends.as_slice()?;
        let config = CoreNextLocationConfig::new(order, backoff);
        let models = markov_fit_indexed(arrow_u32_values(&codes), indices, ends, &config)
            .map_err(PyValueError::new_err)?;
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
        context_codes: ArrowPyArray,
        context_starts: ArrowPyArray,
        context_ends: ArrowPyArray,
        top_k: usize,
    ) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
        let codes = as_u32_array(context_codes, "context_codes")?;
        let starts = context_starts.as_slice()?;
        let ends = context_ends.as_slice()?;
        let (out_codes, out_probs, out_starts, out_ends) =
            markov_predict_batch(&self.models, arrow_u32_values(&codes), starts, ends, top_k)
                .map_err(PyValueError::new_err)?;
        Ok((
            Py::new(py, u32_results_into_arrow(out_codes))?.into_any(),
            Py::new(py, f64_results_into_arrow(out_probs))?.into_any(),
            Py::new(
                py,
                u64_results_into_arrow(out_starts.into_iter().map(|v| v as u64).collect()),
            )?
            .into_any(),
            Py::new(
                py,
                u64_results_into_arrow(out_ends.into_iter().map(|v| v as u64).collect()),
            )?
            .into_any(),
        ))
    }

    fn __len__(&self) -> usize {
        self.models.len()
    }
}
