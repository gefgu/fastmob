use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use fastmob_core::measures::individual::motifs::{
    canonical_adjacency_form as core_canonical_adjacency_form,
    compute_daily_motifs as core_compute_daily_motifs,
};

#[pyfunction]
pub fn canonical_adjacency_form(n_nodes: u32, edges: Vec<(u32, u32)>) -> PyResult<i64> {
    core_canonical_adjacency_form(n_nodes, edges).map_err(PyValueError::new_err)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn compute_daily_motifs(
    unique_ids: Vec<String>,
    purposes: Vec<String>,
    start_hours: Vec<u32>,
    end_hours: Vec<u32>,
    date_ids: Vec<i32>,
    durations: Vec<Option<f64>>,
    user_ranges: Vec<(usize, usize)>,
    user_id_labels: Vec<String>,
) -> PyResult<(Vec<String>, Vec<i32>, Vec<i64>)> {
    core_compute_daily_motifs(
        unique_ids,
        purposes,
        start_hours,
        end_hours,
        date_ids,
        durations,
        user_ranges,
        user_id_labels,
    )
    .map_err(PyValueError::new_err)
}
