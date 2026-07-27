use fastmob_core::measures::collective::co_presence_network::{
    build_co_presence_edges, compute_graph_metrics, random_baseline_overlap_threshold,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Groups `(day, location, node)` presence rows by `(day, location)` and
/// emits one edge per unique co-presence pair, with per-edge persistence
/// (fraction of `time_steps` the pair was seen together on). Replaces a
/// Python `itertools.combinations` + `dict[edge, set[day]]` loop, which
/// doesn't scale past toy graphs. Groups larger than `max_group_size` are
/// skipped (`skipped_groups`/`skipped_rows` report how many/how large).
#[pyfunction]
#[pyo3(name = "build_co_presence_edges")]
pub fn build_co_presence_edges_py<'py>(
    py: Python<'py>,
    day_codes: PyReadonlyArray1<'py, i64>,
    location_codes: PyReadonlyArray1<'py, i64>,
    nodes: PyReadonlyArray1<'py, i64>,
    max_group_size: usize,
    time_steps: usize,
) -> PyResult<(
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<u32>>,
    Bound<'py, PyArray1<f64>>,
    u64,
    u64,
)> {
    let day = day_codes.as_slice()?;
    let location = location_codes.as_slice()?;
    let node = nodes.as_slice()?;
    if day.len() != location.len() || day.len() != node.len() {
        return Err(PyValueError::new_err(
            "day_codes, location_codes and nodes must have the same length",
        ));
    }
    let (edge_from, edge_to, persistence, skipped_groups, skipped_rows) =
        py.detach(|| build_co_presence_edges(day, location, node, max_group_size, time_steps));
    Ok((
        edge_from.into_pyarray(py),
        edge_to.into_pyarray(py),
        persistence.into_pyarray(py),
        skipped_groups,
        skipped_rows,
    ))
}

/// Per-node clustering coefficient and per-edge topological overlap
/// (Jaccard similarity of endpoint neighborhoods) for an undirected graph
/// given as an edge list. Replaces pure-Python `O(sum of degree^2)` nested
/// loops over `set`-based adjacency.
#[pyfunction]
#[pyo3(name = "graph_metrics")]
pub fn graph_metrics_py<'py>(
    py: Python<'py>,
    node_count: usize,
    edge_from: PyReadonlyArray1<'py, u32>,
    edge_to: PyReadonlyArray1<'py, u32>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let from = edge_from.as_slice()?;
    let to = edge_to.as_slice()?;
    if from.len() != to.len() {
        return Err(PyValueError::new_err(
            "edge_from and edge_to must have the same length",
        ));
    }
    let metrics = py.detach(|| compute_graph_metrics(node_count, from, to));
    Ok((
        metrics.clustering_coefficient.into_pyarray(py),
        metrics.topological_overlap.into_pyarray(py),
    ))
}

/// Samples random node pairs and returns their topological overlaps'
/// `(1 - p_rnd)`-quantile -- the null-model significance threshold used by
/// social-tie inference. Deterministic given `seed`.
#[pyfunction]
#[pyo3(name = "random_baseline_overlap_threshold")]
#[allow(clippy::too_many_arguments)]
pub fn random_baseline_overlap_threshold_py(
    py: Python<'_>,
    node_count: usize,
    edge_from: PyReadonlyArray1<u32>,
    edge_to: PyReadonlyArray1<u32>,
    samples: usize,
    p_rnd: f64,
    seed: u64,
) -> PyResult<f64> {
    let from = edge_from.as_slice()?;
    let to = edge_to.as_slice()?;
    if from.len() != to.len() {
        return Err(PyValueError::new_err(
            "edge_from and edge_to must have the same length",
        ));
    }
    Ok(py.detach(|| random_baseline_overlap_threshold(node_count, from, to, samples, p_rnd, seed)))
}
