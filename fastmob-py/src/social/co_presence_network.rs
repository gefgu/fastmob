use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float32Array, Float64Array, Int64Array, UInt8Array, UInt32Array, UInt64Array,
};
use arrow_buffer::ScalarBuffer;
use fastmob_core::social::co_presence_network::{
    aggregate_flat_events, contact_graph_metrics, event_graphs, expected_degree_graph,
    flatten_events, local_clustering_coefficients, recast_classify, rnd_graph, t_rnd_graph,
    topological_overlaps, validate_recast,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3_arrow::PyArray as ArrowPyArray;

fn u32_values(input: ArrowPyArray, name: &str) -> PyResult<ScalarBuffer<u32>> {
    let (array, _) = input.into_inner();
    let values = array
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| PyValueError::new_err(format!("expected uint32 Arrow array for {name}")))?;
    if values.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }
    Ok(values.values().clone())
}
fn u64_values(input: ArrowPyArray, name: &str) -> PyResult<ScalarBuffer<u64>> {
    let (array, _) = input.into_inner();
    let values = array
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| PyValueError::new_err(format!("expected uint64 Arrow array for {name}")))?;
    if values.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }
    Ok(values.values().clone())
}
fn i64_values(input: ArrowPyArray, name: &str) -> PyResult<ScalarBuffer<i64>> {
    let (array, _) = input.into_inner();
    let values = array
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| PyValueError::new_err(format!("expected int64 Arrow array for {name}")))?;
    if values.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }
    Ok(values.values().clone())
}
fn arrow<T: Array + 'static>(values: T) -> ArrowPyArray {
    ArrowPyArray::from_array_ref(Arc::new(values) as ArrayRef)
}
fn validate_lengths(from: &[u32], to: &[u32]) -> PyResult<()> {
    if from.len() != to.len() {
        Err(PyValueError::new_err(
            "edge_from and edge_to must have the same length",
        ))
    } else {
        Ok(())
    }
}

fn validate_graph(node_count: usize, from: &[u32], to: &[u32]) -> PyResult<()> {
    validate_lengths(from, to)?;
    if from
        .iter()
        .chain(to.iter())
        .any(|&node| node as usize >= node_count)
    {
        return Err(PyValueError::new_err(
            "edge endpoints must be smaller than node_count",
        ));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "contact_graph_metrics")]
pub fn contact_graph_metrics_py<'py>(
    py: Python<'py>,
    node_count: usize,
    edge_from: PyReadonlyArray1<'py, u32>,
    edge_to: PyReadonlyArray1<'py, u32>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let from = edge_from.as_slice()?;
    let to = edge_to.as_slice()?;
    validate_graph(node_count, from, to)?;
    let metrics = py.detach(|| contact_graph_metrics(node_count, from, to));
    Ok((
        metrics.clustering_coefficient.into_pyarray(py),
        metrics.topological_overlap.into_pyarray(py),
    ))
}

#[pyfunction]
#[pyo3(name = "recast_local_clustering_coefficients")]
pub fn recast_local_clustering_coefficients_py<'py>(
    py: Python<'py>,
    node_count: usize,
    edge_from: PyReadonlyArray1<'py, u32>,
    edge_to: PyReadonlyArray1<'py, u32>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let from = edge_from.as_slice()?;
    let to = edge_to.as_slice()?;
    validate_graph(node_count, from, to)?;
    Ok(py
        .detach(|| local_clustering_coefficients(node_count, from, to))
        .into_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "recast_topological_overlaps")]
pub fn recast_topological_overlaps_py<'py>(
    py: Python<'py>,
    node_count: usize,
    edge_from: PyReadonlyArray1<'py, u32>,
    edge_to: PyReadonlyArray1<'py, u32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let from = edge_from.as_slice()?;
    let to = edge_to.as_slice()?;
    validate_graph(node_count, from, to)?;
    Ok(py
        .detach(|| topological_overlaps(node_count, from, to))
        .into_pyarray(py))
}

#[pyfunction]
#[pyo3(name = "recast_expected_degree_graph")]
pub fn recast_expected_degree_graph_py<'py>(
    py: Python<'py>,
    degrees: PyReadonlyArray1<'py, u64>,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray1<u32>>, Bound<'py, PyArray1<u32>>)> {
    let degrees = degrees.as_slice()?;
    if degrees.iter().any(|&degree| degree > usize::MAX as u64) {
        return Err(PyValueError::new_err(
            "degrees exceed this platform's supported range",
        ));
    }
    let degree: Vec<usize> = degrees.iter().map(|&value| value as usize).collect();
    let (from, to) = py.detach(|| expected_degree_graph(&degree, seed));
    Ok((from.into_pyarray(py), to.into_pyarray(py)))
}

#[pyfunction]
#[pyo3(name = "recast_aggregate_event_graphs")]
pub fn recast_aggregate_event_graphs_py(
    py: Python<'_>,
    edge_offsets: ArrowPyArray,
    edge_from: ArrowPyArray,
    edge_to: ArrowPyArray,
) -> PyResult<(ArrowPyArray, ArrowPyArray, ArrowPyArray)> {
    let offsets = u64_values(edge_offsets, "edge_offsets")?;
    let from = u32_values(edge_from, "edge_from")?;
    let to = u32_values(edge_to, "edge_to")?;
    if from.len() != to.len() {
        return Err(PyValueError::new_err(
            "edge_from and edge_to must have the same length",
        ));
    }
    if offsets.is_empty()
        || offsets.first() != Some(&0)
        || offsets.last().copied() != Some(from.len() as u64)
        || offsets.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(PyValueError::new_err(
            "edge_offsets must start at 0, end at edge count, and be non-decreasing",
        ));
    }
    let (from, to, persistence) = py.detach(|| aggregate_flat_events(&offsets, &from, &to));
    Ok((
        arrow(UInt32Array::from(from)),
        arrow(UInt32Array::from(to)),
        arrow(Float64Array::from(persistence)),
    ))
}
fn validate_recast_args(
    users: &[u32],
    locations: &[u32],
    starts: &[i64],
    ends: &[i64],
    min_contact_ms: i64,
    p_rnd: f64,
    replicas: usize,
) -> PyResult<()> {
    if users.len() != locations.len() || users.len() != starts.len() || users.len() != ends.len() {
        return Err(PyValueError::new_err(
            "users, locations, starts_ms and ends_ms must have the same length",
        ));
    }
    if min_contact_ms <= 0 {
        return Err(PyValueError::new_err("min_contact_ms must be positive"));
    }
    if !(0.0..=1.0).contains(&p_rnd) {
        return Err(PyValueError::new_err("p_rnd must be between 0 and 1"));
    }
    if replicas == 0 {
        return Err(PyValueError::new_err("replicas must be at least 1"));
    }
    if ends.iter().zip(starts).any(|(end, start)| end < start) {
        return Err(PyValueError::new_err("ends_ms must be >= starts_ms"));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "recast_event_graphs")]
pub fn recast_event_graphs_py(
    py: Python<'_>,
    users: ArrowPyArray,
    locations: ArrowPyArray,
    starts_ms: ArrowPyArray,
    ends_ms: ArrowPyArray,
    min_contact_ms: i64,
) -> PyResult<(ArrowPyArray, ArrowPyArray, ArrowPyArray, ArrowPyArray)> {
    let users = u32_values(users, "users")?;
    let locations = u32_values(locations, "locations")?;
    let starts = i64_values(starts_ms, "starts_ms")?;
    let ends = i64_values(ends_ms, "ends_ms")?;
    if users.len() != locations.len() || users.len() != starts.len() || users.len() != ends.len() {
        return Err(PyValueError::new_err(
            "users, locations, starts_ms and ends_ms must have the same length",
        ));
    }
    if min_contact_ms <= 0 {
        return Err(PyValueError::new_err("min_contact_ms must be positive"));
    }
    if ends.iter().zip(&starts).any(|(end, start)| end < start) {
        return Err(PyValueError::new_err("ends_ms must be >= starts_ms"));
    }
    let graph = py.detach(|| {
        flatten_events(&event_graphs(
            &users,
            &locations,
            &starts,
            &ends,
            min_contact_ms,
        ))
    });
    Ok((
        arrow(Int64Array::from(graph.window_starts_ms)),
        arrow(UInt64Array::from(graph.edge_offsets)),
        arrow(UInt32Array::from(graph.edge_from)),
        arrow(UInt32Array::from(graph.edge_to)),
    ))
}

#[pyfunction]
#[pyo3(name = "recast_classify")]
#[allow(clippy::too_many_arguments)]
pub fn recast_classify_py(
    py: Python<'_>,
    node_count: usize,
    users: ArrowPyArray,
    locations: ArrowPyArray,
    starts_ms: ArrowPyArray,
    ends_ms: ArrowPyArray,
    min_contact_ms: i64,
    p_rnd: f64,
    replicas: usize,
    seed: u64,
) -> PyResult<(
    ArrowPyArray,
    ArrowPyArray,
    ArrowPyArray,
    ArrowPyArray,
    ArrowPyArray,
    f64,
    f64,
    usize,
)> {
    let users = u32_values(users, "users")?;
    let locations = u32_values(locations, "locations")?;
    let starts = i64_values(starts_ms, "starts_ms")?;
    let ends = i64_values(ends_ms, "ends_ms")?;
    validate_recast_args(
        &users,
        &locations,
        &starts,
        &ends,
        min_contact_ms,
        p_rnd,
        replicas,
    )?;
    let result = py.detach(|| {
        recast_classify(
            node_count,
            &users,
            &locations,
            &starts,
            &ends,
            min_contact_ms,
            p_rnd,
            replicas,
            seed,
        )
    });
    Ok((
        arrow(UInt32Array::from(result.edge_from)),
        arrow(UInt32Array::from(result.edge_to)),
        arrow(Float32Array::from(result.persistence)),
        arrow(Float32Array::from(result.topological_overlap)),
        arrow(UInt8Array::from(result.classes)),
        result.persistence_threshold,
        result.overlap_threshold,
        result.time_steps,
    ))
}

#[pyfunction]
#[pyo3(name = "recast_rnd")]
pub fn recast_rnd_py(
    py: Python<'_>,
    node_count: usize,
    edge_from: ArrowPyArray,
    edge_to: ArrowPyArray,
    seed: u64,
) -> PyResult<(ArrowPyArray, ArrowPyArray)> {
    let from = u32_values(edge_from, "edge_from")?;
    let to = u32_values(edge_to, "edge_to")?;
    validate_lengths(&from, &to)?;
    if from.iter().chain(&to).any(|&x| x as usize >= node_count) {
        return Err(PyValueError::new_err(
            "edge endpoints must be smaller than node_count",
        ));
    }
    let (from, to) = py.detach(|| rnd_graph(node_count, &from, &to, seed));
    Ok((arrow(UInt32Array::from(from)), arrow(UInt32Array::from(to))))
}

#[pyfunction]
#[pyo3(name = "recast_t_rnd")]
#[allow(clippy::too_many_arguments)]
pub fn recast_t_rnd_py(
    py: Python<'_>,
    node_count: usize,
    window_starts_ms: ArrowPyArray,
    edge_offsets: ArrowPyArray,
    edge_from: ArrowPyArray,
    edge_to: ArrowPyArray,
    replica: usize,
    seed: u64,
) -> PyResult<(ArrowPyArray, ArrowPyArray, ArrowPyArray, ArrowPyArray)> {
    let windows = i64_values(window_starts_ms, "window_starts_ms")?;
    let offsets = u64_values(edge_offsets, "edge_offsets")?;
    let from = u32_values(edge_from, "edge_from")?;
    let to = u32_values(edge_to, "edge_to")?;
    validate_lengths(&from, &to)?;
    if offsets.len() != windows.len() + 1
        || offsets.first() != Some(&0)
        || offsets.last().copied() != Some(from.len() as u64)
        || offsets.windows(2).any(|x| x[0] > x[1])
    {
        return Err(PyValueError::new_err(
            "edge_offsets must start at 0, end at edge count, and have one entry per window plus one",
        ));
    }
    let graph =
        py.detach(|| t_rnd_graph(node_count, &windows, &offsets, &from, &to, replica, seed));
    Ok((
        arrow(Int64Array::from(graph.window_starts_ms)),
        arrow(UInt64Array::from(graph.edge_offsets)),
        arrow(UInt32Array::from(graph.edge_from)),
        arrow(UInt32Array::from(graph.edge_to)),
    ))
}

#[pyfunction]
#[pyo3(name = "recast_validate")]
#[allow(clippy::too_many_arguments)]
pub fn recast_validate_py(
    py: Python<'_>,
    node_count: usize,
    users: ArrowPyArray,
    locations: ArrowPyArray,
    starts_ms: ArrowPyArray,
    ends_ms: ArrowPyArray,
    min_contact_ms: i64,
    p_rnd: f64,
    replicas: usize,
    seed: u64,
) -> PyResult<Py<PyAny>> {
    let users = u32_values(users, "users")?;
    let locations = u32_values(locations, "locations")?;
    let starts = i64_values(starts_ms, "starts_ms")?;
    let ends = i64_values(ends_ms, "ends_ms")?;
    validate_recast_args(
        &users,
        &locations,
        &starts,
        &ends,
        min_contact_ms,
        p_rnd,
        replicas,
    )?;
    let result = py.detach(|| {
        validate_recast(
            node_count,
            &users,
            &locations,
            &starts,
            &ends,
            min_contact_ms,
            p_rnd,
            replicas,
            seed,
        )
    });
    let c = result.classification;
    let g = result.graph;
    let out = PyList::empty(py);
    out.append(arrow(UInt32Array::from(c.edge_from)))?;
    out.append(arrow(UInt32Array::from(c.edge_to)))?;
    out.append(arrow(Float32Array::from(c.persistence)))?;
    out.append(arrow(Float32Array::from(c.topological_overlap)))?;
    out.append(arrow(UInt8Array::from(c.classes)))?;
    out.append(c.persistence_threshold)?;
    out.append(c.overlap_threshold)?;
    out.append(c.time_steps)?;
    out.append(arrow(Float32Array::from(result.observed_persistence)))?;
    out.append(arrow(Float32Array::from(result.observed_overlap)))?;
    out.append(arrow(Float32Array::from(result.random_persistence)))?;
    out.append(arrow(Float32Array::from(result.random_overlap)))?;
    out.append(arrow(Float64Array::from(result.full_observed_clustering)))?;
    out.append(arrow(Float64Array::from(result.full_random_mean)))?;
    out.append(arrow(Float64Array::from(result.full_random_std)))?;
    out.append(arrow(Float64Array::from(
        result.random_only_observed_clustering,
    )))?;
    out.append(arrow(Float64Array::from(result.random_only_random_mean)))?;
    out.append(arrow(Float64Array::from(result.random_only_random_std)))?;
    out.append(arrow(Int64Array::from(g.window_starts_ms)))?;
    out.append(arrow(UInt64Array::from(g.edge_offsets)))?;
    out.append(arrow(UInt32Array::from(g.edge_from)))?;
    out.append(arrow(UInt32Array::from(g.edge_to)))?;
    Ok(out.into_any().unbind())
}
