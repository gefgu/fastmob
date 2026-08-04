use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float64Array, Int64Array, UInt32Array, UInt64Array, UInt8Array,
};
use fastmob_core::social::co_presence_network::{
    event_graphs, flatten_events, recast_classify, rnd_graph, t_rnd_graph, validate_recast,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use pyo3_arrow::PyArray as ArrowPyArray;

fn u32_values(input: ArrowPyArray, name: &str) -> PyResult<Vec<u32>> {
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
    Ok((0..values.len()).map(|i| values.value(i)).collect())
}
fn u64_values(input: ArrowPyArray, name: &str) -> PyResult<Vec<u64>> {
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
    Ok((0..values.len()).map(|i| values.value(i)).collect())
}
fn i64_values(input: ArrowPyArray, name: &str) -> PyResult<Vec<i64>> {
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
    Ok((0..values.len()).map(|i| values.value(i)).collect())
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
        arrow(Float64Array::from(result.persistence)),
        arrow(Float64Array::from(result.topological_overlap)),
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
        return Err(PyValueError::new_err("edge_offsets must start at 0, end at edge count, and have one entry per window plus one"));
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
    out.append(arrow(Float64Array::from(c.persistence)))?;
    out.append(arrow(Float64Array::from(c.topological_overlap)))?;
    out.append(arrow(UInt8Array::from(c.classes)))?;
    out.append(c.persistence_threshold)?;
    out.append(c.overlap_threshold)?;
    out.append(c.time_steps)?;
    out.append(arrow(Float64Array::from(result.observed_persistence)))?;
    out.append(arrow(Float64Array::from(result.observed_overlap)))?;
    out.append(arrow(Float64Array::from(result.random_persistence)))?;
    out.append(arrow(Float64Array::from(result.random_overlap)))?;
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
