use std::sync::Arc;

use arrow_array::{Array, ArrayRef, Float64Array, Int64Array, UInt32Array, UInt8Array};
use fastmob_core::social::co_presence_network::recast_classify;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
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

/// Arrow-only native RECAST classifier over factorized staypoint intervals.
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
    swaps_per_edge: usize,
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
    if ends.iter().zip(&starts).any(|(end, start)| end < start) {
        return Err(PyValueError::new_err("ends_ms must be >= starts_ms"));
    }
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
            swaps_per_edge,
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
