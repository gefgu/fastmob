use fastmob_core::models::social_graph::{edges_to_csr, random_geometric_graph};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{arrow_i64_values, as_i64_array, i64_results_into_arrow};

#[pyfunction]
pub fn model_social_graph_edges_to_csr(
    srcs: PyReadonlyArray1<i64>,
    dsts: PyReadonlyArray1<i64>,
    n_nodes: usize,
) -> PyResult<(Vec<i64>, Vec<i64>)> {
    let s: Vec<usize> = srcs
        .as_slice()?
        .iter()
        .map(|&v| v.max(0) as usize)
        .collect();
    let d: Vec<usize> = dsts
        .as_slice()?
        .iter()
        .map(|&v| v.max(0) as usize)
        .collect();
    let (starts, nbrs) = edges_to_csr(&s, &d, n_nodes);
    Ok((
        starts.into_iter().map(|v| v as i64).collect(),
        nbrs.into_iter().map(|v| v as i64).collect(),
    ))
}

#[pyfunction]
pub fn model_social_graph_random_geometric(
    n_nodes: usize,
    radius: f64,
    seed: u64,
) -> (Vec<i64>, Vec<i64>) {
    let (starts, nbrs) = random_geometric_graph(n_nodes, radius, seed);
    (
        starts.into_iter().map(|v| v as i64).collect(),
        nbrs.into_iter().map(|v| v as i64).collect(),
    )
}

#[pyfunction]
pub fn model_social_graph_edges_to_csr_arrow<'py>(
    py: Python<'py>,
    srcs: ArrowPyArray,
    dsts: ArrowPyArray,
    n_nodes: usize,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let srcs = as_i64_array(srcs, "srcs")?;
    let dsts = as_i64_array(dsts, "dsts")?;
    let sources: Vec<usize> = arrow_i64_values(&srcs)
        .iter()
        .map(|&v| v.max(0) as usize)
        .collect();
    let destinations: Vec<usize> = arrow_i64_values(&dsts)
        .iter()
        .map(|&v| v.max(0) as usize)
        .collect();
    let (starts, neighbors) = edges_to_csr(&sources, &destinations, n_nodes);
    Ok((
        Py::new(
            py,
            i64_results_into_arrow(starts.into_iter().map(|v| v as i64).collect()),
        )?
        .into_any(),
        Py::new(
            py,
            i64_results_into_arrow(neighbors.into_iter().map(|v| v as i64).collect()),
        )?
        .into_any(),
    ))
}

#[pyfunction]
pub fn model_social_graph_random_geometric_arrow<'py>(
    py: Python<'py>,
    n_nodes: usize,
    radius: f64,
    seed: u64,
) -> PyResult<(Py<PyAny>, Py<PyAny>)> {
    let (starts, neighbors) = random_geometric_graph(n_nodes, radius, seed);
    Ok((
        Py::new(
            py,
            i64_results_into_arrow(starts.into_iter().map(|v| v as i64).collect()),
        )?
        .into_any(),
        Py::new(
            py,
            i64_results_into_arrow(neighbors.into_iter().map(|v| v as i64).collect()),
        )?
        .into_any(),
    ))
}
