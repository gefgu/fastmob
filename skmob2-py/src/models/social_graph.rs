use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use skmob2_core::models::social_graph::{edges_to_csr, random_geometric_graph};

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
