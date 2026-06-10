use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use skmob2_core::models::markov_diary::{
    markov_diary_batch_generate_impl, markov_diary_build_cdf_impl, markov_diary_generate_impl,
    markov_diary_normalize_impl, markov_diary_update_chain_impl,
};

#[pyfunction]
pub fn markov_diary_update_chain<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, u64>,
    shift: usize,
    counts: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let vals: Vec<usize> = values.as_slice()?.iter().map(|&v| v as usize).collect();
    let mut out = counts.as_slice()?.to_vec();
    markov_diary_update_chain_impl(&vals, shift, &mut out).map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray(py))
}

#[pyfunction]
pub fn markov_diary_normalize<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let mut out = counts.as_slice().unwrap_or(&[]).to_vec();
    markov_diary_normalize_impl(&mut out);
    out.into_pyarray(py)
}

#[pyfunction]
pub fn markov_diary_build_cdf<'py>(
    py: Python<'py>,
    probs: PyReadonlyArray1<'py, f64>,
) -> Bound<'py, PyArray1<f64>> {
    let cdf = markov_diary_build_cdf_impl(probs.as_slice().unwrap_or(&[]));
    cdf.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (cdf_matrix, diary_length, start_ts, n_agents, master_seed))]
pub fn markov_diary_batch_generate<'py>(
    py: Python<'py>,
    cdf_matrix: Option<PyReadonlyArray1<'py, f64>>,
    diary_length: usize,
    start_ts: i64,
    n_agents: usize,
    master_seed: u64,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i32>>,
    Vec<usize>,
    Vec<usize>,
)> {
    let cdf_slice: Option<&[f64]> = match &cdf_matrix {
        Some(m) => Some(m.as_slice()?),
        None => None,
    };
    let (flat_ts, flat_locs, starts, ends) =
        markov_diary_batch_generate_impl(cdf_slice, diary_length, start_ts, n_agents, master_seed);
    Ok((
        flat_ts.into_pyarray(py),
        flat_locs.into_pyarray(py),
        starts,
        ends,
    ))
}

#[pyfunction]
#[pyo3(signature = (cdf_matrix, diary_length, start_ts, seed))]
pub fn markov_diary_generate<'py>(
    py: Python<'py>,
    cdf_matrix: PyReadonlyArray1<'py, f64>,
    diary_length: usize,
    start_ts: i64,
    seed: u64,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i32>>)> {
    let matrix = cdf_matrix.as_slice()?;
    let (ts, locs) = markov_diary_generate_impl(matrix, diary_length, start_ts, seed);
    Ok((ts.into_pyarray(py), locs.into_pyarray(py)))
}
