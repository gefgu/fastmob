use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use skmob2_core::models::markov_diary::{
    markov_diary_batch_generate_impl, markov_diary_build_cdf_impl,
    markov_diary_fit_from_arrays_impl, markov_diary_generate_impl, markov_diary_normalize_impl,
    markov_diary_update_chain_impl,
};

const SECONDS_PER_DAY: i64 = 86_400;

#[inline]
fn slot_seconds_for(slots_per_day: usize) -> i64 {
    SECONDS_PER_DAY / slots_per_day as i64
}

#[pyfunction]
#[pyo3(signature = (values, shift, counts, slots_per_day=24))]
pub fn markov_diary_update_chain<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, u64>,
    shift: usize,
    counts: PyReadonlyArray1<'py, f64>,
    slots_per_day: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let vals: Vec<usize> = values.as_slice()?.iter().map(|&v| v as usize).collect();
    let mut out = counts.as_slice()?.to_vec();
    markov_diary_update_chain_impl(&vals, shift, &mut out, slots_per_day)
        .map_err(PyValueError::new_err)?;
    Ok(out.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (counts, slots_per_day=24))]
pub fn markov_diary_normalize<'py>(
    py: Python<'py>,
    counts: PyReadonlyArray1<'py, f64>,
    slots_per_day: usize,
) -> Bound<'py, PyArray1<f64>> {
    let mut out = counts.as_slice().unwrap_or(&[]).to_vec();
    markov_diary_normalize_impl(&mut out, slots_per_day * 2);
    out.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (probs, slots_per_day=24))]
pub fn markov_diary_build_cdf<'py>(
    py: Python<'py>,
    probs: PyReadonlyArray1<'py, f64>,
    slots_per_day: usize,
) -> Bound<'py, PyArray1<f64>> {
    let cdf = markov_diary_build_cdf_impl(probs.as_slice().unwrap_or(&[]), slots_per_day * 2);
    cdf.into_pyarray(py)
}

#[pyfunction]
#[pyo3(signature = (uids, timestamps_ns, loc_codes, n_individuals, slots_per_day=24))]
pub fn markov_diary_fit_from_arrays<'py>(
    py: Python<'py>,
    uids: PyReadonlyArray1<'py, i64>,
    timestamps_ns: PyReadonlyArray1<'py, i64>,
    loc_codes: PyReadonlyArray1<'py, i64>,
    n_individuals: usize,
    slots_per_day: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let cdf = markov_diary_fit_from_arrays_impl(
        uids.as_slice()?,
        timestamps_ns.as_slice()?,
        loc_codes.as_slice()?,
        n_individuals,
        slots_per_day,
    )
    .map_err(PyValueError::new_err)?;
    Ok(cdf.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (cdf_matrix, diary_length, start_ts, n_agents, master_seed, slots_per_day=24))]
pub fn markov_diary_batch_generate<'py>(
    py: Python<'py>,
    cdf_matrix: Option<PyReadonlyArray1<'py, f64>>,
    diary_length: usize,
    start_ts: i64,
    n_agents: usize,
    master_seed: u64,
    slots_per_day: usize,
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
    let (flat_ts, flat_locs, starts, ends) = markov_diary_batch_generate_impl(
        cdf_slice,
        diary_length,
        start_ts,
        n_agents,
        master_seed,
        slots_per_day,
        slot_seconds_for(slots_per_day),
    );
    Ok((
        flat_ts.into_pyarray(py),
        flat_locs.into_pyarray(py),
        starts,
        ends,
    ))
}

#[pyfunction]
#[pyo3(signature = (cdf_matrix, diary_length, start_ts, seed, slots_per_day=24))]
pub fn markov_diary_generate<'py>(
    py: Python<'py>,
    cdf_matrix: PyReadonlyArray1<'py, f64>,
    diary_length: usize,
    start_ts: i64,
    seed: u64,
    slots_per_day: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i32>>)> {
    let matrix = cdf_matrix.as_slice()?;
    let (ts, locs) = markov_diary_generate_impl(
        matrix,
        diary_length,
        start_ts,
        seed,
        slots_per_day,
        slot_seconds_for(slots_per_day),
    );
    Ok((ts.into_pyarray(py), locs.into_pyarray(py)))
}
