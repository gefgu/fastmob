use fastmob_core::measures::individual::motifs::{
    canonical_adjacency_form as core_canonical_adjacency_form,
    compute_daily_motifs as core_compute_daily_motifs,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_bool_values, arrow_i64_values, arrow_u64_values, as_bool_array, as_i64_array,
    as_nullable_f64_array, as_u64_array, i64_results_into_arrow,
};

#[pyfunction]
pub fn canonical_adjacency_form(n_nodes: u32, edges: Vec<(u32, u32)>) -> PyResult<i64> {
    core_canonical_adjacency_form(n_nodes, edges).map_err(PyValueError::new_err)
}

type DailyMotifsPy<'py> = (Bound<'py, PyArray1<usize>>, ArrowPyArray, ArrowPyArray);

/// Every measure-data argument is a dense Arrow array (Arrow-only binding
/// convention: node identity and hours/dates/durations are all fixed-width
/// numeric/boolean, so there is no string data left to marshal through
/// PyO3's per-element list extraction). ``user_ends`` stays plain NumPy —
/// it is per-user boundary metadata, not measure data.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn compute_daily_motifs<'py>(
    py: Python<'py>,
    node_codes: ArrowPyArray,
    is_home_row: ArrowPyArray,
    start_hours: ArrowPyArray,
    end_hours: ArrowPyArray,
    date_ids: ArrowPyArray,
    durations: ArrowPyArray,
    user_ends: PyReadonlyArray1<'py, usize>,
    is_home_by_code: ArrowPyArray,
) -> PyResult<DailyMotifsPy<'py>> {
    let node_codes = as_u64_array(node_codes, "node_codes")?;
    let is_home_row = as_bool_array(is_home_row, "is_home_row")?;
    let start_hours = as_u64_array(start_hours, "start_hours")?;
    let end_hours = as_u64_array(end_hours, "end_hours")?;
    let date_ids = as_i64_array(date_ids, "date_ids")?;
    let durations = as_nullable_f64_array(durations, "durations")?;
    let is_home_by_code = as_bool_array(is_home_by_code, "is_home_by_code")?;

    let node_codes_values = arrow_u64_values(&node_codes);
    let is_home_row_values = arrow_bool_values(&is_home_row);
    let start_hours_values = arrow_u64_values(&start_hours);
    let end_hours_values = arrow_u64_values(&end_hours);
    let date_ids_values = arrow_i64_values(&date_ids);
    let durations_values: Vec<Option<f64>> = durations.iter().collect();
    let is_home_by_code_values = arrow_bool_values(&is_home_by_code);
    let user_ends_values = user_ends.as_slice()?;

    let (out_user_idx, out_date_ids, out_motif_ids) = py
        .detach(|| {
            core_compute_daily_motifs(
                node_codes_values,
                &is_home_row_values,
                start_hours_values,
                end_hours_values,
                date_ids_values,
                &durations_values,
                user_ends_values,
                &is_home_by_code_values,
            )
        })
        .map_err(PyValueError::new_err)?;

    Ok((
        out_user_idx.into_pyarray(py),
        i64_results_into_arrow(out_date_ids),
        i64_results_into_arrow(out_motif_ids),
    ))
}
