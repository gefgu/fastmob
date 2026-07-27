use fastmob_core::measures::individual::recency_rank::{
    recency_rank_indexed_values_impl, recency_rank_presorted_impl,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_valid_rows, arrow_values, as_nullable_f64_array, f64_results_into_arrow,
    u64_results_into_arrow,
};

type RecencyRankValues<'py> = (
    ArrowPyArray,
    ArrowPyArray,
    ArrowPyArray,
    Bound<'py, PyArray1<usize>>,
);

#[pyfunction]
pub fn recency_rank_values_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<RecencyRankValues<'py>> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    let (out_lats, out_lngs, out_ranks, out_user_indices) = recency_rank_indexed_values_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        indices.as_slice()?,
        ends.as_slice()?,
        valid_rows.as_deref(),
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        f64_results_into_arrow(out_lats),
        f64_results_into_arrow(out_lngs),
        u64_results_into_arrow(out_ranks),
        out_user_indices.into_pyarray(py),
    ))
}

#[pyfunction]
pub fn recency_rank_presorted<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<RecencyRankValues<'py>> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    let (out_lats, out_lngs, out_ranks, out_user_indices) = recency_rank_presorted_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        ends.as_slice()?,
        valid_rows.as_deref(),
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        f64_results_into_arrow(out_lats),
        f64_results_into_arrow(out_lngs),
        u64_results_into_arrow(out_ranks),
        out_user_indices.into_pyarray(py),
    ))
}
