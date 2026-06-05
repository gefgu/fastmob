use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use skmob2_core::measures::individual::recency_rank::recency_rank_indexed_impl;

use crate::utils::{arrow_valid_rows, arrow_values, as_nullable_f64_array, f64_results_into_arrow};

type RecencyRankNumpy<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
);

type RecencyRankArrow<'py> = (
    PyArray,
    PyArray,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
);

#[pyfunction]
pub fn recency_rank_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<RecencyRankNumpy<'py>> {
    let (out_lats, out_lngs, out_starts, out_ends) = recency_rank_indexed_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        indices.as_slice()?,
        ends.as_slice()?,
        None,
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        out_lats.into_pyarray(py),
        out_lngs.into_pyarray(py),
        out_starts.into_pyarray(py),
        out_ends.into_pyarray(py),
    ))
}

#[pyfunction]
pub fn recency_rank_indexed_arrow<'py>(
    py: Python<'py>,
    latitudes: PyArray,
    longitudes: PyArray,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<RecencyRankArrow<'py>> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    let (out_lats, out_lngs, out_starts, out_ends) = recency_rank_indexed_impl(
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
        out_starts.into_pyarray(py),
        out_ends.into_pyarray(py),
    ))
}
