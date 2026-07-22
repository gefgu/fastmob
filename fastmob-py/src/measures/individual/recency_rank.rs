use fastmob_core::measures::individual::recency_rank::{
    recency_rank_indexed_values_impl, recency_rank_presorted_impl,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{
    arrow_valid_rows, arrow_values, as_nullable_f64_array, f64_results_into_arrow,
    u64_results_into_arrow,
};

type RecencyRankValues<'py> = (PyArray, PyArray, PyArray, Bound<'py, PyArray1<usize>>);

fn is_arrow_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.hasattr("__arrow_c_array__")
}

#[pyfunction]
pub fn recency_rank_values_indexed<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<RecencyRankValues<'py>> {
    if let (Ok(latitudes), Ok(longitudes)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let (out_lats, out_lngs, out_ranks, out_user_indices) = recency_rank_indexed_values_impl(
            latitudes.as_slice()?,
            longitudes.as_slice()?,
            indices.as_slice()?,
            ends.as_slice()?,
            None,
        )
        .map_err(PyValueError::new_err)?;
        return Ok((
            f64_results_into_arrow(out_lats),
            f64_results_into_arrow(out_lngs),
            u64_results_into_arrow(out_ranks),
            out_user_indices.into_pyarray(py),
        ));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? {
        let latitudes = as_nullable_f64_array(latitudes.extract::<PyArray>()?, "latitudes")?;
        let longitudes = as_nullable_f64_array(longitudes.extract::<PyArray>()?, "longitudes")?;
        let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
        let (out_lats, out_lngs, out_ranks, out_user_indices) = recency_rank_indexed_values_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            indices.as_slice()?,
            ends.as_slice()?,
            valid_rows.as_deref(),
        )
        .map_err(PyValueError::new_err)?;
        return Ok((
            f64_results_into_arrow(out_lats),
            f64_results_into_arrow(out_lngs),
            u64_results_into_arrow(out_ranks),
            out_user_indices.into_pyarray(py),
        ));
    }

    Err(PyTypeError::new_err(
        "latitudes and longitudes must both be NumPy arrays or both be Arrow arrays",
    ))
}

#[pyfunction]
pub fn recency_rank_presorted<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<RecencyRankValues<'py>> {
    if let (Ok(latitudes), Ok(longitudes)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let (out_lats, out_lngs, out_ranks, out_user_indices) = recency_rank_presorted_impl(
            latitudes.as_slice()?,
            longitudes.as_slice()?,
            ends.as_slice()?,
            None,
        )
        .map_err(PyValueError::new_err)?;
        return Ok((
            f64_results_into_arrow(out_lats),
            f64_results_into_arrow(out_lngs),
            u64_results_into_arrow(out_ranks),
            out_user_indices.into_pyarray(py),
        ));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? {
        let latitudes = as_nullable_f64_array(latitudes.extract::<PyArray>()?, "latitudes")?;
        let longitudes = as_nullable_f64_array(longitudes.extract::<PyArray>()?, "longitudes")?;
        let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
        let (out_lats, out_lngs, out_ranks, out_user_indices) = recency_rank_presorted_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            ends.as_slice()?,
            valid_rows.as_deref(),
        )
        .map_err(PyValueError::new_err)?;
        return Ok((
            f64_results_into_arrow(out_lats),
            f64_results_into_arrow(out_lngs),
            u64_results_into_arrow(out_ranks),
            out_user_indices.into_pyarray(py),
        ));
    }

    Err(PyTypeError::new_err(
        "latitudes and longitudes must both be NumPy arrays or both be Arrow arrays",
    ))
}
