use arrow_array::Array;
use fastmob_core::measures::individual::location_frequency::{
    frequency_rank_indexed_impl, frequency_rank_indexed_with_row_validity_impl,
    frequency_rank_presorted_impl, frequency_rank_presorted_with_row_validity_impl,
    location_frequency_indexed_values_impl,
    location_frequency_indexed_values_with_row_validity_impl,
    location_frequency_presorted_values_impl,
    location_frequency_presorted_values_with_row_validity_impl,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{
    arrow_values, as_nullable_f64_array, f64_results_into_arrow, u64_results_into_arrow,
};

type LocFreqValues<'py> = (
    PyArray,
    PyArray,
    PyArray,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    PyArray,
);

type FrequencyRank<'py> = (PyArray, PyArray, PyArray, Bound<'py, PyArray1<usize>>);

fn is_arrow_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.hasattr("__arrow_c_array__")
}

fn location_frequency_values_from_arrow<'py>(
    py: Python<'py>,
    latitudes: PyArray,
    longitudes: PyArray,
    indices: Option<PyReadonlyArray1<'py, usize>>,
    ends: PyReadonlyArray1<'py, usize>,
    normalize: bool,
) -> PyResult<LocFreqValues<'py>> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let lat_values = arrow_values(&latitudes);
    let lng_values = arrow_values(&longitudes);
    let ends = ends.as_slice()?;
    let lat_nulls = latitudes.nulls();
    let lng_nulls = longitudes.nulls();

    let (out_lats, out_lngs, out_values, out_user_indices, out_starts, out_ends, rank_means) =
        match indices {
            Some(indices) => {
                let indices = indices.as_slice()?;
                if lat_nulls.is_none() && lng_nulls.is_none() {
                    location_frequency_indexed_values_impl(
                        lat_values, lng_values, indices, ends, normalize, None,
                    )
                } else {
                    location_frequency_indexed_values_with_row_validity_impl(
                        lat_values,
                        lng_values,
                        indices,
                        ends,
                        normalize,
                        |row| {
                            lat_nulls.is_none_or(|nulls| nulls.is_valid(row))
                                && lng_nulls.is_none_or(|nulls| nulls.is_valid(row))
                        },
                    )
                }
            }
            None => {
                if lat_nulls.is_none() && lng_nulls.is_none() {
                    location_frequency_presorted_values_impl(
                        lat_values, lng_values, ends, normalize, None,
                    )
                } else {
                    location_frequency_presorted_values_with_row_validity_impl(
                        lat_values,
                        lng_values,
                        ends,
                        normalize,
                        |row| {
                            lat_nulls.is_none_or(|nulls| nulls.is_valid(row))
                                && lng_nulls.is_none_or(|nulls| nulls.is_valid(row))
                        },
                    )
                }
            }
        }
        .map_err(PyValueError::new_err)?;

    Ok((
        f64_results_into_arrow(out_lats),
        f64_results_into_arrow(out_lngs),
        f64_results_into_arrow(out_values),
        out_user_indices.into_pyarray(py),
        out_starts.into_pyarray(py),
        out_ends.into_pyarray(py),
        f64_results_into_arrow(rank_means),
    ))
}

fn frequency_rank_from_arrow<'py>(
    py: Python<'py>,
    latitudes: PyArray,
    longitudes: PyArray,
    indices: Option<PyReadonlyArray1<'py, usize>>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<FrequencyRank<'py>> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let lat_values = arrow_values(&latitudes);
    let lng_values = arrow_values(&longitudes);
    let ends = ends.as_slice()?;
    let lat_nulls = latitudes.nulls();
    let lng_nulls = longitudes.nulls();

    let (out_lats, out_lngs, out_ranks, out_user_indices) = match indices {
        Some(indices) => {
            let indices = indices.as_slice()?;
            if lat_nulls.is_none() && lng_nulls.is_none() {
                frequency_rank_indexed_impl(lat_values, lng_values, indices, ends, None)
            } else {
                frequency_rank_indexed_with_row_validity_impl(
                    lat_values,
                    lng_values,
                    indices,
                    ends,
                    |row| {
                        lat_nulls.is_none_or(|nulls| nulls.is_valid(row))
                            && lng_nulls.is_none_or(|nulls| nulls.is_valid(row))
                    },
                )
            }
        }
        None => {
            if lat_nulls.is_none() && lng_nulls.is_none() {
                frequency_rank_presorted_impl(lat_values, lng_values, ends, None)
            } else {
                frequency_rank_presorted_with_row_validity_impl(
                    lat_values,
                    lng_values,
                    ends,
                    |row| {
                        lat_nulls.is_none_or(|nulls| nulls.is_valid(row))
                            && lng_nulls.is_none_or(|nulls| nulls.is_valid(row))
                    },
                )
            }
        }
    }
    .map_err(PyValueError::new_err)?;

    Ok((
        f64_results_into_arrow(out_lats),
        f64_results_into_arrow(out_lngs),
        u64_results_into_arrow(out_ranks),
        out_user_indices.into_pyarray(py),
    ))
}

#[pyfunction]
pub fn location_frequency_values_indexed<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    normalize: bool,
) -> PyResult<LocFreqValues<'py>> {
    if let (Ok(latitudes), Ok(longitudes)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let (out_lats, out_lngs, out_values, out_user_indices, out_starts, out_ends, rank_means) =
            location_frequency_indexed_values_impl(
                latitudes.as_slice()?,
                longitudes.as_slice()?,
                indices.as_slice()?,
                ends.as_slice()?,
                normalize,
                None,
            )
            .map_err(PyValueError::new_err)?;
        return Ok((
            f64_results_into_arrow(out_lats),
            f64_results_into_arrow(out_lngs),
            f64_results_into_arrow(out_values),
            out_user_indices.into_pyarray(py),
            out_starts.into_pyarray(py),
            out_ends.into_pyarray(py),
            f64_results_into_arrow(rank_means),
        ));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? {
        return location_frequency_values_from_arrow(
            py,
            latitudes.extract::<PyArray>()?,
            longitudes.extract::<PyArray>()?,
            Some(indices),
            ends,
            normalize,
        );
    }

    Err(PyTypeError::new_err(
        "latitudes and longitudes must both be NumPy arrays or both be Arrow arrays",
    ))
}

#[pyfunction]
pub fn location_frequency_presorted<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
    normalize: bool,
) -> PyResult<LocFreqValues<'py>> {
    if let (Ok(latitudes), Ok(longitudes)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let (out_lats, out_lngs, out_values, out_user_indices, out_starts, out_ends, rank_means) =
            location_frequency_presorted_values_impl(
                latitudes.as_slice()?,
                longitudes.as_slice()?,
                ends.as_slice()?,
                normalize,
                None,
            )
            .map_err(PyValueError::new_err)?;
        return Ok((
            f64_results_into_arrow(out_lats),
            f64_results_into_arrow(out_lngs),
            f64_results_into_arrow(out_values),
            out_user_indices.into_pyarray(py),
            out_starts.into_pyarray(py),
            out_ends.into_pyarray(py),
            f64_results_into_arrow(rank_means),
        ));
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? {
        return location_frequency_values_from_arrow(
            py,
            latitudes.extract::<PyArray>()?,
            longitudes.extract::<PyArray>()?,
            None,
            ends,
            normalize,
        );
    }

    Err(PyTypeError::new_err(
        "latitudes and longitudes must both be NumPy arrays or both be Arrow arrays",
    ))
}

#[pyfunction]
pub fn frequency_rank_indexed<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<FrequencyRank<'py>> {
    if let (Ok(latitudes), Ok(longitudes)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let (out_lats, out_lngs, out_ranks, out_user_indices) = frequency_rank_indexed_impl(
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
        return frequency_rank_from_arrow(
            py,
            latitudes.extract::<PyArray>()?,
            longitudes.extract::<PyArray>()?,
            Some(indices),
            ends,
        );
    }

    Err(PyTypeError::new_err(
        "latitudes and longitudes must both be NumPy arrays or both be Arrow arrays",
    ))
}

#[pyfunction]
pub fn frequency_rank_presorted<'py>(
    py: Python<'py>,
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<FrequencyRank<'py>> {
    if let (Ok(latitudes), Ok(longitudes)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let (out_lats, out_lngs, out_ranks, out_user_indices) = frequency_rank_presorted_impl(
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
        return frequency_rank_from_arrow(
            py,
            latitudes.extract::<PyArray>()?,
            longitudes.extract::<PyArray>()?,
            None,
            ends,
        );
    }

    Err(PyTypeError::new_err(
        "latitudes and longitudes must both be NumPy arrays or both be Arrow arrays",
    ))
}
