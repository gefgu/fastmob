use arrow_array::Array;
use fastmob_core::measures::individual::location_frequency::{
    frequency_rank_indexed_impl, frequency_rank_indexed_with_row_validity_impl,
    frequency_rank_presorted_impl, frequency_rank_presorted_with_row_validity_impl,
    location_frequency_indexed_impl, location_frequency_indexed_values_impl,
    location_frequency_indexed_values_with_row_validity_impl,
    location_frequency_indexed_with_row_validity_impl, location_frequency_presorted_values_impl,
    location_frequency_presorted_values_with_row_validity_impl,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{
    arrow_values, as_nullable_f64_array, f64_results_into_arrow, u64_results_into_arrow,
};

type LocFreqNumpy<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
);

type LocFreqArrow<'py> = (
    PyArray,
    PyArray,
    PyArray,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
);

type LocFreqValuesNumpy<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<f64>>,
);

type LocFreqValuesArrow<'py> = (
    PyArray,
    PyArray,
    PyArray,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    PyArray,
);

type FrequencyRankNumpy<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<usize>>,
);

type FrequencyRankArrow<'py> = (PyArray, PyArray, PyArray, Bound<'py, PyArray1<usize>>);

#[pyfunction]
pub fn location_frequency_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<LocFreqNumpy<'py>> {
    let (out_lats, out_lngs, out_counts, out_starts, out_ends) = location_frequency_indexed_impl(
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
        out_counts.into_pyarray(py),
        out_starts.into_pyarray(py),
        out_ends.into_pyarray(py),
    ))
}

#[pyfunction]
pub fn location_frequency_indexed_arrow<'py>(
    py: Python<'py>,
    latitudes: PyArray,
    longitudes: PyArray,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<LocFreqArrow<'py>> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let lat_values = arrow_values(&latitudes);
    let lng_values = arrow_values(&longitudes);
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;
    let lat_nulls = latitudes.nulls();
    let lng_nulls = longitudes.nulls();
    let (out_lats, out_lngs, out_counts, out_starts, out_ends) =
        if lat_nulls.is_none() && lng_nulls.is_none() {
            location_frequency_indexed_impl(lat_values, lng_values, indices, ends, None)
        } else {
            location_frequency_indexed_with_row_validity_impl(
                lat_values,
                lng_values,
                indices,
                ends,
                |idx| {
                    lat_nulls.is_none_or(|nulls| nulls.is_valid(idx))
                        && lng_nulls.is_none_or(|nulls| nulls.is_valid(idx))
                },
            )
        }
        .map_err(PyValueError::new_err)?;
    Ok((
        f64_results_into_arrow(out_lats),
        f64_results_into_arrow(out_lngs),
        u64_results_into_arrow(out_counts),
        out_starts.into_pyarray(py),
        out_ends.into_pyarray(py),
    ))
}

#[pyfunction]
pub fn location_frequency_values_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    normalize: bool,
) -> PyResult<LocFreqValuesNumpy<'py>> {
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
    Ok((
        out_lats.into_pyarray(py),
        out_lngs.into_pyarray(py),
        out_values.into_pyarray(py),
        out_user_indices.into_pyarray(py),
        out_starts.into_pyarray(py),
        out_ends.into_pyarray(py),
        rank_means.into_pyarray(py),
    ))
}

#[pyfunction]
pub fn location_frequency_values_indexed_arrow<'py>(
    py: Python<'py>,
    latitudes: PyArray,
    longitudes: PyArray,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    normalize: bool,
) -> PyResult<LocFreqValuesArrow<'py>> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let lat_values = arrow_values(&latitudes);
    let lng_values = arrow_values(&longitudes);
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;
    let lat_nulls = latitudes.nulls();
    let lng_nulls = longitudes.nulls();
    let (out_lats, out_lngs, out_values, out_user_indices, out_starts, out_ends, rank_means) =
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
                |idx| {
                    lat_nulls.is_none_or(|nulls| nulls.is_valid(idx))
                        && lng_nulls.is_none_or(|nulls| nulls.is_valid(idx))
                },
            )
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

#[pyfunction]
pub fn location_frequency_presorted_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    ends: PyReadonlyArray1<'py, usize>,
    normalize: bool,
) -> PyResult<LocFreqValuesNumpy<'py>> {
    let (out_lats, out_lngs, out_values, out_user_indices, out_starts, out_ends, rank_means) =
        location_frequency_presorted_values_impl(
            latitudes.as_slice()?,
            longitudes.as_slice()?,
            ends.as_slice()?,
            normalize,
            None,
        )
        .map_err(PyValueError::new_err)?;
    Ok((
        out_lats.into_pyarray(py),
        out_lngs.into_pyarray(py),
        out_values.into_pyarray(py),
        out_user_indices.into_pyarray(py),
        out_starts.into_pyarray(py),
        out_ends.into_pyarray(py),
        rank_means.into_pyarray(py),
    ))
}

#[pyfunction]
pub fn location_frequency_presorted_arrow<'py>(
    py: Python<'py>,
    latitudes: PyArray,
    longitudes: PyArray,
    ends: PyReadonlyArray1<'py, usize>,
    normalize: bool,
) -> PyResult<LocFreqValuesArrow<'py>> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let lat_values = arrow_values(&latitudes);
    let lng_values = arrow_values(&longitudes);
    let ends = ends.as_slice()?;
    let lat_nulls = latitudes.nulls();
    let lng_nulls = longitudes.nulls();
    let (out_lats, out_lngs, out_values, out_user_indices, out_starts, out_ends, rank_means) =
        if lat_nulls.is_none() && lng_nulls.is_none() {
            location_frequency_presorted_values_impl(lat_values, lng_values, ends, normalize, None)
        } else {
            location_frequency_presorted_values_with_row_validity_impl(
                lat_values,
                lng_values,
                ends,
                normalize,
                |idx| {
                    lat_nulls.is_none_or(|nulls| nulls.is_valid(idx))
                        && lng_nulls.is_none_or(|nulls| nulls.is_valid(idx))
                },
            )
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

#[pyfunction]
pub fn frequency_rank_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<FrequencyRankNumpy<'py>> {
    let (out_lats, out_lngs, out_ranks, out_user_indices) = frequency_rank_indexed_impl(
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
        out_ranks.into_pyarray(py),
        out_user_indices.into_pyarray(py),
    ))
}

#[pyfunction]
pub fn frequency_rank_indexed_arrow<'py>(
    py: Python<'py>,
    latitudes: PyArray,
    longitudes: PyArray,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<FrequencyRankArrow<'py>> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let lat_values = arrow_values(&latitudes);
    let lng_values = arrow_values(&longitudes);
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;
    let lat_nulls = latitudes.nulls();
    let lng_nulls = longitudes.nulls();
    let (out_lats, out_lngs, out_ranks, out_user_indices) =
        if lat_nulls.is_none() && lng_nulls.is_none() {
            frequency_rank_indexed_impl(lat_values, lng_values, indices, ends, None)
        } else {
            frequency_rank_indexed_with_row_validity_impl(
                lat_values,
                lng_values,
                indices,
                ends,
                |idx| {
                    lat_nulls.is_none_or(|nulls| nulls.is_valid(idx))
                        && lng_nulls.is_none_or(|nulls| nulls.is_valid(idx))
                },
            )
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
pub fn frequency_rank_presorted_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<FrequencyRankNumpy<'py>> {
    let (out_lats, out_lngs, out_ranks, out_user_indices) = frequency_rank_presorted_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        ends.as_slice()?,
        None,
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        out_lats.into_pyarray(py),
        out_lngs.into_pyarray(py),
        out_ranks.into_pyarray(py),
        out_user_indices.into_pyarray(py),
    ))
}

#[pyfunction]
pub fn frequency_rank_presorted_arrow<'py>(
    py: Python<'py>,
    latitudes: PyArray,
    longitudes: PyArray,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<FrequencyRankArrow<'py>> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let lat_values = arrow_values(&latitudes);
    let lng_values = arrow_values(&longitudes);
    let ends = ends.as_slice()?;
    let lat_nulls = latitudes.nulls();
    let lng_nulls = longitudes.nulls();
    let (out_lats, out_lngs, out_ranks, out_user_indices) =
        if lat_nulls.is_none() && lng_nulls.is_none() {
            frequency_rank_presorted_impl(lat_values, lng_values, ends, None)
        } else {
            frequency_rank_presorted_with_row_validity_impl(lat_values, lng_values, ends, |idx| {
                lat_nulls.is_none_or(|nulls| nulls.is_valid(idx))
                    && lng_nulls.is_none_or(|nulls| nulls.is_valid(idx))
            })
        }
        .map_err(PyValueError::new_err)?;
    Ok((
        f64_results_into_arrow(out_lats),
        f64_results_into_arrow(out_lngs),
        u64_results_into_arrow(out_ranks),
        out_user_indices.into_pyarray(py),
    ))
}
