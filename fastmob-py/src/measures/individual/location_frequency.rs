use crate::utils::ArrowUsizeArrayExt;
use arrow_array::Array;
use fastmob_core::measures::individual::location_frequency::{
    frequency_rank_indexed_impl, frequency_rank_indexed_with_row_validity_impl,
    frequency_rank_presorted_impl, frequency_rank_presorted_with_row_validity_impl,
    location_frequency_indexed_values_impl,
    location_frequency_indexed_values_with_row_validity_impl,
    location_frequency_presorted_values_impl,
    location_frequency_presorted_values_with_row_validity_impl,
};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_values, as_nullable_f64_array, f64_results_into_arrow, u64_results_into_arrow,
};

type LocFreqValues<'py> = (
    ArrowPyArray,
    ArrowPyArray,
    ArrowPyArray,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<u64>>,
    ArrowPyArray,
);

type FrequencyRank<'py> = (
    ArrowPyArray,
    ArrowPyArray,
    ArrowPyArray,
    Bound<'py, PyArray1<u64>>,
);

fn location_frequency_values_from_arrow<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    indices: Option<pyo3_arrow::PyArray>,
    ends: pyo3_arrow::PyArray,
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
                        lat_values, lng_values, &indices, &ends, normalize, None,
                    )
                } else {
                    location_frequency_indexed_values_with_row_validity_impl(
                        lat_values,
                        lng_values,
                        &indices,
                        &ends,
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
                        lat_values, lng_values, &ends, normalize, None,
                    )
                } else {
                    location_frequency_presorted_values_with_row_validity_impl(
                        lat_values,
                        lng_values,
                        &ends,
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

    let to_u64 = |values: Vec<usize>| {
        values
            .into_iter()
            .map(|value| value as u64)
            .collect::<Vec<_>>()
    };
    Ok((
        f64_results_into_arrow(out_lats),
        f64_results_into_arrow(out_lngs),
        f64_results_into_arrow(out_values),
        to_u64(out_user_indices).into_pyarray(py),
        to_u64(out_starts).into_pyarray(py),
        to_u64(out_ends).into_pyarray(py),
        f64_results_into_arrow(rank_means),
    ))
}

fn frequency_rank_from_arrow<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    indices: Option<pyo3_arrow::PyArray>,
    ends: pyo3_arrow::PyArray,
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
                frequency_rank_indexed_impl(lat_values, lng_values, &indices, &ends, None)
            } else {
                frequency_rank_indexed_with_row_validity_impl(
                    lat_values,
                    lng_values,
                    &indices,
                    &ends,
                    |row| {
                        lat_nulls.is_none_or(|nulls| nulls.is_valid(row))
                            && lng_nulls.is_none_or(|nulls| nulls.is_valid(row))
                    },
                )
            }
        }
        None => {
            if lat_nulls.is_none() && lng_nulls.is_none() {
                frequency_rank_presorted_impl(lat_values, lng_values, &ends, None)
            } else {
                frequency_rank_presorted_with_row_validity_impl(
                    lat_values,
                    lng_values,
                    &ends,
                    |row| {
                        lat_nulls.is_none_or(|nulls| nulls.is_valid(row))
                            && lng_nulls.is_none_or(|nulls| nulls.is_valid(row))
                    },
                )
            }
        }
    }
    .map_err(PyValueError::new_err)?;

    let out_user_indices = out_user_indices
        .into_iter()
        .map(|value| value as u64)
        .collect::<Vec<_>>();
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
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
    normalize: bool,
) -> PyResult<LocFreqValues<'py>> {
    location_frequency_values_from_arrow(py, latitudes, longitudes, Some(indices), ends, normalize)
}

#[pyfunction]
pub fn location_frequency_presorted<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
    normalize: bool,
) -> PyResult<LocFreqValues<'py>> {
    location_frequency_values_from_arrow(py, latitudes, longitudes, None, ends, normalize)
}

#[pyfunction]
pub fn frequency_rank_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
) -> PyResult<FrequencyRank<'py>> {
    frequency_rank_from_arrow(py, latitudes, longitudes, Some(indices), ends)
}

#[pyfunction]
pub fn frequency_rank_presorted<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
) -> PyResult<FrequencyRank<'py>> {
    frequency_rank_from_arrow(py, latitudes, longitudes, None, ends)
}
