use arrow_array::{Array, TimestampMicrosecondArray};
use fastmob_core::measures::fitting::daily_location_lognormal::daily_unique_location_histogram_impl;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::{measures::individual::factorization::factorize_array, utils::u64_results_into_arrow};

fn timestamp_days(array: ArrowPyArray) -> PyResult<(TimestampMicrosecondArray, Vec<i64>)> {
    let (array, _) = array.into_inner();
    let timestamps = array
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .cloned()
        .ok_or_else(|| {
            PyValueError::new_err("expected timestamp[us] Arrow array for timestamps")
        })?;
    let days = (0..timestamps.len())
        .map(|index| timestamps.value(index).div_euclid(86_400_000_000))
        .collect();
    Ok((timestamps, days))
}

/// Build the daily distinct-location histogram from Arrow columns.
#[pyfunction]
pub fn daily_unique_location_histogram_arrow(
    py: Python<'_>,
    user_ids: ArrowPyArray,
    location_ids: ArrowPyArray,
    timestamps: ArrowPyArray,
) -> PyResult<(ArrowPyArray, ArrowPyArray)> {
    let (user_ids, _) = user_ids.into_inner();
    let (location_ids, _) = location_ids.into_inner();
    let (timestamps, days) = timestamp_days(timestamps)?;
    let (counts, frequencies) = py
        .detach(|| {
            let (user_codes, _) = factorize_array(user_ids.as_ref(), false, false)?;
            let (location_codes, _) = factorize_array(location_ids.as_ref(), false, false)?;
            let valid = (0..days.len())
                .map(|index| {
                    user_ids.is_valid(index)
                        && location_ids.is_valid(index)
                        && timestamps.is_valid(index)
                })
                .collect::<Vec<_>>();
            daily_unique_location_histogram_impl(&user_codes, &location_codes, &days, &valid)
        })
        .map_err(PyValueError::new_err)?;
    Ok((
        u64_results_into_arrow(counts),
        u64_results_into_arrow(frequencies),
    ))
}
