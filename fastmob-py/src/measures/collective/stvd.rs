use arrow_array::{Array, TimestampMicrosecondArray, UInt32Array};
use fastmob_core::measures::collective::stvd::mean_area_volume_impl;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{f64_results_into_arrow, u32_results_into_arrow};

fn optional_u32(array: ArrowPyArray, name: &str) -> PyResult<UInt32Array> {
    let (array_ref, _field) = array.into_inner();
    array_ref
        .as_any()
        .downcast_ref::<UInt32Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected uint32 Arrow array for {name}")))
}

fn optional_timestamp_us(array: ArrowPyArray, name: &str) -> PyResult<TimestampMicrosecondArray> {
    let (array_ref, _field) = array.into_inner();
    array_ref
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .cloned()
        .ok_or_else(|| {
            PyValueError::new_err(format!("expected timestamp[us] Arrow array for {name}"))
        })
}

/// Aggregate factorized staypoints into STVD rows without Python per-slot objects.
#[pyfunction]
pub fn mean_area_volume_arrow(
    py: Python<'_>,
    area_codes: ArrowPyArray,
    user_codes: ArrowPyArray,
    starts_us: ArrowPyArray,
    ends_us: ArrowPyArray,
) -> PyResult<(ArrowPyArray, ArrowPyArray, ArrowPyArray)> {
    let area_codes = optional_u32(area_codes, "area_codes")?;
    let user_codes = optional_u32(user_codes, "user_codes")?;
    let starts_us = optional_timestamp_us(starts_us, "starts_us")?;
    let ends_us = optional_timestamp_us(ends_us, "ends_us")?;
    let rows = py
        .detach(|| {
            let areas = (0..area_codes.len())
                .map(|index| area_codes.is_valid(index).then(|| area_codes.value(index)))
                .collect::<Vec<_>>();
            let users = (0..user_codes.len())
                .map(|index| user_codes.is_valid(index).then(|| user_codes.value(index)))
                .collect::<Vec<_>>();
            let starts = (0..starts_us.len())
                .map(|index| starts_us.is_valid(index).then(|| starts_us.value(index)))
                .collect::<Vec<_>>();
            let ends = (0..ends_us.len())
                .map(|index| ends_us.is_valid(index).then(|| ends_us.value(index)))
                .collect::<Vec<_>>();
            mean_area_volume_impl(&areas, &users, &starts, &ends)
        })
        .map_err(PyValueError::new_err)?;
    let mut areas = Vec::with_capacity(rows.len());
    let mut minutes = Vec::with_capacity(rows.len());
    let mut means = Vec::with_capacity(rows.len());
    for (area, minute, mean) in rows {
        areas.push(area);
        minutes.push(minute as u32);
        means.push(mean);
    }
    Ok((
        u32_results_into_arrow(areas),
        u32_results_into_arrow(minutes),
        f64_results_into_arrow(means),
    ))
}
