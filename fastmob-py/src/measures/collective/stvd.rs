use arrow_array::{Array, ArrowPrimitiveType, PrimitiveArray, TimestampMicrosecondArray, UInt32Array};
use fastmob_core::measures::collective::stvd::mean_area_volume_impl;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{f64_results_into_arrow, u32_results_into_arrow};

fn nullable_u32(array: ArrowPyArray, name: &str) -> PyResult<UInt32Array> {
    let (array_ref, _field) = array.into_inner();
    array_ref
        .as_any()
        .downcast_ref::<UInt32Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected uint32 Arrow array for {name}")))
}

fn nullable_timestamp_us(array: ArrowPyArray, name: &str) -> PyResult<TimestampMicrosecondArray> {
    let (array_ref, _field) = array.into_inner();
    array_ref
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected timestamp[us] Arrow array for {name}")))
}

/// Zero-copy value slice for any dense primitive Arrow array, accounting for
/// a non-zero array offset -- the u32/timestamp analogue of
/// `crate::utils::arrow_values`, which only covers `Float64Type`.
fn arrow_values<T: ArrowPrimitiveType>(array: &PrimitiveArray<T>) -> &[T::Native] {
    let start = array.offset();
    let end = start + array.len();
    &array.values()[start..end]
}

/// Per-row validity across mixed array types, mirroring `crate::utils::arrow_valid_rows`
/// (which is monomorphic over `Float64Array`) via the shared `Array::is_valid` trait method.
fn arrow_valid_rows(arrays: &[&dyn Array]) -> Option<Vec<bool>> {
    if arrays.iter().all(|a| a.null_count() == 0) {
        return None;
    }
    let n = arrays[0].len();
    Some((0..n).map(|idx| arrays.iter().all(|a| a.is_valid(idx))).collect())
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
    let area_codes = nullable_u32(area_codes, "area_codes")?;
    let user_codes = nullable_u32(user_codes, "user_codes")?;
    let starts_us = nullable_timestamp_us(starts_us, "starts_us")?;
    let ends_us = nullable_timestamp_us(ends_us, "ends_us")?;

    let rows = py
        .detach(|| {
            let valid_rows = arrow_valid_rows(&[
                &area_codes as &dyn Array,
                &user_codes as &dyn Array,
                &starts_us as &dyn Array,
                &ends_us as &dyn Array,
            ]);
            mean_area_volume_impl(
                arrow_values(&area_codes),
                arrow_values(&user_codes),
                arrow_values(&starts_us),
                arrow_values(&ends_us),
                valid_rows.as_deref(),
            )
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
