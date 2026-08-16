use arrow_array::{
    Array, TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
    TimestampSecondArray,
};
use arrow_schema::{DataType, TimeUnit};
use fastmob_core::measures::individual::time_ordering::{
    presorted_ranges_for_u32_codes, split_ordered_index_ranges, time_ordered_indices_for_u32_codes,
    time_ordered_indices_single_user, validate_grouped_u32_codes,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{arrow_u32_values, as_u32_array, extract_arrow_array, u64_results_into_arrow};

enum TimestampValues {
    Second(TimestampSecondArray),
    Millisecond(TimestampMillisecondArray),
    Microsecond(TimestampMicrosecondArray),
    Nanosecond(TimestampNanosecondArray),
    Normalized(Vec<i64>),
}

impl TimestampValues {
    fn values(&self) -> &[i64] {
        match self {
            Self::Second(array) => array.values(),
            Self::Millisecond(array) => array.values(),
            Self::Microsecond(array) => array.values(),
            Self::Nanosecond(array) => array.values(),
            Self::Normalized(values) => values,
        }
    }
}

fn timestamp_values(array: PyArray, name: &str) -> PyResult<TimestampValues> {
    let (array_ref, _field) = array.into_inner();
    macro_rules! timestamp_array {
        ($ty:ty, $variant:ident) => {{
            let array = array_ref
                .as_any()
                .downcast_ref::<$ty>()
                .expect("timestamp datatype and array type must match")
                .clone();
            if array.null_count() == 0 {
                Ok(TimestampValues::$variant(array))
            } else {
                let mut values = array.values().to_vec();
                for (idx, value) in values.iter_mut().enumerate() {
                    if array.is_null(idx) {
                        *value = i64::MIN;
                    }
                }
                Ok(TimestampValues::Normalized(values))
            }
        }};
    }

    match array_ref.data_type() {
        DataType::Timestamp(TimeUnit::Second, _) => timestamp_array!(TimestampSecondArray, Second),
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            timestamp_array!(TimestampMillisecondArray, Millisecond)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            timestamp_array!(TimestampMicrosecondArray, Microsecond)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            timestamp_array!(TimestampNanosecondArray, Nanosecond)
        }
        _ => Err(PyValueError::new_err(format!(
            "expected timestamp Arrow array for {name}"
        ))),
    }
}

fn index_results(values: Vec<u64>) -> PyArray {
    u64_results_into_arrow(values)
}

#[pyfunction]
pub fn presorted_user_starts_ends(
    py: Python<'_>,
    uids: &Bound<'_, PyAny>,
) -> PyResult<(PyArray, PyArray)> {
    let uids = as_u32_array(extract_arrow_array(uids, "uids")?, "uids")?;
    let (starts, ends) = py.detach(|| presorted_ranges_for_u32_codes(arrow_u32_values(&uids)));
    Ok((index_results(starts), index_results(ends)))
}

#[pyfunction]
#[pyo3(signature = (uids, timestamps, check_timestamps = true))]
pub fn validate_presorted_user_timestamps(
    py: Python<'_>,
    uids: &Bound<'_, PyAny>,
    timestamps: Option<&Bound<'_, PyAny>>,
    check_timestamps: bool,
) -> PyResult<bool> {
    let uids = as_u32_array(extract_arrow_array(uids, "uids")?, "uids")?;
    let timestamps = timestamps
        .map(|values| timestamp_values(extract_arrow_array(values, "timestamps")?, "timestamps"))
        .transpose()?;
    Ok(py.detach(|| {
        validate_grouped_u32_codes(
            arrow_u32_values(&uids),
            timestamps.as_ref().map(TimestampValues::values),
            check_timestamps,
        )
    }))
}

#[pyfunction]
#[pyo3(signature = (uids, timestamps = None, check_timestamps = true))]
pub fn validate_grouped_user_timestamps(
    py: Python<'_>,
    uids: &Bound<'_, PyAny>,
    timestamps: Option<&Bound<'_, PyAny>>,
    check_timestamps: bool,
) -> PyResult<bool> {
    let uids = as_u32_array(extract_arrow_array(uids, "uids")?, "uids")?;
    let timestamps = timestamps
        .map(|values| timestamp_values(extract_arrow_array(values, "timestamps")?, "timestamps"))
        .transpose()?;
    Ok(py.detach(|| {
        validate_grouped_u32_codes(
            arrow_u32_values(&uids),
            timestamps.as_ref().map(TimestampValues::values),
            check_timestamps,
        )
    }))
}

#[pyfunction]
#[pyo3(signature = (uids, timestamps, num_groups = None))]
pub fn time_ordered_user_indices(
    py: Python<'_>,
    uids: Option<&Bound<'_, PyAny>>,
    timestamps: &Bound<'_, PyAny>,
    num_groups: Option<usize>,
) -> PyResult<(PyArray, PyArray)> {
    let timestamps =
        timestamp_values(extract_arrow_array(timestamps, "timestamps")?, "timestamps")?;
    let ordered = if let Some(uids) = uids {
        let uids = as_u32_array(extract_arrow_array(uids, "uids")?, "uids")?;
        let num_groups = num_groups.ok_or_else(|| {
            PyValueError::new_err("num_groups is required when uids are provided")
        })?;
        py.detach(|| {
            time_ordered_indices_for_u32_codes(
                arrow_u32_values(&uids),
                timestamps.values(),
                num_groups,
            )
        })
        .map_err(PyValueError::new_err)?
    } else {
        py.detach(|| time_ordered_indices_single_user(timestamps.values()))
    };
    let (indices, ends) = split_ordered_index_ranges(ordered);
    Ok((index_results(indices), index_results(ends)))
}
