use arrow_array::types::{
    Date32Type, Date64Type, Int8Type, Int16Type, Int32Type, Int64Type, TimestampMicrosecondType,
    TimestampMillisecondType, TimestampNanosecondType, TimestampSecondType, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow_array::{
    Array, PrimitiveArray, TimestampMicrosecondArray, TimestampMillisecondArray,
    TimestampNanosecondArray, TimestampSecondArray,
};
use arrow_schema::{DataType, TimeUnit};
use fastmob_core::measures::individual::time_ordering::{
    presorted_ranges_for_u32_codes, run_boundaries, split_ordered_index_ranges,
    time_ordered_indices_for_u32_codes, time_ordered_indices_single_user,
    validate_grouped_u32_codes, validate_non_decreasing_timestamps,
    validate_non_decreasing_within_ends,
};
use fastmob_core::utils::all_finite;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{
    arrow_u32_values, arrow_u64_values, as_nullable_f64_array, as_u32_array, as_u64_array,
    extract_arrow_array, u64_results_into_arrow,
};

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

/// Validate that rows are grouped by uid -- and, when `check_timestamps`, in
/// non-decreasing time order within each group.
///
/// `uids` may be `None`, meaning the frame has no uid column and is therefore a
/// single individual: grouping is vacuous and only the timestamp order matters.
#[pyfunction]
#[pyo3(signature = (uids = None, timestamps = None, check_timestamps = true))]
pub fn validate_grouped_user_timestamps(
    py: Python<'_>,
    uids: Option<&Bound<'_, PyAny>>,
    timestamps: Option<&Bound<'_, PyAny>>,
    check_timestamps: bool,
) -> PyResult<bool> {
    let uids = uids
        .map(|values| as_u32_array(extract_arrow_array(values, "uids")?, "uids"))
        .transpose()?;
    let timestamps = timestamps
        .map(|values| timestamp_values(extract_arrow_array(values, "timestamps")?, "timestamps"))
        .transpose()?;
    let timestamps = timestamps.as_ref().map(TimestampValues::values);

    Ok(py.detach(|| match uids {
        Some(uids) => {
            validate_grouped_u32_codes(arrow_u32_values(&uids), timestamps, check_timestamps)
        }
        None if !check_timestamps => true,
        None => match timestamps {
            Some(timestamps) => validate_non_decreasing_timestamps(timestamps),
            None => false,
        },
    }))
}

/// Start/end offsets of each maximal run of equal uid values, or `None` when
/// the column's type is not one this can scan directly.
///
/// Only fixed-width primitive columns with no nulls are handled: those are a
/// flat buffer whose runs are found by comparing adjacent elements, which is
/// what makes this an order of magnitude cheaper than `run_end_encode`.
/// Anything else (strings, dictionaries, a column with nulls) returns `None`
/// so the caller can fall back rather than get a subtly different grouping.
#[pyfunction]
pub fn value_run_boundaries(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
) -> PyResult<Option<(PyArray, PyArray)>> {
    let (array, _field) = extract_arrow_array(values, "values")?.into_inner();
    if array.null_count() != 0 {
        return Ok(None);
    }

    macro_rules! boundaries_for {
        ($($arrow_type:ty),+ $(,)?) => {
            $(
                if let Some(typed) = array.as_any().downcast_ref::<PrimitiveArray<$arrow_type>>() {
                    let values = typed.values();
                    let (starts, ends) = py.detach(|| run_boundaries(values));
                    return Ok(Some((index_results(starts), index_results(ends))));
                }
            )+
        };
    }

    boundaries_for!(
        Int8Type,
        Int16Type,
        Int32Type,
        Int64Type,
        UInt8Type,
        UInt16Type,
        UInt32Type,
        UInt64Type,
        Date32Type,
        Date64Type,
        TimestampSecondType,
        TimestampMillisecondType,
        TimestampMicrosecondType,
        TimestampNanosecondType,
    );
    Ok(None)
}

/// Validate that timestamps rise within every run delimited by `ends`.
///
/// The counterpart to `validate_grouped_user_timestamps` for callers that
/// already hold run boundaries -- deriving them from a `run_end_encode` scan is
/// cheaper than dense-coding the uid column when the answer turns out to be yes.
#[pyfunction]
pub fn validate_timestamps_within_ends(
    py: Python<'_>,
    timestamps: &Bound<'_, PyAny>,
    ends: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let timestamps =
        timestamp_values(extract_arrow_array(timestamps, "timestamps")?, "timestamps")?;
    let ends = as_u64_array(extract_arrow_array(ends, "ends")?, "ends")?;
    Ok(py.detach(|| {
        validate_non_decreasing_within_ends(timestamps.values(), arrow_u64_values(&ends))
    }))
}

/// Do the coordinate columns hold only finite, non-null values?
///
/// The other half of the presorted fast path's precondition: those kernels sum
/// coordinates with no per-row validity test, so they may only be used on data
/// the `indexed` kernels would not have filtered.
#[pyfunction]
pub fn coordinates_all_finite(
    py: Python<'_>,
    latitudes: &Bound<'_, PyAny>,
    longitudes: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let latitudes =
        as_nullable_f64_array(extract_arrow_array(latitudes, "latitudes")?, "latitudes")?;
    let longitudes =
        as_nullable_f64_array(extract_arrow_array(longitudes, "longitudes")?, "longitudes")?;
    if latitudes.null_count() != 0 || longitudes.null_count() != 0 {
        return Ok(false);
    }
    Ok(py.detach(|| all_finite(latitudes.values()) && all_finite(longitudes.values())))
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
