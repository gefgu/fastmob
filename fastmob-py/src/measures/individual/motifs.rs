use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, Float64Array, LargeStringArray, StringArray, StringViewArray,
    TimestampMicrosecondArray, UInt16Array, UInt32Array,
};
use fastmob_core::measures::individual::motifs::{
    canonical_adjacency_form as core_canonical_adjacency_form,
    compute_daily_motifs_indexed as core_compute_daily_motifs_indexed,
    compute_daily_motifs_indexed_joined as core_compute_daily_motifs_indexed_joined,
    compute_daily_motifs_presorted as core_compute_daily_motifs_presorted,
    compute_daily_motifs_presorted_joined as core_compute_daily_motifs_presorted_joined,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::{PyArray as ArrowPyArray, PyChunkedArray};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::utils::{
    arrow_u32_values, arrow_usize_values, arrow_values, as_f64_array,
    as_u64_array,
    i32_results_into_arrow, i64_results_into_arrow, u64_results_into_arrow,
};

#[pyfunction]
pub fn canonical_adjacency_form(n_nodes: u32, edges: Vec<(u32, u32)>) -> PyResult<i64> {
    core_canonical_adjacency_form(n_nodes, edges).map_err(PyValueError::new_err)
}

type DailyMotifsPy = (ArrowPyArray, ArrowPyArray, ArrowPyArray);

enum PurposeChunk {
    Utf8(StringArray),
    LargeUtf8(LargeStringArray),
    Utf8View(StringViewArray),
}

impl PurposeChunk {
    fn try_new(array: &ArrayRef) -> PyResult<Self> {
        if let Some(values) = array.as_any().downcast_ref::<StringArray>() {
            return Ok(Self::Utf8(values.clone()));
        }
        if let Some(values) = array.as_any().downcast_ref::<LargeStringArray>() {
            return Ok(Self::LargeUtf8(values.clone()));
        }
        if let Some(values) = array.as_any().downcast_ref::<StringViewArray>() {
            return Ok(Self::Utf8View(values.clone()));
        }
        Err(PyValueError::new_err(format!(
            "expected string Arrow chunks for purposes, got {}",
            array.data_type()
        )))
    }

    fn len(&self) -> usize {
        match self {
            Self::Utf8(values) => values.len(),
            Self::LargeUtf8(values) => values.len(),
            Self::Utf8View(values) => values.len(),
        }
    }

    fn discover(&self, offset: usize) -> (FxHashMap<String, usize>, Option<usize>) {
        match self {
            Self::Utf8(values) => discover_purposes(values.len(), offset, |index| {
                values.is_valid(index).then(|| values.value(index))
            }),
            Self::LargeUtf8(values) => discover_purposes(values.len(), offset, |index| {
                values.is_valid(index).then(|| values.value(index))
            }),
            Self::Utf8View(values) => discover_purposes(values.len(), offset, |index| {
                values.is_valid(index).then(|| values.value(index))
            }),
        }
    }

    fn encode(&self, mapping: &FxHashMap<String, u16>, null_code: Option<u16>) -> Vec<u16> {
        match self {
            Self::Utf8(values) => encode_purpose_chunk(values.len(), mapping, null_code, |index| {
                values.is_valid(index).then(|| values.value(index))
            }),
            Self::LargeUtf8(values) => {
                encode_purpose_chunk(values.len(), mapping, null_code, |index| {
                    values.is_valid(index).then(|| values.value(index))
                })
            }
            Self::Utf8View(values) => {
                encode_purpose_chunk(values.len(), mapping, null_code, |index| {
                    values.is_valid(index).then(|| values.value(index))
                })
            }
        }
    }
}

fn discover_purposes<'a, F>(
    len: usize,
    offset: usize,
    value_at: F,
) -> (FxHashMap<String, usize>, Option<usize>)
where
    F: Fn(usize) -> Option<&'a str> + Sync,
{
    (0..len)
        .into_par_iter()
        .fold(
            || (FxHashMap::default(), None),
            |(mut values, null_index), index| {
                if let Some(value) = value_at(index) {
                    values
                        .entry(value.to_owned())
                        .and_modify(|first: &mut usize| *first = (*first).min(offset + index))
                        .or_insert(offset + index);
                    (values, null_index)
                } else {
                    (
                        values,
                        Some(
                            null_index
                                .map_or(offset + index, |first: usize| first.min(offset + index)),
                        ),
                    )
                }
            },
        )
        .reduce(
            || (FxHashMap::default(), None),
            |(mut left, left_null), (right, right_null)| {
                for (value, index) in right {
                    left.entry(value)
                        .and_modify(|first| *first = (*first).min(index))
                        .or_insert(index);
                }
                let null_index = match (left_null, right_null) {
                    (Some(left), Some(right)) => Some(left.min(right)),
                    (left, right) => left.or(right),
                };
                (left, null_index)
            },
        )
}

fn encode_purpose_chunk<'a, F>(
    len: usize,
    mapping: &FxHashMap<String, u16>,
    null_code: Option<u16>,
    value_at: F,
) -> Vec<u16>
where
    F: Fn(usize) -> Option<&'a str> + Sync,
{
    (0..len)
        .into_par_iter()
        .map(|index| match value_at(index) {
            Some(value) => mapping[value],
            None => null_code.expect("null purpose code must exist"),
        })
        .collect()
}

#[pyfunction]
pub fn encode_motif_purposes(values: PyChunkedArray) -> PyResult<(ArrowPyArray, u16, u16)> {
    let chunks: Vec<PurposeChunk> = values
        .chunks()
        .iter()
        .map(PurposeChunk::try_new)
        .collect::<PyResult<_>>()?;
    let mut unique_values: FxHashMap<String, usize> = FxHashMap::default();
    let mut null_index: Option<usize> = None;
    let mut offset = 0;
    for chunk in &chunks {
        let (chunk_values, chunk_null) = chunk.discover(offset);
        for (value, index) in chunk_values {
            unique_values
                .entry(value)
                .and_modify(|first| *first = (*first).min(index))
                .or_insert(index);
        }
        null_index = match (null_index, chunk_null) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (left, right) => left.or(right),
        };
        offset += chunk.len();
    }

    let mut groups: Vec<(Option<String>, usize)> = unique_values
        .into_iter()
        .map(|(value, index)| (Some(value), index))
        .collect();
    if let Some(index) = null_index {
        groups.push((None, index));
    }
    groups.sort_unstable_by_key(|(_, index)| *index);
    if groups.len() >= u16::MAX as usize {
        return Err(PyValueError::new_err(
            "motif purpose encoding supports at most 65,534 distinct values",
        ));
    }
    // One past the highest real/null code ever assigned (0..groups.len()) --
    // used by the Staypoints/Locations hierarchy lookup to mark a
    // (user_idx, location_code) miss, distinct from both real codes and
    // from home_code's u16::MAX "HOME not found" sentinel
    // (groups.len() <= 65533 < 65535).
    let unmatched_code = groups.len() as u16;

    let mut mapping = FxHashMap::default();
    let mut null_code = None;
    for (code, (value, _)) in groups.into_iter().enumerate() {
        if let Some(value) = value {
            mapping.insert(value, code as u16);
        } else {
            null_code = Some(code as u16);
        }
    }
    let home_code = mapping.get("HOME").copied().unwrap_or(u16::MAX);
    let mut codes = Vec::with_capacity(offset);
    for chunk in &chunks {
        codes.extend(chunk.encode(&mapping, null_code));
    }
    Ok((
        ArrowPyArray::from_array_ref(Arc::new(UInt16Array::from(codes))),
        home_code,
        unmatched_code,
    ))
}

fn as_u16_array(arr: ArrowPyArray, name: &str) -> PyResult<UInt16Array> {
    let (array_ref, _field) = arr.into_inner();
    let array = array_ref
        .as_any()
        .downcast_ref::<UInt16Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected uint16 Arrow array for {name}")))?;
    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }
    Ok(array)
}

fn u16_values(array: &UInt16Array) -> &[u16] {
    let start = array.offset();
    let end = start + array.len();
    &array.values()[start..end]
}

fn as_u32_array(arr: ArrowPyArray, name: &str) -> PyResult<UInt32Array> {
    let (array_ref, _field) = arr.into_inner();
    let array = array_ref
        .as_any()
        .downcast_ref::<UInt32Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected uint32 Arrow array for {name}")))?;
    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }
    Ok(array)
}

fn u32_values(array: &UInt32Array) -> &[u32] {
    let start = array.offset();
    let end = start + array.len();
    &array.values()[start..end]
}

fn as_timestamp_us_array(arr: ArrowPyArray, name: &str) -> PyResult<TimestampMicrosecondArray> {
    let (array_ref, _field) = arr.into_inner();
    let array = array_ref
        .as_any()
        .downcast_ref::<TimestampMicrosecondArray>()
        .cloned()
        .ok_or_else(|| {
            PyValueError::new_err(format!("expected timestamp[us] Arrow array for {name}"))
        })?;
    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }
    Ok(array)
}

fn optional_duration_array(arr: Option<ArrowPyArray>) -> PyResult<Option<Float64Array>> {
    arr.map(|arr| as_f64_array(arr, "durations")).transpose()
}

fn timestamp_values(array: &TimestampMicrosecondArray) -> &[i64] {
    let start = array.offset();
    let end = start + array.len();
    &array.values()[start..end]
}

fn motif_result(
    result: Result<(Vec<usize>, Vec<i32>, Vec<i64>), String>,
) -> PyResult<DailyMotifsPy> {
    let (users, dates, motifs) = result.map_err(PyValueError::new_err)?;
    Ok((
        u64_results_into_arrow(users.into_iter().map(|value| value as u64).collect()),
        i32_results_into_arrow(dates),
        i64_results_into_arrow(motifs),
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn daily_motifs_indexed<'py>(
    py: Python<'py>,
    location_codes: ArrowPyArray,
    purpose_codes: ArrowPyArray,
    start_timestamps: ArrowPyArray,
    end_timestamps: ArrowPyArray,
    durations: Option<ArrowPyArray>,
    indices: ArrowPyArray,
    ends: ArrowPyArray,
    home_purpose_code: u16,
) -> PyResult<DailyMotifsPy> {
    let location_codes = as_u32_array(location_codes, "location_codes")?;
    let purpose_codes = as_u16_array(purpose_codes, "purpose_codes")?;
    let starts = as_timestamp_us_array(start_timestamps, "start_timestamps")?;
    let ends_ts = as_timestamp_us_array(end_timestamps, "end_timestamps")?;
    let durations = optional_duration_array(durations)?;
    let duration_values = durations.as_ref().map(arrow_values);
    let indices = as_u64_array(indices, "indices")?;
    let ends = as_u64_array(ends, "ends")?;
    let index_values = arrow_usize_values(&indices);
    let end_values = arrow_usize_values(&ends);
    let result = py.detach(|| {
        core_compute_daily_motifs_indexed(
            arrow_u32_values(&location_codes),
            u16_values(&purpose_codes),
            timestamp_values(&starts),
            timestamp_values(&ends_ts),
            duration_values,
            index_values,
            end_values,
            home_purpose_code,
        )
    });
    motif_result(result)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn daily_motifs_presorted<'py>(
    py: Python<'py>,
    location_codes: ArrowPyArray,
    purpose_codes: ArrowPyArray,
    start_timestamps: ArrowPyArray,
    end_timestamps: ArrowPyArray,
    durations: Option<ArrowPyArray>,
    ends: ArrowPyArray,
    home_purpose_code: u16,
) -> PyResult<DailyMotifsPy> {
    let location_codes = as_u32_array(location_codes, "location_codes")?;
    let purpose_codes = as_u16_array(purpose_codes, "purpose_codes")?;
    let starts = as_timestamp_us_array(start_timestamps, "start_timestamps")?;
    let ends_ts = as_timestamp_us_array(end_timestamps, "end_timestamps")?;
    let durations = optional_duration_array(durations)?;
    let duration_values = durations.as_ref().map(arrow_values);
    let ends = as_u64_array(ends, "ends")?;
    let end_values = arrow_usize_values(&ends);
    let result = py.detach(|| {
        core_compute_daily_motifs_presorted(
            arrow_u32_values(&location_codes),
            u16_values(&purpose_codes),
            timestamp_values(&starts),
            timestamp_values(&ends_ts),
            duration_values,
            end_values,
            home_purpose_code,
        )
    });
    motif_result(result)
}

/// Build the `(user_idx, location_code) -> purpose_code` lookup used by the
/// Staypoints/Locations hierarchy integration, from three small
/// Locations-grain arrays. Sequential, not parallel: this table is one row
/// per `(uid, location_id)`, dramatically smaller than the visits table it
/// serves, so a parallel fold/reduce (as `discover_purposes` uses elsewhere
/// in this file) buys nothing here. Duplicate `(user_idx, location_code)`
/// keys (shouldn't happen -- Locations is one row per `(uid, location_id)`
/// by construction) silently keep the last-inserted value.
fn build_purpose_lookup(
    user_idx: ArrowPyArray,
    location_code: ArrowPyArray,
    purpose_code: ArrowPyArray,
        ) -> PyResult<FxHashMap<(u32, u32), u16>> {
    let user_idx = as_u32_array(user_idx, "user_idx")?;
    let location_code = as_u32_array(location_code, "location_code")?;
    let purpose_code = as_u16_array(purpose_code, "purpose_code")?;
    let user_idx_values = u32_values(&user_idx);
    let location_code_values = arrow_u32_values(&location_code);
    let purpose_code_values = u16_values(&purpose_code);
    let mut lookup: FxHashMap<(u32, u32), u16> =
        FxHashMap::with_capacity_and_hasher(user_idx_values.len(), Default::default());
    for i in 0..user_idx_values.len() {
        lookup.insert(
            (user_idx_values[i], location_code_values[i]),
            purpose_code_values[i],
        );
    }
    Ok(lookup)
}

/// Rust-join sibling of [`daily_motifs_indexed`]: takes
/// `location_codes`/`start_timestamps`/`end_timestamps`/optional
/// `durations` (no `purpose_codes`), plus `lookup_user_idx`/
/// `lookup_location_code`/`lookup_purpose_code` -- one row per
/// `(uid, location_id)`, resolved from a `Locations` table.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn daily_motifs_indexed_joined<'py>(
    py: Python<'py>,
    location_codes: ArrowPyArray,
    start_timestamps: ArrowPyArray,
    end_timestamps: ArrowPyArray,
    durations: Option<ArrowPyArray>,
    lookup_user_idx: ArrowPyArray,
    lookup_location_code: ArrowPyArray,
    lookup_purpose_code: ArrowPyArray,
    indices: ArrowPyArray,
    ends: ArrowPyArray,
    home_purpose_code: u16,
    unmatched_purpose_code: u16,
) -> PyResult<DailyMotifsPy> {
    let location_codes = as_u32_array(location_codes, "location_codes")?;
    let starts = as_timestamp_us_array(start_timestamps, "start_timestamps")?;
    let ends_ts = as_timestamp_us_array(end_timestamps, "end_timestamps")?;
    let durations = optional_duration_array(durations)?;
    let duration_values = durations.as_ref().map(arrow_values);
    let indices = as_u64_array(indices, "indices")?;
    let ends = as_u64_array(ends, "ends")?;
    let index_values = arrow_usize_values(&indices);
    let end_values = arrow_usize_values(&ends);
    let lookup = build_purpose_lookup(lookup_user_idx, lookup_location_code, lookup_purpose_code)?;
    let result = py.detach(|| {
        core_compute_daily_motifs_indexed_joined(
            arrow_u32_values(&location_codes),
            timestamp_values(&starts),
            timestamp_values(&ends_ts),
            duration_values,
            index_values,
            end_values,
            home_purpose_code,
            &lookup,
            unmatched_purpose_code,
        )
    });
    motif_result(result)
}

/// Rust-join sibling of [`daily_motifs_presorted`]. See
/// [`daily_motifs_indexed_joined`].
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn daily_motifs_presorted_joined<'py>(
    py: Python<'py>,
    location_codes: ArrowPyArray,
    start_timestamps: ArrowPyArray,
    end_timestamps: ArrowPyArray,
    durations: Option<ArrowPyArray>,
    lookup_user_idx: ArrowPyArray,
    lookup_location_code: ArrowPyArray,
    lookup_purpose_code: ArrowPyArray,
    ends: ArrowPyArray,
    home_purpose_code: u16,
    unmatched_purpose_code: u16,
) -> PyResult<DailyMotifsPy> {
    let location_codes = as_u32_array(location_codes, "location_codes")?;
    let starts = as_timestamp_us_array(start_timestamps, "start_timestamps")?;
    let ends_ts = as_timestamp_us_array(end_timestamps, "end_timestamps")?;
    let durations = optional_duration_array(durations)?;
    let duration_values = durations.as_ref().map(arrow_values);
    let ends = as_u64_array(ends, "ends")?;
    let end_values = arrow_usize_values(&ends);
    let lookup = build_purpose_lookup(lookup_user_idx, lookup_location_code, lookup_purpose_code)?;
    let result = py.detach(|| {
        core_compute_daily_motifs_presorted_joined(
            arrow_u32_values(&location_codes),
            timestamp_values(&starts),
            timestamp_values(&ends_ts),
            duration_values,
            end_values,
            home_purpose_code,
            &lookup,
            unmatched_purpose_code,
        )
    });
    motif_result(result)
}
