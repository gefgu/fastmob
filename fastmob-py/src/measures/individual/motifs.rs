use std::time::Instant;

use arrow_array::{Array, LargeStringArray, StringArray, TimestampMicrosecondArray};
use fastmob_core::measures::individual::motifs::{
    canonical_adjacency_form as core_canonical_adjacency_form,
    compute_daily_motifs_indexed as core_compute_daily_motifs_indexed,
    compute_daily_motifs_presorted as core_compute_daily_motifs_presorted,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;
use rustc_hash::FxHashMap;

use crate::utils::{
    arrow_u64_values, arrow_values, as_f64_array, as_u64_array, i32_results_into_arrow,
    i64_results_into_arrow, u64_results_into_arrow,
};

#[pyfunction]
pub fn canonical_adjacency_form(n_nodes: u32, edges: Vec<(u32, u32)>) -> PyResult<i64> {
    core_canonical_adjacency_form(n_nodes, edges).map_err(PyValueError::new_err)
}

type DailyMotifsPy = (ArrowPyArray, ArrowPyArray, ArrowPyArray);

enum PurposeArray {
    Utf8(StringArray),
    LargeUtf8(LargeStringArray),
}

fn as_purpose_array(arr: ArrowPyArray) -> PyResult<PurposeArray> {
    let (array_ref, _field) = arr.into_inner();
    if let Some(array) = array_ref.as_any().downcast_ref::<StringArray>() {
        return Ok(PurposeArray::Utf8(array.clone()));
    }
    if let Some(array) = array_ref.as_any().downcast_ref::<LargeStringArray>() {
        return Ok(PurposeArray::LargeUtf8(array.clone()));
    }
    Err(PyValueError::new_err(
        "expected string Arrow array for purposes",
    ))
}

fn encode_purpose_values<'a>(values: impl Iterator<Item = Option<&'a str>>) -> (Vec<u64>, u64) {
    let mut mapping: FxHashMap<&str, u64> = FxHashMap::default();
    let mut null_code = None;
    let mut home_code = None;
    let mut codes = Vec::with_capacity(values.size_hint().0);
    for value in values {
        let code = if let Some(value) = value {
            if let Some(&code) = mapping.get(value) {
                code
            } else {
                let code = mapping.len() as u64 + u64::from(null_code.is_some());
                mapping.insert(value, code);
                code
            }
        } else {
            *null_code.get_or_insert(mapping.len() as u64)
        };
        if value == Some("HOME") {
            home_code = Some(code);
        }
        codes.push(code);
    }
    let missing_home = mapping.len() as u64 + u64::from(null_code.is_some());
    (codes, home_code.unwrap_or(missing_home))
}

fn encode_purposes(array: &PurposeArray) -> (Vec<u64>, u64) {
    match array {
        PurposeArray::Utf8(values) => encode_purpose_values(
            (0..values.len()).map(|index| values.is_valid(index).then(|| values.value(index))),
        ),
        PurposeArray::LargeUtf8(values) => encode_purpose_values(
            (0..values.len()).map(|index| values.is_valid(index).then(|| values.value(index))),
        ),
    }
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
    purposes: ArrowPyArray,
    start_timestamps: ArrowPyArray,
    end_timestamps: ArrowPyArray,
    durations: Option<ArrowPyArray>,
    indices: ArrowPyArray,
    ends: ArrowPyArray,
) -> PyResult<DailyMotifsPy> {
    let location_codes = as_u64_array(location_codes, "location_codes")?;
    let purposes = as_purpose_array(purposes)?;
    let starts = as_timestamp_us_array(start_timestamps, "start_timestamps")?;
    let ends_ts = as_timestamp_us_array(end_timestamps, "end_timestamps")?;
    let durations = durations
        .map(|values| as_f64_array(values, "durations"))
        .transpose()?;
    let duration_values = durations.as_ref().map(arrow_values);
    let indices = as_u64_array(indices, "indices")?;
    let ends = as_u64_array(ends, "ends")?;
    let index_values: Vec<usize> = arrow_u64_values(&indices)
        .iter()
        .map(|&value| value as usize)
        .collect();
    let end_values: Vec<usize> = arrow_u64_values(&ends)
        .iter()
        .map(|&value| value as usize)
        .collect();
    let result = py.detach(|| {
        let encode_started = Instant::now();
        let (purpose_codes, home_purpose_code) = encode_purposes(&purposes);
        if std::env::var_os("FASTMOB_PROFILE_MOTIFS").is_some() {
            eprintln!(
                "fastmob motif rust: purpose_encoding={:.6}s",
                encode_started.elapsed().as_secs_f64()
            );
        }
        core_compute_daily_motifs_indexed(
            arrow_u64_values(&location_codes),
            &purpose_codes,
            timestamp_values(&starts),
            timestamp_values(&ends_ts),
            duration_values,
            &index_values,
            &end_values,
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
    purposes: ArrowPyArray,
    start_timestamps: ArrowPyArray,
    end_timestamps: ArrowPyArray,
    durations: Option<ArrowPyArray>,
    ends: ArrowPyArray,
) -> PyResult<DailyMotifsPy> {
    let location_codes = as_u64_array(location_codes, "location_codes")?;
    let purposes = as_purpose_array(purposes)?;
    let starts = as_timestamp_us_array(start_timestamps, "start_timestamps")?;
    let ends_ts = as_timestamp_us_array(end_timestamps, "end_timestamps")?;
    let durations = durations
        .map(|values| as_f64_array(values, "durations"))
        .transpose()?;
    let duration_values = durations.as_ref().map(arrow_values);
    let ends = as_u64_array(ends, "ends")?;
    let end_values: Vec<usize> = arrow_u64_values(&ends)
        .iter()
        .map(|&value| value as usize)
        .collect();
    let result = py.detach(|| {
        let encode_started = Instant::now();
        let (purpose_codes, home_purpose_code) = encode_purposes(&purposes);
        if std::env::var_os("FASTMOB_PROFILE_MOTIFS").is_some() {
            eprintln!(
                "fastmob motif rust: purpose_encoding={:.6}s",
                encode_started.elapsed().as_secs_f64()
            );
        }
        core_compute_daily_motifs_presorted(
            arrow_u64_values(&location_codes),
            &purpose_codes,
            timestamp_values(&starts),
            timestamp_values(&ends_ts),
            duration_values,
            &end_values,
            home_purpose_code,
        )
    });
    motif_result(result)
}
