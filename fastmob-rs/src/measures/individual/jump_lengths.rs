use fastmob_core::measures::individual::jump_lengths::jump_lengths_presorted_impl;
use polars::prelude::*;
use std::borrow::Cow;
use std::collections::HashSet;
use std::time::Instant;
use thiserror::Error;

const DATETIME_CANDIDATES: &[&str] = &["datetime", "check-in_time", "timestamp", "time"];
const LAT_CANDIDATES: &[&str] = &["lat", "latitude"];
const LNG_CANDIDATES: &[&str] = &["lng", "lon", "longitude", "long"];
const UID_CANDIDATES: &[&str] = &["uid", "user", "user_id"];

#[derive(Debug, Clone, Default)]
pub struct JumpLengthsOptions {
    pub datetime_col: Option<String>,
    pub lat_col: Option<String>,
    pub lng_col: Option<String>,
    pub uid_col: Option<String>,
    pub presorted: bool,
    pub merge: bool,
}

#[derive(Debug, Clone)]
pub enum JumpLengthsResult {
    Grouped(DataFrame),
    Flat(Series),
}

#[derive(Debug, Error)]
pub enum FastmobRsError {
    #[error("could not find required column(s): {0}. Available columns: {1:?}")]
    MissingColumns(String, Vec<String>),
    #[error("column {0:?} has unsupported dtype {1:?}")]
    UnsupportedDtype(String, DataType),
    #[error("fastmob-core error: {0}")]
    Core(String),
    #[error(transparent)]
    Polars(#[from] PolarsError),
}

#[derive(Debug, Clone)]
struct ResolvedColumns {
    datetime: String,
    lat: String,
    lng: String,
    uid: Option<String>,
}

#[derive(Debug, Clone)]
struct Prepared<'a> {
    latitudes: Cow<'a, [f64]>,
    longitudes: Cow<'a, [f64]>,
}

fn pick_existing(columns: &[String], candidates: &[&str]) -> Option<String> {
    candidates
        .iter()
        .find(|candidate| columns.iter().any(|column| column == **candidate))
        .map(|value| (*value).to_string())
}

fn resolve_columns(
    df: &DataFrame,
    options: &JumpLengthsOptions,
) -> Result<ResolvedColumns, FastmobRsError> {
    let columns = df
        .get_column_names()
        .iter()
        .map(|name| name.as_str().to_string())
        .collect::<Vec<_>>();
    let datetime = options
        .datetime_col
        .clone()
        .or_else(|| pick_existing(&columns, DATETIME_CANDIDATES));
    let lat = options
        .lat_col
        .clone()
        .or_else(|| pick_existing(&columns, LAT_CANDIDATES));
    let lng = options
        .lng_col
        .clone()
        .or_else(|| pick_existing(&columns, LNG_CANDIDATES));
    let uid = options
        .uid_col
        .clone()
        .or_else(|| pick_existing(&columns, UID_CANDIDATES));

    let missing = [
        ("datetime", datetime.as_ref()),
        ("latitude", lat.as_ref()),
        ("longitude", lng.as_ref()),
    ]
    .into_iter()
    .filter_map(|(name, value)| value.is_none().then_some(name))
    .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(FastmobRsError::MissingColumns(missing.join(", "), columns));
    }
    Ok(ResolvedColumns {
        datetime: datetime.unwrap(),
        lat: lat.unwrap(),
        lng: lng.unwrap(),
        uid,
    })
}

fn f64_values<'a>(df: &'a DataFrame, name: &str) -> Result<Cow<'a, [f64]>, FastmobRsError> {
    let series = df.column(name)?.as_materialized_series();
    match series.dtype() {
        DataType::Float64 => {
            let values = series.f64()?;
            if let Ok(slice) = values.cont_slice() {
                return Ok(Cow::Borrowed(slice));
            }
            if values.null_count() == 0 {
                let rechunked = values.rechunk();
                return Ok(Cow::Owned(rechunked.cont_slice()?.to_vec()));
            }
            Ok(Cow::Owned(
                (0..values.len())
                    .map(|idx| values.get(idx).unwrap_or(f64::NAN))
                    .collect(),
            ))
        }
        DataType::Float32 => {
            let values = series.f32()?;
            if let Ok(slice) = values.cont_slice() {
                return Ok(Cow::Owned(
                    slice.iter().map(|&value| value as f64).collect(),
                ));
            }
            if values.null_count() == 0 {
                let rechunked = values.rechunk();
                return Ok(Cow::Owned(
                    rechunked
                        .cont_slice()?
                        .iter()
                        .map(|&value| value as f64)
                        .collect(),
                ));
            }
            Ok(Cow::Owned(
                (0..values.len())
                    .map(|idx| values.get(idx).map_or(f64::NAN, |value| value as f64))
                    .collect(),
            ))
        }
        _ => {
            let cast = series.cast(&DataType::Float64)?;
            let values = cast.f64()?;
            if let Ok(slice) = values.cont_slice() {
                return Ok(Cow::Owned(slice.to_vec()));
            }
            Ok(Cow::Owned(
                (0..values.len())
                    .map(|idx| values.get(idx).unwrap_or(f64::NAN))
                    .collect(),
            ))
        }
    }
}

fn finite_no_null_f64_column(df: &DataFrame, name: &str) -> Result<bool, FastmobRsError> {
    let series = df.column(name)?.as_materialized_series();
    match series.dtype() {
        DataType::Float64 => {
            let values = series.f64()?;
            Ok(values.null_count() == 0
                && values
                    .downcast_iter()
                    .all(|array| array.values().iter().all(|value| value.is_finite())))
        }
        DataType::Float32 => {
            let values = series.f32()?;
            Ok(values.null_count() == 0
                && values
                    .downcast_iter()
                    .all(|array| array.values().iter().all(|value| value.is_finite())))
        }
        _ => Ok(false),
    }
}

fn i64_values_for_sortedness(series: &Series) -> Result<Option<Vec<i64>>, FastmobRsError> {
    match series.dtype() {
        DataType::Int64
        | DataType::Int32
        | DataType::Int16
        | DataType::Int8
        | DataType::UInt32
        | DataType::UInt16
        | DataType::UInt8 => {
            let cast = series.cast(&DataType::Int64)?;
            let values = cast.i64()?;
            if values.null_count() != 0 {
                return Ok(None);
            }
            if let Ok(slice) = values.cont_slice() {
                return Ok(Some(slice.to_vec()));
            }
            let rechunked = values.rechunk();
            Ok(Some(rechunked.cont_slice()?.to_vec()))
        }
        DataType::UInt64 => {
            let values = series.u64()?;
            if values.null_count() != 0 {
                return Ok(None);
            }
            let mut out = Vec::with_capacity(values.len());
            for value in values.into_no_null_iter() {
                let Ok(value) = i64::try_from(value) else {
                    return Ok(None);
                };
                out.push(value);
            }
            Ok(Some(out))
        }
        DataType::Datetime(_, _) => {
            let values = &series.datetime()?.phys;
            if values.null_count() != 0 {
                return Ok(None);
            }
            if let Ok(slice) = values.cont_slice() {
                return Ok(Some(slice.to_vec()));
            }
            let rechunked = values.rechunk();
            Ok(Some(rechunked.cont_slice()?.to_vec()))
        }
        DataType::Date => {
            let values = &series.date()?.phys;
            if values.null_count() != 0 {
                return Ok(None);
            }
            if let Ok(slice) = values.cont_slice() {
                return Ok(Some(slice.iter().map(|&value| value as i64).collect()));
            }
            let rechunked = values.rechunk();
            Ok(Some(
                rechunked
                    .cont_slice()?
                    .iter()
                    .map(|&value| value as i64)
                    .collect(),
            ))
        }
        _ => Ok(None),
    }
}

fn monotonic_direction(previous: i64, current: i64) -> i8 {
    match current.cmp(&previous) {
        std::cmp::Ordering::Greater => 1,
        std::cmp::Ordering::Less => -1,
        std::cmp::Ordering::Equal => 0,
    }
}

fn string_uid_grouped_monotonic(
    uids: &StringChunked,
    timestamps: &[i64],
) -> Result<bool, FastmobRsError> {
    if uids.null_count() != 0 || uids.len() != timestamps.len() {
        return Ok(false);
    }

    let mut iter = uids.into_iter();
    let Some(Some(mut current_uid)) = iter.next() else {
        return Ok(timestamps.is_empty());
    };

    let mut seen_groups = HashSet::new();
    let mut direction = 0i8;
    for (idx, uid) in iter.enumerate() {
        let Some(uid) = uid else {
            return Ok(false);
        };
        let idx = idx + 1;
        if uid != current_uid {
            seen_groups.insert(current_uid);
            if seen_groups.contains(uid) {
                return Ok(false);
            }
            current_uid = uid;
            direction = 0;
            continue;
        }

        let current = monotonic_direction(timestamps[idx - 1], timestamps[idx]);
        if current != 0 {
            if direction == 0 {
                direction = current;
            } else if direction != current {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn clean_grouped_monotonic(
    df: &DataFrame,
    columns: &ResolvedColumns,
) -> Result<bool, FastmobRsError> {
    if !finite_no_null_f64_column(df, &columns.lat)?
        || !finite_no_null_f64_column(df, &columns.lng)?
    {
        return Ok(false);
    }

    let Some(timestamps) =
        i64_values_for_sortedness(df.column(&columns.datetime)?.as_materialized_series())?
    else {
        return Ok(false);
    };
    if timestamps.len() <= 1 {
        return Ok(true);
    }

    let Some(uid_col) = columns.uid.as_deref() else {
        let mut direction = 0i8;
        for window in timestamps.windows(2) {
            let current = monotonic_direction(window[0], window[1]);
            if current != 0 {
                if direction == 0 {
                    direction = current;
                } else if direction != current {
                    return Ok(false);
                }
            }
        }
        return Ok(true);
    };

    let Some(uids) = i64_values_for_sortedness(df.column(uid_col)?.as_materialized_series())?
    else {
        let series = df.column(uid_col)?.as_materialized_series();
        return match series.dtype() {
            DataType::String => string_uid_grouped_monotonic(series.str()?, &timestamps),
            _ => {
                let cast = series.cast(&DataType::String)?;
                string_uid_grouped_monotonic(cast.str()?, &timestamps)
            }
        };
    };
    if uids.len() != timestamps.len() {
        return Ok(false);
    }

    let mut seen_groups = HashSet::new();
    let mut current_uid = uids[0];
    let mut direction = 0i8;
    for idx in 1..uids.len() {
        if uids[idx] != current_uid {
            seen_groups.insert(current_uid);
            if seen_groups.contains(&uids[idx]) {
                return Ok(false);
            }
            current_uid = uids[idx];
            direction = 0;
            continue;
        }

        let current = monotonic_direction(timestamps[idx - 1], timestamps[idx]);
        if current != 0 {
            if direction == 0 {
                direction = current;
            } else if direction != current {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn boundaries_from_values<T: PartialEq>(values: &[T]) -> (Vec<usize>, Vec<usize>) {
    if values.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut starts = vec![0usize];
    let mut ends = Vec::new();
    for (idx, (previous, current)) in values.iter().zip(&values[1..]).enumerate() {
        if current != previous {
            let idx = idx + 1;
            starts.push(idx);
            ends.push(idx);
        }
    }
    ends.push(values.len());
    (ends, starts)
}

fn boundaries_from_string_chunked(values: &StringChunked) -> (Vec<usize>, Vec<usize>) {
    let mut iter = values.into_iter();
    let Some(mut previous) = iter.next() else {
        return (Vec::new(), Vec::new());
    };
    let mut starts = vec![0usize];
    let mut ends = Vec::new();
    let mut len = 1usize;
    for current in iter {
        if current != previous {
            starts.push(len);
            ends.push(len);
            previous = current;
        }
        len += 1;
    }
    ends.push(len);
    (ends, starts)
}

fn presorted_ends(
    df: &DataFrame,
    uid_col: Option<&str>,
    len: usize,
) -> Result<(Vec<usize>, Option<Vec<usize>>), FastmobRsError> {
    let Some(uid_col) = uid_col else {
        return Ok((if len == 0 { Vec::new() } else { vec![len] }, None));
    };
    let series = df.column(uid_col)?.as_materialized_series();
    let (ends, starts) = match series.dtype() {
        DataType::Int64 | DataType::Int32 | DataType::Int16 | DataType::Int8 => {
            let cast = series.cast(&DataType::Int64)?;
            let values = cast.i64()?;
            if let Ok(slice) = values.cont_slice() {
                boundaries_from_values(slice)
            } else if values.null_count() == 0 {
                let rechunked = values.rechunk();
                boundaries_from_values(rechunked.cont_slice()?)
            } else {
                boundaries_from_values(
                    &(0..values.len())
                        .map(|idx| values.get(idx))
                        .collect::<Vec<_>>(),
                )
            }
        }
        DataType::UInt64 | DataType::UInt32 | DataType::UInt16 | DataType::UInt8 => {
            let cast = series.cast(&DataType::UInt64)?;
            let values = cast.u64()?;
            if let Ok(slice) = values.cont_slice() {
                boundaries_from_values(slice)
            } else if values.null_count() == 0 {
                let rechunked = values.rechunk();
                boundaries_from_values(rechunked.cont_slice()?)
            } else {
                boundaries_from_values(
                    &(0..values.len())
                        .map(|idx| values.get(idx))
                        .collect::<Vec<_>>(),
                )
            }
        }
        DataType::String => boundaries_from_string_chunked(series.str()?),
        _ => {
            let cast = series.cast(&DataType::String)?;
            boundaries_from_string_chunked(cast.str()?)
        }
    };
    Ok((ends, Some(starts)))
}

fn prepare<'a>(
    df: &'a DataFrame,
    columns: &ResolvedColumns,
) -> Result<Prepared<'a>, FastmobRsError> {
    let t0 = Instant::now();
    let latitudes = f64_values(df, &columns.lat)?;
    println!("rust: prepare lat f64: {:.6}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let longitudes = f64_values(df, &columns.lng)?;
    println!("rust: prepare lng f64: {:.6}s", t0.elapsed().as_secs_f64());

    Ok(Prepared {
        latitudes,
        longitudes,
    })
}

fn cleaned_work_frame(
    df: &DataFrame,
    columns: &ResolvedColumns,
) -> Result<DataFrame, FastmobRsError> {
    let mut select_exprs = Vec::new();
    let mut required = Vec::new();

    if let Some(uid_col) = columns.uid.as_deref() {
        select_exprs.push(col(uid_col));
        required.push(uid_col);
    }
    select_exprs.push(col(&columns.lat).cast(DataType::Float64));
    select_exprs.push(col(&columns.lng).cast(DataType::Float64));
    select_exprs.push(col(&columns.datetime));
    required.extend([
        columns.lat.as_str(),
        columns.lng.as_str(),
        columns.datetime.as_str(),
    ]);
    Ok(df
        .clone()
        .lazy()
        .select(select_exprs)
        .drop_nulls(Some(cols(required)))
        .filter(
            col(&columns.lat)
                .is_finite()
                .and(col(&columns.lng).is_finite()),
        )
        .collect()?)
}

fn sort_work_frame(df: &DataFrame, columns: &ResolvedColumns) -> Result<DataFrame, FastmobRsError> {
    let mut sort_exprs = Vec::new();
    if let Some(uid_col) = columns.uid.as_deref() {
        sort_exprs.push(uid_col);
    }
    sort_exprs.push(columns.datetime.as_str());

    Ok(df
        .clone()
        .lazy()
        .sort(sort_exprs, SortMultipleOptions::default())
        .collect()?)
}

fn list_series(
    name: &str,
    starts: &[usize],
    ends: &[usize],
    values: &[f64],
) -> Result<Series, FastmobRsError> {
    let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
        name.into(),
        starts.len(),
        values.len(),
        DataType::Float64,
    );
    for (&start, &end) in starts.iter().zip(ends) {
        builder.append_slice(&values[start..end]);
    }
    Ok(builder.finish().into_series())
}

fn grouped_output(
    df: &DataFrame,
    uid_col: Option<&str>,
    uid_rows: Option<Vec<usize>>,
    starts: Vec<usize>,
    ends: Vec<usize>,
    values: Vec<f64>,
) -> Result<DataFrame, FastmobRsError> {
    let t0 = Instant::now();
    let jumps = list_series("jump_lengths", &starts, &ends, &values)?;
    println!(
        "rust: grouped_output list series: {:.6}s",
        t0.elapsed().as_secs_f64()
    );
    let mut columns = Vec::<Column>::new();
    if let (Some(uid_col), Some(uid_rows)) = (uid_col, uid_rows) {
        let t0 = Instant::now();
        let row_indices = uid_rows
            .into_iter()
            .map(|idx| idx as IdxSize)
            .collect::<Vec<_>>();
        let mut uid_series = df
            .column(uid_col)?
            .as_materialized_series()
            .take_slice(&row_indices)?;
        uid_series.rename(uid_col.into());
        columns.push(uid_series.into_column());
        println!(
            "rust: grouped_output uid take: {:.6}s",
            t0.elapsed().as_secs_f64()
        );
    }
    columns.push(jumps.into_column());
    let t0 = Instant::now();
    let out = DataFrame::new(columns)?;
    println!(
        "rust: grouped_output dataframe new: {:.6}s",
        t0.elapsed().as_secs_f64()
    );
    Ok(out)
}

pub fn jump_lengths(
    df: &DataFrame,
    options: JumpLengthsOptions,
) -> Result<JumpLengthsResult, FastmobRsError> {
    let total = Instant::now();
    let t0 = Instant::now();
    let columns = resolve_columns(df, &options)?;
    println!("rust: resolve columns: {:.6}s", t0.elapsed().as_secs_f64());
    let t0 = Instant::now();
    let work_df = if options.presorted {
        println!("rust: sortedness check: 0.000000s");
        println!("rust: skip sort: true");
        df.clone()
    } else {
        let t_clean = Instant::now();
        let cleaned = cleaned_work_frame(df, &columns)?;
        println!(
            "rust: clean work frame: {:.6}s",
            t_clean.elapsed().as_secs_f64()
        );

        let t_check = Instant::now();
        let can_skip_sort = clean_grouped_monotonic(&cleaned, &columns)?;
        println!(
            "rust: sortedness check: {:.6}s",
            t_check.elapsed().as_secs_f64()
        );
        println!("rust: skip sort: {}", can_skip_sort);

        if can_skip_sort {
            cleaned
        } else {
            let t_sort = Instant::now();
            let sorted = sort_work_frame(&cleaned, &columns)?;
            println!(
                "rust: sort cleaned work frame: {:.6}s",
                t_sort.elapsed().as_secs_f64()
            );
            sorted
        }
    };
    println!(
        "rust: polars work frame: {:.6}s",
        t0.elapsed().as_secs_f64()
    );
    let t0 = Instant::now();
    let prepared = prepare(&work_df, &columns)?;
    println!("rust: prepare total: {:.6}s", t0.elapsed().as_secs_f64());
    if prepared.latitudes.len() != prepared.longitudes.len() {
        return Err(FastmobRsError::Core(
            "latitudes and longitudes must have the same length".to_string(),
        ));
    }

    let (starts, ends, values, uid_rows) = if options.presorted {
        let t0 = Instant::now();
        let (ends, uid_rows) =
            presorted_ends(&work_df, columns.uid.as_deref(), prepared.latitudes.len())?;
        println!("rust: presorted ends: {:.6}s", t0.elapsed().as_secs_f64());
        let t0 = Instant::now();
        let (starts, value_ends, values) = jump_lengths_presorted_impl(
            prepared.latitudes.as_ref(),
            prepared.longitudes.as_ref(),
            &ends,
        )
        .map_err(FastmobRsError::Core)?;
        println!(
            "rust: core presorted compute: {:.6}s",
            t0.elapsed().as_secs_f64()
        );
        (starts, value_ends, values, uid_rows)
    } else {
        let t0 = Instant::now();
        let (ends, uid_rows) =
            presorted_ends(&work_df, columns.uid.as_deref(), prepared.latitudes.len())?;
        println!(
            "rust: sorted presorted ends: {:.6}s",
            t0.elapsed().as_secs_f64()
        );
        let t0 = Instant::now();
        let (starts, value_ends, values) = jump_lengths_presorted_impl(
            prepared.latitudes.as_ref(),
            prepared.longitudes.as_ref(),
            &ends,
        )
        .map_err(FastmobRsError::Core)?;
        println!(
            "rust: core sorted presorted compute: {:.6}s",
            t0.elapsed().as_secs_f64()
        );
        (starts, value_ends, values, uid_rows)
    };

    if options.merge {
        let t0 = Instant::now();
        let series = Series::new("jump_lengths".into(), values);
        println!("rust: flat series: {:.6}s", t0.elapsed().as_secs_f64());
        println!(
            "rust: jump_lengths total: {:.6}s",
            total.elapsed().as_secs_f64()
        );
        Ok(JumpLengthsResult::Flat(series))
    } else {
        let t0 = Instant::now();
        let out = grouped_output(
            &work_df,
            columns.uid.as_deref(),
            uid_rows,
            starts,
            ends,
            values,
        )?;
        println!(
            "rust: grouped output total: {:.6}s",
            t0.elapsed().as_secs_f64()
        );
        println!(
            "rust: jump_lengths total: {:.6}s",
            total.elapsed().as_secs_f64()
        );
        Ok(JumpLengthsResult::Grouped(out))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fastmob_core::measures::individual::jump_lengths::jump_lengths_km;

    fn grouped(df: DataFrame, presorted: bool) -> DataFrame {
        match jump_lengths(
            &df,
            JumpLengthsOptions {
                presorted,
                ..Default::default()
            },
        )
        .unwrap()
        {
            JumpLengthsResult::Grouped(df) => df,
            JumpLengthsResult::Flat(_) => panic!("expected grouped result"),
        }
    }

    #[test]
    fn unsorted_input_is_ordered_by_user_time_stably() {
        let df = df![
            "uid" => ["b", "a", "b", "a", "a", "b"],
            "datetime" => [
                2_000i64, 2_000, 0, 0, 1_000, 1_000,
            ],
            "lat" => [10.0, 0.0, 10.0, 0.0, 0.0, 10.0],
            "lng" => [4.0, 3.0, 0.0, 0.0, 1.0, 2.0],
        ]
        .unwrap();
        let out = grouped(df, false);
        let jumps = out.column("jump_lengths").unwrap().list().unwrap();
        let uid = out.column("uid").unwrap().str().unwrap();
        let a_idx = (0..out.height())
            .find(|&idx| uid.get(idx) == Some("a"))
            .unwrap();
        let b_idx = (0..out.height())
            .find(|&idx| uid.get(idx) == Some("b"))
            .unwrap();
        let a = jumps.get_as_series(a_idx).unwrap();
        let b = jumps.get_as_series(b_idx).unwrap();
        assert_eq!(
            a.f64().unwrap().into_no_null_iter().collect::<Vec<_>>(),
            jump_lengths_km(vec![0.0, 0.0, 0.0], vec![0.0, 1.0, 3.0]).unwrap()
        );
        assert_eq!(
            b.f64().unwrap().into_no_null_iter().collect::<Vec<_>>(),
            jump_lengths_km(vec![10.0, 10.0, 10.0], vec![0.0, 2.0, 4.0]).unwrap()
        );
    }

    #[test]
    fn presorted_input_uses_existing_group_order() {
        let df = df![
            "uid" => ["a", "a", "a", "b", "b", "b"],
            "datetime" => [0i64, 1_000, 2_000, 0, 1_000, 2_000],
            "lat" => [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            "lng" => [0.0, 1.0, 3.0, 0.0, 2.0, 4.0],
        ]
        .unwrap();
        let out = grouped(df, true);
        assert_eq!(out.height(), 2);
    }

    #[test]
    fn no_uid_column_returns_single_group() {
        let df = df![
            "datetime" => [0i64, 1_000, 2_000],
            "lat" => [0.0, 0.0, 0.0],
            "lng" => [0.0, 1.0, 3.0],
        ]
        .unwrap();
        let out = grouped(df, false);
        assert_eq!(out.height(), 1);
        assert!(out.column("uid").is_err());
    }

    #[test]
    fn merge_returns_flat_series() {
        let df = df![
            "datetime" => [0i64, 1_000, 2_000],
            "lat" => [0.0, 0.0, 0.0],
            "lng" => [0.0, 1.0, 3.0],
        ]
        .unwrap();
        let result = jump_lengths(
            &df,
            JumpLengthsOptions {
                merge: true,
                ..Default::default()
            },
        )
        .unwrap();
        match result {
            JumpLengthsResult::Flat(series) => assert_eq!(series.len(), 2),
            JumpLengthsResult::Grouped(_) => panic!("expected flat result"),
        }
    }

    #[test]
    fn grouped_output_preserves_integer_uid_dtype() {
        let df = df![
            "uid" => [2i64, 1, 2, 1],
            "datetime" => [1_000i64, 0, 0, 1_000],
            "lat" => [10.0, 0.0, 10.0, 0.0],
            "lng" => [1.0, 0.0, 0.0, 1.0],
        ]
        .unwrap();
        let out = grouped(df, false);
        let uid = out.column("uid").unwrap();
        assert_eq!(uid.dtype(), &DataType::Int64);
        let mut values = uid
            .as_materialized_series()
            .i64()
            .unwrap()
            .into_no_null_iter()
            .collect::<Vec<_>>();
        values.sort_unstable();
        assert_eq!(values, vec![1, 2]);
    }
}
