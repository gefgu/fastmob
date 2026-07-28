//! Vendored copy of citybehavex's current jumps/RoG orchestration.
//!
//! This is the implementation `fastmob-rs` has to beat, copied verbatim from
//! `citybehavex/web/backend/src/comparison/{features,util,activity}.rs` at the
//! time the migration started. It is vendored rather than depended on for two
//! reasons: citybehavex depends on this repo by relative path, so a dependency
//! back the other way would be circular; and freezing the baseline keeps later
//! citybehavex edits from silently moving the benchmark's goalposts.
//!
//! Do not "improve" anything here — its value is being an unmodified copy.
//! The lint allowances below exist so clippy cannot pressure it to drift.

#![allow(clippy::needless_borrow)]

use polars::prelude::*;
use rustc_hash::FxHashMap;

#[allow(dead_code)] // mirrors citybehavex's struct verbatim
pub struct JumpsRog {
    pub jumps: Vec<f64>,
    pub rog: Vec<f64>,
}

fn to_datetime_expr(schema: &Schema, name: &str) -> Expr {
    match schema.get(name) {
        Some(DataType::String) => col(name).str().to_datetime(
            Some(TimeUnit::Microseconds),
            None,
            StrptimeOptions {
                format: None,
                strict: false,
                exact: true,
                cache: true,
            },
            lit("raise"),
        ),
        Some(DataType::Datetime(_, _)) => col(name),
        _ => col(name).cast(DataType::Datetime(TimeUnit::Microseconds, None)),
    }
}

fn canonical_user_ids(uid: &Series) -> PolarsResult<Series> {
    if matches!(
        uid.dtype(),
        DataType::Int64
            | DataType::Int32
            | DataType::Int16
            | DataType::Int8
            | DataType::UInt64
            | DataType::UInt32
            | DataType::UInt16
            | DataType::UInt8
    ) {
        return Ok(uid.cast(&DataType::Int64)?.with_name("user_id".into()));
    }

    let uid = uid.cast(&DataType::String)?;
    let uid = uid.str()?;
    let mut labels = FxHashMap::<String, i64>::default();
    let mut next = 0i64;
    let values: Vec<Option<i64>> = uid
        .into_iter()
        .map(|value| {
            value.map(|value| {
                *labels.entry(value.to_string()).or_insert_with(|| {
                    let current = next;
                    next += 1;
                    current
                })
            })
        })
        .collect();
    Ok(Series::new("user_id".into(), values))
}

fn canonical_user_ids_vec(uid: &Series) -> PolarsResult<Vec<i64>> {
    Ok(canonical_user_ids(uid)?
        .i64()?
        .into_iter()
        .map(|v| v.unwrap_or(i64::MIN))
        .collect())
}

fn contiguous_user_ranges<T: PartialEq>(sorted_uid: &[T]) -> (Vec<usize>, Vec<usize>) {
    let indices: Vec<usize> = (0..sorted_uid.len()).collect();
    let mut ends = Vec::new();
    let mut i = 0;
    while i < sorted_uid.len() {
        let mut j = i + 1;
        while j < sorted_uid.len() && sorted_uid[j] == sorted_uid[i] {
            j += 1;
        }
        ends.push(j);
        i = j;
    }
    (indices, ends)
}

pub fn jumps_rog(
    df: &DataFrame,
    uid_col: &str,
    lat_col: &str,
    lng_col: &str,
    datetime_col: &str,
) -> PolarsResult<JumpsRog> {
    if df.height() == 0 {
        return Ok(JumpsRog {
            jumps: Vec::new(),
            rog: Vec::new(),
        });
    }
    let schema = df.schema();
    let dt_expr = to_datetime_expr(&schema, datetime_col);
    let sorted = df
        .clone()
        .lazy()
        .select([
            col(uid_col),
            col(lat_col).cast(DataType::Float64),
            col(lng_col).cast(DataType::Float64),
            dt_expr.alias(datetime_col),
        ])
        .drop_nulls(Some(cols([uid_col, lat_col, lng_col, datetime_col])))
        .filter(col(lat_col).is_finite().and(col(lng_col).is_finite()))
        .sort([uid_col, datetime_col], SortMultipleOptions::default())
        .collect()?;
    if sorted.height() == 0 {
        return Ok(JumpsRog {
            jumps: Vec::new(),
            rog: Vec::new(),
        });
    }

    let uid: Vec<i64> = canonical_user_ids_vec(sorted.column(uid_col)?.as_materialized_series())?;
    let lat: Vec<f64> = sorted
        .column(lat_col)?
        .f64()?
        .into_iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect();
    let lng: Vec<f64> = sorted
        .column(lng_col)?
        .f64()?
        .into_iter()
        .map(|v| v.unwrap_or(f64::NAN))
        .collect();

    let (_indices, ends) = contiguous_user_ranges(&uid);
    let (_starts, _ends, jumps_all) =
        fastmob_core::measures::individual::jump_lengths::jump_lengths_presorted_impl(
            &lat, &lng, &ends,
        )
        .expect("presorted jump lengths");
    let jumps = jumps_all.into_iter().filter(|d| *d > 0.0).collect();
    let (rog_all, valid) =
        fastmob_core::measures::individual::radius_of_gyration::radius_of_gyration_presorted_impl(
            &lat, &lng, &ends,
        );
    let rog = rog_all
        .into_iter()
        .zip(valid)
        .filter_map(|(value, is_valid)| is_valid.then_some(value))
        .collect();
    Ok(JumpsRog { jumps, rog })
}
