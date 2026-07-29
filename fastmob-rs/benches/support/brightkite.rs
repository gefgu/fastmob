//! Loads the cached Brightkite check-in dataset for benchmarking.
//!
//! Uses fastmob's existing cache at `tests/shared/data/`, so benches are
//! reproducible from a clean checkout without a network fetch.

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use flate2::read::GzDecoder;
use polars::prelude::*;

pub fn dataset_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("fastmob-rs has a parent directory")
        .join("tests/shared/data/loc-brightkite_totalCheckins.txt.gz")
}

/// Read the first `rows` usable check-ins as a `[uid, datetime, lat, lng]`
/// frame.
///
/// Rows with unparsable fields are skipped here rather than left for the
/// measure under test, so every implementation being compared sees the same
/// input and the benchmark measures orchestration rather than parsing.
pub fn load(rows: usize) -> DataFrame {
    let path = dataset_path();
    let file = File::open(&path)
        .unwrap_or_else(|err| panic!("missing Brightkite cache at {}: {err}", path.display()));
    let reader = BufReader::new(GzDecoder::new(file));

    let mut uids = Vec::with_capacity(rows);
    let mut datetimes = Vec::with_capacity(rows);
    let mut lats = Vec::with_capacity(rows);
    let mut lngs = Vec::with_capacity(rows);

    for line in reader.lines() {
        if uids.len() >= rows {
            break;
        }
        let line = line.expect("readable Brightkite line");
        let mut fields = line.split('\t');
        let (Some(uid), Some(datetime), Some(lat), Some(lng)) =
            (fields.next(), fields.next(), fields.next(), fields.next())
        else {
            continue;
        };
        let (Ok(uid), Ok(lat), Ok(lng)) =
            (uid.parse::<i64>(), lat.parse::<f64>(), lng.parse::<f64>())
        else {
            continue;
        };
        if !lat.is_finite() || !lng.is_finite() {
            continue;
        }
        uids.push(uid);
        datetimes.push(datetime.to_string());
        lats.push(lat);
        lngs.push(lng);
    }

    DataFrame::new(vec![
        Series::new("uid".into(), uids).into_column(),
        Series::new("datetime".into(), datetimes).into_column(),
        Series::new("lat".into(), lats).into_column(),
        Series::new("lng".into(), lngs).into_column(),
    ])
    .expect("well-formed Brightkite frame")
}

/// The same data with the datetime column already parsed to `Datetime`.
///
/// This is the shape citybehavex actually works with — its frames come from
/// parquet, where the column is already typed. Benchmarking both shapes matters
/// because parsing 4M datetime strings costs an order of magnitude more than
/// arranging them, and would otherwise mask everything else in the pipeline.
pub fn load_parsed(rows: usize) -> DataFrame {
    load(rows)
        .lazy()
        .with_column(col("datetime").str().to_datetime(
            Some(TimeUnit::Microseconds),
            None,
            StrptimeOptions {
                format: None,
                strict: false,
                exact: true,
                cache: true,
            },
            lit("raise"),
        ))
        .drop_nulls(Some(cols(["datetime"])))
        .collect()
        .expect("parsed Brightkite frame")
}
