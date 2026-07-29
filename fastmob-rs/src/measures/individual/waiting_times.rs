//! Waiting times: the gap between a user's consecutive fixes.

use fastmob_core::measures::individual::waiting_times::waiting_times_flat_impl;
use polars::prelude::*;

use crate::error::FastmobRsError;
use crate::prepare::PreparedTrajectory;

/// Seconds between consecutive fixes, flattened across every user.
///
/// Users with fewer than two fixes contribute nothing. Seconds is fastmob's
/// unit for this measure; callers wanting minutes divide at the call site.
///
/// Accepts either preparation. Prefer
/// [`prepare_temporal`](crate::prepare_temporal): this measure never reads
/// coordinates, and `prepare` would drop fixes with unusable ones, inventing a
/// longer gap between their neighbours.
pub fn waiting_times_flat(prep: &PreparedTrajectory) -> Result<Vec<f64>, FastmobRsError> {
    let seconds = timestamps_seconds(prep);
    waiting_times_flat_impl(&seconds, prep.ends()).map_err(FastmobRsError::Core)
}

/// Waiting times (seconds) as a single-column `"waiting_times"` frame.
pub fn waiting_times(prep: &PreparedTrajectory) -> Result<DataFrame, FastmobRsError> {
    let values = waiting_times_flat(prep)?;
    Ok(DataFrame::new(vec![
        Series::new("waiting_times".into(), values).into_column(),
    ])?)
}

fn timestamps_seconds(prep: &PreparedTrajectory) -> Vec<f64> {
    prep.timestamps_ms()
        .iter()
        .map(|&ms| ms as f64 / 1_000.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prepare::{Cols, prepare, prepare_temporal};

    fn frame(uids: [i64; 5], seconds: [i64; 5]) -> DataFrame {
        let datetimes: Vec<i64> = seconds.iter().map(|s| s * 1_000_000).collect();
        df![
            "uid" => uids.to_vec(),
            "datetime" => datetimes,
            "lat" => vec![0.0; 5],
            "lng" => vec![0.0; 5],
        ]
        .unwrap()
        .lazy()
        .with_column(col("datetime").cast(DataType::Datetime(TimeUnit::Microseconds, None)))
        .collect()
        .unwrap()
    }

    #[test]
    fn gaps_are_measured_in_seconds_within_each_user() {
        let df = frame([1, 1, 1, 2, 2], [0, 60, 200, 0, 30]);
        let values = waiting_times_flat(&prepare(&df, Cols::auto()).unwrap()).unwrap();
        assert_eq!(values, vec![60.0, 140.0, 30.0]);
    }

    #[test]
    fn gaps_never_span_two_users() {
        let df = frame([1, 1, 2, 2, 2], [0, 10, 1_000, 1_010, 1_020]);
        let values = waiting_times_flat(&prepare(&df, Cols::auto()).unwrap()).unwrap();
        assert_eq!(values, vec![10.0, 10.0, 10.0]);
        assert!(!values.contains(&990.0), "no gap should bridge users");
    }

    #[test]
    fn unsorted_input_is_time_ordered_first() {
        let sorted = frame([1, 1, 1, 2, 2], [0, 60, 200, 0, 30]);
        let shuffled = frame([1, 2, 1, 1, 2], [200, 30, 0, 60, 0]);
        assert_eq!(
            waiting_times_flat(&prepare(&shuffled, Cols::auto()).unwrap()).unwrap(),
            waiting_times_flat(&prepare(&sorted, Cols::auto()).unwrap()).unwrap()
        );
    }

    #[test]
    fn single_fix_users_contribute_nothing() {
        let df = frame([1, 2, 3, 4, 5], [0, 1, 2, 3, 4]);
        assert!(
            waiting_times_flat(&prepare(&df, Cols::auto()).unwrap())
                .unwrap()
                .is_empty()
        );
    }

    /// The reason `prepare_temporal` exists: a fix with an unusable coordinate
    /// still happened at a real time, and dropping it would silently merge the
    /// two gaps around it into one longer one.
    #[test]
    fn a_bad_coordinate_does_not_erase_its_time_gap() {
        let df = df![
            "uid" => [1i64, 1, 1],
            "datetime" => [0i64, 60_000_000, 120_000_000],
            "lat" => [Some(0.0), None, Some(0.0)],
            "lng" => [Some(0.0), Some(0.0), Some(0.0)],
        ]
        .unwrap()
        .lazy()
        .with_column(col("datetime").cast(DataType::Datetime(TimeUnit::Microseconds, None)))
        .collect()
        .unwrap();

        let temporal = prepare_temporal(&df, Cols::auto()).unwrap();
        assert_eq!(waiting_times_flat(&temporal).unwrap(), vec![60.0, 60.0]);

        // Spatial preparation drops the middle row, fusing the two gaps.
        let spatial = prepare(&df, Cols::auto()).unwrap();
        assert_eq!(waiting_times_flat(&spatial).unwrap(), vec![120.0]);
    }

    #[test]
    fn time_only_preparation_needs_no_coordinate_columns_at_all() {
        let df = df![
            "uid" => [1i64, 1],
            "datetime" => [0i64, 90_000_000],
        ]
        .unwrap()
        .lazy()
        .with_column(col("datetime").cast(DataType::Datetime(TimeUnit::Microseconds, None)))
        .collect()
        .unwrap();
        let prep = prepare_temporal(&df, Cols::auto()).unwrap();
        assert!(!prep.has_coordinates());
        assert_eq!(waiting_times_flat(&prep).unwrap(), vec![90.0]);
    }

    #[test]
    #[should_panic(expected = "prepare_temporal")]
    fn coordinates_from_a_time_only_trajectory_are_refused() {
        let df = frame([1, 1, 1, 2, 2], [0, 60, 200, 0, 30]);
        let _ = prepare_temporal(&df, Cols::auto()).unwrap().lat();
    }

    #[test]
    fn frame_output_has_one_row_per_gap() {
        let df = frame([1, 1, 1, 2, 2], [0, 60, 200, 0, 30]);
        let out = waiting_times(&prepare(&df, Cols::auto()).unwrap()).unwrap();
        assert_eq!(out.height(), 3);
        assert_eq!(out.get_column_names(), vec!["waiting_times"]);
    }
}
