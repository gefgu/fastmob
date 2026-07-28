//! Jump lengths: Haversine distance (km) between a user's consecutive fixes.

use fastmob_core::measures::individual::jump_lengths::jump_lengths_presorted_impl;
use polars::prelude::*;

use crate::error::FastmobRsError;
use crate::prepare::PreparedTrajectory;

/// Flat jump lengths across every user, concatenated in arranged group order.
///
/// This is the zero-conversion entry point: the kernel's `Vec<f64>` is returned
/// untouched. Prefer it whenever the caller wants raw values — building a
/// Polars `Series` and extracting it again cost 16.5 ms at 4M rows, more than
/// three times the distance computation itself.
pub fn jump_lengths_flat(prep: &PreparedTrajectory) -> Result<Vec<f64>, FastmobRsError> {
    Ok(compute(prep)?.2)
}

/// Per-user jump lengths as a `[uid, "jump_lengths"]` frame, where each row
/// holds one user's jumps as a `List(Float64)`.
///
/// Mirrors the Python `jump_lengths(merge=False)` output shape. The uid column
/// is omitted when the input frame had no user column.
pub fn jump_lengths(prep: &PreparedTrajectory) -> Result<DataFrame, FastmobRsError> {
    let (starts, ends, values) = compute(prep)?;
    let jumps = list_series("jump_lengths", &starts, &ends, &values);

    let mut columns = Vec::with_capacity(2);
    if let Some(labels) = prep.uid_labels()? {
        columns.push(labels.into_column());
    }
    columns.push(jumps.into_column());
    Ok(DataFrame::new(columns)?)
}

/// Per-user start/end offsets into the flat values array, plus the values.
type GroupedJumps = (Vec<usize>, Vec<usize>, Vec<f64>);

fn compute(prep: &PreparedTrajectory) -> Result<GroupedJumps, FastmobRsError> {
    jump_lengths_presorted_impl(prep.lat(), prep.lng(), prep.ends()).map_err(FastmobRsError::Core)
}

fn list_series(name: &str, starts: &[usize], ends: &[usize], values: &[f64]) -> Series {
    let mut builder = ListPrimitiveChunkedBuilder::<Float64Type>::new(
        name.into(),
        starts.len(),
        values.len(),
        DataType::Float64,
    );
    for (&start, &end) in starts.iter().zip(ends) {
        builder.append_slice(&values[start..end]);
    }
    builder.finish().into_series()
}

#[cfg(test)]
mod tests {
    use fastmob_core::measures::individual::jump_lengths::jump_lengths_km;

    use super::*;
    use crate::prepare::{Cols, prepare};

    /// `jump_lengths_km` is a different fastmob-core entry point (scalar
    /// Haversine) than the adjacent-distance kernel the measure uses, and the
    /// two disagree in the last bits. These tests are about *ordering*, so they
    /// compare numerically; bit-exactness against the reference implementation
    /// is pinned separately in `tests/parity.rs`.
    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (got, want) in actual.iter().zip(expected) {
            assert!((got - want).abs() < 1e-9, "got {got}, want {want}");
        }
    }

    fn unsorted_two_users() -> DataFrame {
        df![
            "uid" => ["b", "a", "b", "a", "a", "b"],
            "datetime" => [2_000i64, 2_000, 0, 0, 1_000, 1_000],
            "lat" => [10.0, 0.0, 10.0, 0.0, 0.0, 10.0],
            "lng" => [4.0, 3.0, 0.0, 0.0, 1.0, 2.0],
        ]
        .unwrap()
    }

    #[test]
    fn unsorted_input_is_ordered_by_user_then_time() {
        let prep = prepare(&unsorted_two_users(), Cols::auto()).unwrap();
        let out = jump_lengths(&prep).unwrap();
        let jumps = out.column("jump_lengths").unwrap().list().unwrap();
        let uid = out.column("uid").unwrap().str().unwrap();

        let a = jumps.get_as_series(0).unwrap();
        let b = jumps.get_as_series(1).unwrap();
        assert_eq!(uid.get(0), Some("a"));
        assert_eq!(uid.get(1), Some("b"));
        assert_close(
            &a.f64().unwrap().into_no_null_iter().collect::<Vec<_>>(),
            &jump_lengths_km(vec![0.0, 0.0, 0.0], vec![0.0, 1.0, 3.0]).unwrap(),
        );
        assert_close(
            &b.f64().unwrap().into_no_null_iter().collect::<Vec<_>>(),
            &jump_lengths_km(vec![10.0, 10.0, 10.0], vec![0.0, 2.0, 4.0]).unwrap(),
        );
    }

    #[test]
    fn flat_output_concatenates_groups_in_arranged_order() {
        let prep = prepare(&unsorted_two_users(), Cols::auto()).unwrap();
        let flat = jump_lengths_flat(&prep).unwrap();
        let mut expected = jump_lengths_km(vec![0.0, 0.0, 0.0], vec![0.0, 1.0, 3.0]).unwrap();
        expected.extend(jump_lengths_km(vec![10.0, 10.0, 10.0], vec![0.0, 2.0, 4.0]).unwrap());
        assert_close(&flat, &expected);
    }

    #[test]
    fn already_arranged_input_gives_the_same_answer_as_unsorted_input() {
        let arranged = df![
            "uid" => ["a", "a", "a", "b", "b", "b"],
            "datetime" => [0i64, 1_000, 2_000, 0, 1_000, 2_000],
            "lat" => [0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
            "lng" => [0.0, 1.0, 3.0, 0.0, 2.0, 4.0],
        ]
        .unwrap();
        let from_arranged = jump_lengths_flat(&prepare(&arranged, Cols::auto()).unwrap()).unwrap();
        let from_unsorted =
            jump_lengths_flat(&prepare(&unsorted_two_users(), Cols::auto()).unwrap()).unwrap();
        assert_eq!(from_arranged, from_unsorted);
    }

    #[test]
    fn no_uid_column_returns_a_single_group_without_a_uid_column() {
        let df = df![
            "datetime" => [0i64, 1_000, 2_000],
            "lat" => [0.0, 0.0, 0.0],
            "lng" => [0.0, 1.0, 3.0],
        ]
        .unwrap();
        let out = jump_lengths(&prepare(&df, Cols::auto()).unwrap()).unwrap();
        assert_eq!(out.height(), 1);
        assert!(out.column("uid").is_err());
    }

    #[test]
    fn integer_uid_dtype_is_preserved_in_the_output() {
        let df = df![
            "uid" => [2i64, 1, 2, 1],
            "datetime" => [1_000i64, 0, 0, 1_000],
            "lat" => [10.0, 0.0, 10.0, 0.0],
            "lng" => [1.0, 0.0, 0.0, 1.0],
        ]
        .unwrap();
        let out = jump_lengths(&prepare(&df, Cols::auto()).unwrap()).unwrap();
        let uid = out.column("uid").unwrap();
        assert_eq!(uid.dtype(), &DataType::Int64);
        assert_eq!(
            uid.as_materialized_series()
                .i64()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
    }

    #[test]
    fn single_point_users_contribute_no_jumps() {
        let df = df![
            "uid" => [1i64, 2, 2],
            "datetime" => [0i64, 0, 1_000],
            "lat" => [0.0, 0.0, 0.0],
            "lng" => [0.0, 0.0, 1.0],
        ]
        .unwrap();
        let prep = prepare(&df, Cols::auto()).unwrap();
        assert_eq!(jump_lengths_flat(&prep).unwrap().len(), 1);
        let out = jump_lengths(&prep).unwrap();
        assert_eq!(out.height(), 2);
        assert_eq!(
            out.column("jump_lengths")
                .unwrap()
                .list()
                .unwrap()
                .get_as_series(0)
                .unwrap()
                .len(),
            0
        );
    }

    #[test]
    fn empty_input_produces_no_jumps() {
        let df = df![
            "uid" => Vec::<i64>::new(),
            "datetime" => Vec::<i64>::new(),
            "lat" => Vec::<f64>::new(),
            "lng" => Vec::<f64>::new(),
        ]
        .unwrap();
        let prep = prepare(&df, Cols::auto()).unwrap();
        assert!(jump_lengths_flat(&prep).unwrap().is_empty());
        assert_eq!(jump_lengths(&prep).unwrap().height(), 0);
    }
}
