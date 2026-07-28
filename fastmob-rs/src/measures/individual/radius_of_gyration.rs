//! Radius of gyration (km): the characteristic distance a user travels from
//! their centre of mass.

use fastmob_core::measures::individual::radius_of_gyration::radius_of_gyration_presorted_impl;
use polars::prelude::*;

use crate::error::FastmobRsError;
use crate::prepare::PreparedTrajectory;

/// One radius of gyration per user, in arranged group order, paired with a
/// validity mask.
///
/// The mask is `false` for users the kernel could not produce a value for; the
/// corresponding entry in the values vector is meaningless and must be
/// discarded rather than read.
pub fn radius_of_gyration_flat(prep: &PreparedTrajectory) -> (Vec<f64>, Vec<bool>) {
    radius_of_gyration_presorted_impl(prep.lat(), prep.lng(), prep.ends())
}

/// Per-user radius of gyration as a `[uid, "radius_of_gyration"]` frame.
///
/// Users without a valid value get a null rather than being dropped, so the
/// frame keeps one row per user. The uid column is omitted when the input frame
/// had no user column.
pub fn radius_of_gyration(prep: &PreparedTrajectory) -> Result<DataFrame, FastmobRsError> {
    let (values, valid) = radius_of_gyration_flat(prep);
    let masked: Vec<Option<f64>> = values
        .into_iter()
        .zip(valid)
        .map(|(value, is_valid)| is_valid.then_some(value))
        .collect();

    let mut columns = Vec::with_capacity(2);
    if let Some(labels) = prep.uid_labels()? {
        columns.push(labels.into_column());
    }
    columns.push(Series::new("radius_of_gyration".into(), masked).into_column());
    Ok(DataFrame::new(columns)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prepare::{Cols, prepare};

    #[test]
    fn single_point_user_has_zero_radius() {
        let df = df![
            "uid" => [1i64],
            "datetime" => [0i64],
            "lat" => [48.85],
            "lng" => [2.35],
        ]
        .unwrap();
        let (values, valid) = radius_of_gyration_flat(&prepare(&df, Cols::auto()).unwrap());
        assert_eq!(valid, vec![true]);
        assert_eq!(values, vec![0.0]);
    }

    #[test]
    fn two_distinct_points_have_a_positive_radius() {
        let df = df![
            "uid" => [1i64, 1],
            "datetime" => [0i64, 1],
            "lat" => [48.85, 48.86],
            "lng" => [2.35, 2.36],
        ]
        .unwrap();
        let (values, _) = radius_of_gyration_flat(&prepare(&df, Cols::auto()).unwrap());
        assert!(values[0] > 0.0);
    }

    #[test]
    fn results_are_one_row_per_user_in_arranged_order() {
        let df = df![
            "uid" => ["b", "a", "b", "a"],
            "datetime" => [1_000i64, 1_000, 0, 0],
            "lat" => [10.0, 0.0, 10.0, 0.0],
            "lng" => [1.0, 1.0, 0.0, 0.0],
        ]
        .unwrap();
        let out = radius_of_gyration(&prepare(&df, Cols::auto()).unwrap()).unwrap();
        assert_eq!(out.height(), 2);
        assert_eq!(
            out.column("uid")
                .unwrap()
                .str()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }

    #[test]
    fn row_order_does_not_change_the_result() {
        let arranged = df![
            "uid" => [1i64, 1, 1],
            "datetime" => [0i64, 1, 2],
            "lat" => [0.0, 1.0, 2.0],
            "lng" => [0.0, 1.0, 2.0],
        ]
        .unwrap();
        let shuffled = df![
            "uid" => [1i64, 1, 1],
            "datetime" => [2i64, 0, 1],
            "lat" => [2.0, 0.0, 1.0],
            "lng" => [2.0, 0.0, 1.0],
        ]
        .unwrap();
        let (from_arranged, _) =
            radius_of_gyration_flat(&prepare(&arranged, Cols::auto()).unwrap());
        let (from_shuffled, _) =
            radius_of_gyration_flat(&prepare(&shuffled, Cols::auto()).unwrap());
        assert_eq!(from_arranged, from_shuffled);
    }

    #[test]
    fn empty_input_produces_no_rows() {
        let df = df![
            "uid" => Vec::<i64>::new(),
            "datetime" => Vec::<i64>::new(),
            "lat" => Vec::<f64>::new(),
            "lng" => Vec::<f64>::new(),
        ]
        .unwrap();
        let prep = prepare(&df, Cols::auto()).unwrap();
        assert!(radius_of_gyration_flat(&prep).0.is_empty());
        assert_eq!(radius_of_gyration(&prep).unwrap().height(), 0);
    }
}
