//! Activity transition matrices and daily activity distributions.
//!
//! These operate on *visit* tables (one row per stay, with a purpose/activity
//! label) rather than raw fixes, so they take the frame as the caller arranged
//! it instead of re-arranging it: visit order is part of the input's meaning,
//! and the relevant time column is often an arrival/departure pair rather than
//! a single datetime.

use fastmob_core::measures::individual::activity::{
    activity_transition_counts, daily_activity_percentages,
};
use polars::prelude::*;

use crate::error::FastmobRsError;
use crate::prepare::categorical::{factorize, presorted_group_ends};

/// Percentage of consecutive-visit transitions from one activity to another.
///
/// `df` must already be grouped by user and ordered within each user; the
/// grouping is read from the existing runs in `uid_col`. Returns the category
/// list and `matrix[from][to]` as a percentage of all transitions across every
/// user.
pub fn activity_transition_matrix(
    df: &DataFrame,
    uid_col: &str,
    activity_col: &str,
) -> Result<(Vec<String>, Vec<Vec<f64>>), FastmobRsError> {
    let activities = factorize(df.column(activity_col)?.as_materialized_series())?;
    let categories = activities.categories;
    let n = categories.len();
    if n == 0 {
        return Ok((categories, Vec::new()));
    }

    let ends = presorted_group_ends(df.column(uid_col)?.as_materialized_series())?;
    let indices: Vec<usize> = (0..activities.codes.len()).collect();
    let counts = activity_transition_counts(&activities.codes, &indices, &ends, n)
        .map_err(FastmobRsError::Core)?;

    let total: u64 = counts.iter().sum();
    let mut matrix = vec![vec![0.0f64; n]; n];
    if total > 0 {
        for (from, row) in matrix.iter_mut().enumerate() {
            for (to, cell) in row.iter_mut().enumerate() {
                *cell = counts[from * n + to] as f64 / total as f64 * 100.0;
            }
        }
    }
    Ok((categories, matrix))
}

/// Share of each activity per time-of-day bin.
///
/// `start_minutes`/`end_minutes` are minute-of-day in `0..1440`; a visit with no
/// resolvable end passes the same value for both, occupying only its start
/// bin. Returns the category list and `matrix[category][bin]` as percentages
/// per bin column, with `NaN` for a bin no category was active in.
pub fn daily_activity_distribution(
    activities: &Series,
    start_minutes: &[i64],
    end_minutes: &[i64],
    valid_rows: &[bool],
    bin_size_minutes: usize,
) -> Result<(Vec<String>, Vec<Vec<f64>>), FastmobRsError> {
    if bin_size_minutes == 0 || 1440 % bin_size_minutes != 0 {
        return Err(FastmobRsError::Core(format!(
            "bin_size_minutes must divide 1440, got {bin_size_minutes}"
        )));
    }

    let factorized = factorize(activities)?;
    let categories = factorized.categories;
    let n = categories.len();
    if n == 0 {
        return Ok((categories, Vec::new()));
    }

    let bins = 1440 / bin_size_minutes;
    let flat = daily_activity_percentages(
        &factorized.codes,
        start_minutes,
        end_minutes,
        valid_rows,
        n,
        bin_size_minutes,
    )
    .map_err(FastmobRsError::Core)?;

    let matrix = (0..n)
        .map(|category| flat[category * bins..(category + 1) * bins].to_vec())
        .collect();
    Ok((categories, matrix))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn visits(uids: &[i64], purposes: &[&str]) -> DataFrame {
        df![
            "uid" => uids.to_vec(),
            "purpose" => purposes.to_vec(),
        ]
        .unwrap()
    }

    #[test]
    fn transitions_are_percentages_of_every_users_transitions() {
        // One user: home -> work -> home. Two transitions, 50% each.
        let df = visits(&[1, 1, 1], &["home", "work", "home"]);
        let (categories, matrix) = activity_transition_matrix(&df, "uid", "purpose").unwrap();
        assert_eq!(categories, vec!["home", "work"]);
        assert_eq!(matrix[0][1], 50.0);
        assert_eq!(matrix[1][0], 50.0);
        assert_eq!(matrix[0][0], 0.0);
    }

    #[test]
    fn transitions_never_bridge_two_users() {
        // Without grouping, user 1's last visit would pair with user 2's first.
        let df = visits(&[1, 1, 2, 2], &["home", "work", "errand", "home"]);
        let (categories, matrix) = activity_transition_matrix(&df, "uid", "purpose").unwrap();
        let home = categories.iter().position(|c| c == "home").unwrap();
        let errand = categories.iter().position(|c| c == "errand").unwrap();
        let work = categories.iter().position(|c| c == "work").unwrap();
        assert_eq!(matrix[work][errand], 0.0, "work->errand crosses users");
        assert_eq!(matrix[home][work], 50.0);
        assert_eq!(matrix[errand][home], 50.0);
    }

    #[test]
    fn a_single_visit_user_contributes_no_transitions() {
        let df = visits(&[1, 2, 2], &["home", "home", "work"]);
        let (_, matrix) = activity_transition_matrix(&df, "uid", "purpose").unwrap();
        let total: f64 = matrix.iter().flatten().sum();
        assert!((total - 100.0).abs() < 1e-12, "got {total}");
    }

    #[test]
    fn an_empty_visit_table_has_no_categories() {
        let df = df!["uid" => Vec::<i64>::new(), "purpose" => Vec::<&str>::new()].unwrap();
        let (categories, matrix) = activity_transition_matrix(&df, "uid", "purpose").unwrap();
        assert!(categories.is_empty());
        assert!(matrix.is_empty());
    }

    #[test]
    fn daily_distribution_splits_time_across_bins() {
        let activities = Series::new("purpose".into(), ["home", "work"]);
        // 60-minute bins: home occupies bin 0, work occupies bin 1.
        let (categories, matrix) =
            daily_activity_distribution(&activities, &[0, 60], &[59, 119], &[true, true], 60)
                .unwrap();
        assert_eq!(categories, vec!["home", "work"]);
        assert_eq!(matrix.len(), 2);
        assert_eq!(matrix[0].len(), 24);
        assert_eq!(matrix[0][0], 100.0);
        assert_eq!(matrix[1][1], 100.0);
    }

    #[test]
    fn a_bin_size_that_does_not_divide_the_day_is_rejected() {
        let activities = Series::new("purpose".into(), ["home"]);
        assert!(
            daily_activity_distribution(&activities, &[0], &[10], &[true], 7).is_err(),
            "7 does not divide 1440"
        );
        assert!(daily_activity_distribution(&activities, &[0], &[10], &[true], 0).is_err());
    }
}
