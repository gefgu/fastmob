//! [`PreparedTrajectory`]: clean and arrange once, then run many measures.
//!
//! Every measure in this crate is a pure presorted kernel over a
//! `&PreparedTrajectory`. That split exists because the arrange step dominates
//! the cost of any single measure: at 4M Brightkite rows, `prepare` is 72 ms of
//! a 127 ms jumps-and-radius call, while a second measure over the same handle
//! costs ~7 ms. Paying the arrange once and reusing it is worth far more than
//! optimising any individual kernel.
//!
//! [`PreparedTrajectory::filter`] extends the same idea to filtered subsets: a
//! row predicate over an arranged frame yields an arranged frame, so N filters
//! cost one arrange plus N linear boundary recomputations instead of N sorts.

use polars::prelude::*;

use super::arrange;
use super::columns::{Cols, Requires, ResolvedColumns, resolve};
use super::timestamps;
use super::uid_codes;
use crate::error::FastmobRsError;

/// A cleaned, `(user, time)`-arranged trajectory with cached group boundaries.
///
/// Invariants established by [`prepare`] and preserved by [`Self::filter`]:
/// the latitude and longitude columns are `Float64`, contain no nulls and no
/// non-finite values, and are single-chunked; rows are grouped by user and
/// non-decreasing in time within each group.
#[derive(Debug, Clone)]
pub struct PreparedTrajectory {
    frame: DataFrame,
    cols: ResolvedColumns,
    codes: Vec<u32>,
    ends: Vec<usize>,
    starts: Vec<usize>,
    timestamps_ms: Vec<i64>,
    spatial: bool,
}

fn starts_from_ends(ends: &[usize]) -> Vec<usize> {
    let mut starts = Vec::with_capacity(ends.len());
    let mut previous = 0usize;
    for &end in ends {
        starts.push(previous);
        previous = end;
    }
    starts
}

fn contiguous_f64<'a>(frame: &'a DataFrame, name: &str) -> &'a [f64] {
    frame
        .column(name)
        .expect("prepare invariant: resolved column is present")
        .f64()
        .expect("prepare invariant: coordinate column is Float64")
        .cont_slice()
        .expect("prepare invariant: coordinate column is single-chunked and non-null")
}

impl PreparedTrajectory {
    /// Whether this trajectory carries usable coordinates, i.e. came from
    /// [`prepare`] rather than [`prepare_temporal`].
    pub fn has_coordinates(&self) -> bool {
        self.spatial
    }

    /// Arranged latitudes, borrowed directly from the Polars buffer.
    ///
    /// Panics on a [`prepare_temporal`] trajectory, which has not cleaned its
    /// coordinates and may hold nulls or non-finite values.
    pub fn lat(&self) -> &[f64] {
        self.assert_spatial();
        contiguous_f64(
            &self.frame,
            self.cols
                .lat
                .as_deref()
                .expect("spatial trajectory has lat"),
        )
    }

    /// Arranged longitudes, borrowed directly from the Polars buffer.
    ///
    /// Panics on a [`prepare_temporal`] trajectory, for the same reason as
    /// [`Self::lat`].
    pub fn lng(&self) -> &[f64] {
        self.assert_spatial();
        contiguous_f64(
            &self.frame,
            self.cols
                .lng
                .as_deref()
                .expect("spatial trajectory has lng"),
        )
    }

    fn assert_spatial(&self) {
        assert!(
            self.spatial,
            "coordinates were requested from a trajectory built with prepare_temporal, \
             which neither requires nor cleans them; use prepare() instead"
        );
    }

    /// Arranged millisecond timestamps.
    pub fn timestamps_ms(&self) -> &[i64] {
        &self.timestamps_ms
    }

    /// Exclusive end offset of each user's contiguous run — fastmob-core's
    /// `ends` convention.
    pub fn ends(&self) -> &[usize] {
        &self.ends
    }

    /// Inclusive start offset of each user's contiguous run.
    pub fn starts(&self) -> &[usize] {
        &self.starts
    }

    /// Number of users (one group when the frame has no uid column).
    pub fn num_users(&self) -> usize {
        self.ends.len()
    }

    pub fn height(&self) -> usize {
        self.frame.height()
    }

    pub fn is_empty(&self) -> bool {
        self.frame.height() == 0
    }

    /// The cleaned, arranged frame.
    pub fn frame(&self) -> &DataFrame {
        &self.frame
    }

    pub fn columns(&self) -> &ResolvedColumns {
        &self.cols
    }

    /// One uid value per group, in arranged group order. `None` when the frame
    /// has no uid column.
    pub fn uid_labels(&self) -> Result<Option<Series>, FastmobRsError> {
        let Some(uid_col) = self.cols.uid.as_deref() else {
            return Ok(None);
        };
        let rows: Vec<IdxSize> = self.starts.iter().map(|&start| start as IdxSize).collect();
        let mut labels = self
            .frame
            .column(uid_col)?
            .as_materialized_series()
            .take_slice(&rows)?;
        labels.rename(uid_col.into());
        Ok(Some(labels))
    }

    /// Filter rows while preserving the arrangement.
    ///
    /// The mask is applied to the already-arranged frame, so the surviving rows
    /// remain grouped by user and time-ordered; only the group boundaries need
    /// recomputing. This is what lets a caller with many filters over the same
    /// trajectory pay for the arrange exactly once.
    pub fn filter(&self, mask: &BooleanChunked) -> Result<Self, FastmobRsError> {
        let frame = self.frame.filter(mask)?;
        let keep: Vec<bool> = mask
            .into_iter()
            .map(|value| value.unwrap_or(false))
            .collect();

        let codes: Vec<u32> = self
            .codes
            .iter()
            .zip(&keep)
            .filter_map(|(&code, &keep)| keep.then_some(code))
            .collect();
        let timestamps_ms: Vec<i64> = self
            .timestamps_ms
            .iter()
            .zip(&keep)
            .filter_map(|(&value, &keep)| keep.then_some(value))
            .collect();

        let ends = super::arrange::ends_from_arranged_codes(&codes);
        let starts = starts_from_ends(&ends);
        Ok(Self {
            frame,
            cols: self.cols.clone(),
            codes,
            ends,
            starts,
            timestamps_ms,
            spatial: self.spatial,
        })
    }
}

/// Clean and arrange a trajectory, resolving column names automatically.
///
/// Requires latitude and longitude, and drops rows whose coordinates are null
/// or non-finite. Use [`prepare_temporal`] for measures that only need time.
pub fn prepare(df: &DataFrame, cols: Cols) -> Result<PreparedTrajectory, FastmobRsError> {
    prepare_keeping(df, cols, &[])
}

/// Like [`prepare`], but also carries the named extra columns through the
/// cleaning and arranging steps so downstream measures can use them.
pub fn prepare_keeping(
    df: &DataFrame,
    cols: Cols,
    extra: &[&str],
) -> Result<PreparedTrajectory, FastmobRsError> {
    prepare_inner(df, cols, extra, Requires::Coordinates)
}

/// Arrange a trajectory for time-only measures such as waiting times.
///
/// Coordinates are neither required nor cleaned: a fix with a missing latitude
/// still happened at a real time, so dropping it would invent a longer gap
/// between its neighbours. This matches the Python `waiting_times`, which never
/// filters on coordinates either.
///
/// [`PreparedTrajectory::lat`] and [`PreparedTrajectory::lng`] must not be
/// called on the result; check [`PreparedTrajectory::has_coordinates`] first if
/// the provenance is not obvious at the call site.
pub fn prepare_temporal(df: &DataFrame, cols: Cols) -> Result<PreparedTrajectory, FastmobRsError> {
    prepare_inner(df, cols, &[], Requires::TimeOnly)
}

fn prepare_inner(
    df: &DataFrame,
    cols: Cols,
    extra: &[&str],
    requires: Requires,
) -> Result<PreparedTrajectory, FastmobRsError> {
    let cols = resolve(df, &cols, requires)?;
    let schema = df.schema();
    let spatial = requires == Requires::Coordinates;

    let mut projection = Vec::new();
    let mut required = Vec::new();
    if let Some(uid_col) = cols.uid.as_deref() {
        projection.push(col(uid_col));
        required.push(uid_col.to_string());
    }
    if let Some(lat_col) = cols.lat.as_deref() {
        projection.push(col(lat_col).cast(DataType::Float64));
    }
    if let Some(lng_col) = cols.lng.as_deref() {
        projection.push(col(lng_col).cast(DataType::Float64));
    }
    projection.push(timestamps::datetime_expr(schema, &cols.datetime).alias(&cols.datetime));
    required.push(cols.datetime.clone());
    if spatial {
        required.extend([
            cols.lat.clone().expect("spatial preparation resolved lat"),
            cols.lng.clone().expect("spatial preparation resolved lng"),
        ]);
    }
    for name in extra {
        if !required.iter().any(|existing| existing == name) && cols.uid.as_deref() != Some(name) {
            projection.push(col(*name));
        }
    }

    let cleaned = df.clone().lazy().select(projection);
    let cleaned = cleaned.drop_nulls(Some(cols_selector(&required)));
    let cleaned = if spatial {
        cleaned.filter(
            col(cols
                .lat
                .as_deref()
                .expect("spatial preparation resolved lat"))
            .is_finite()
            .and(
                col(cols
                    .lng
                    .as_deref()
                    .expect("spatial preparation resolved lng"))
                .is_finite(),
            ),
        )
    } else {
        cleaned
    };
    let cleaned = cleaned.collect()?;

    let len = cleaned.height();
    let datetime_series = cleaned.column(&cols.datetime)?.as_materialized_series();
    let ordering = timestamps::ordering_key(datetime_series, &cols.datetime)?;

    let built = match cols.uid.as_deref() {
        Some(uid_col) => {
            let series = cleaned.column(uid_col)?.as_materialized_series();
            uid_codes::build(series, len)?
        }
        None => uid_codes::UidCodes {
            codes: vec![0; len],
            num_codes: usize::from(len > 0),
        },
    };

    let arrangement = arrange::arrange(&built.codes, &ordering, built.num_codes);

    let (frame, codes) = match &arrangement.permutation {
        None => (cleaned, built.codes),
        Some(permutation) => {
            let rows: Vec<IdxSize> = permutation.iter().map(|&row| row as IdxSize).collect();
            let frame = cleaned.take(&IdxCa::from_vec("__fastmob_take__".into(), rows))?;
            let codes = permutation
                .iter()
                .map(|&row| built.codes[row as usize])
                .collect();
            (frame, codes)
        }
    };
    let mut frame = frame;
    frame.as_single_chunk_par();

    let datetime_series = frame.column(&cols.datetime)?.as_materialized_series();
    let timestamps_ms = timestamps::milliseconds(datetime_series, &cols.datetime)?;

    let ends = arrangement.ends;
    let starts = starts_from_ends(&ends);
    Ok(PreparedTrajectory {
        frame,
        cols,
        codes,
        ends,
        starts,
        timestamps_ms,
        spatial,
    })
}

fn cols_selector(names: &[String]) -> Selector {
    cols(names.iter().map(|name| name.as_str()).collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> DataFrame {
        df![
            "uid" => ["b", "a", "b", "a", "a", "b"],
            "datetime" => [2_000i64, 2_000, 0, 0, 1_000, 1_000],
            "lat" => [10.0, 0.0, 10.0, 0.0, 0.0, 10.0],
            "lng" => [4.0, 3.0, 0.0, 0.0, 1.0, 2.0],
        ]
        .unwrap()
    }

    #[test]
    fn prepare_groups_by_user_and_orders_by_time() {
        let prep = prepare(&sample(), Cols::auto()).unwrap();
        assert_eq!(prep.ends(), &[3, 6]);
        assert_eq!(prep.starts(), &[0, 3]);
        assert_eq!(prep.lat(), &[0.0, 0.0, 0.0, 10.0, 10.0, 10.0]);
        assert_eq!(prep.lng(), &[0.0, 1.0, 3.0, 0.0, 2.0, 4.0]);
    }

    #[test]
    fn uid_labels_are_one_per_group_in_arranged_order() {
        let prep = prepare(&sample(), Cols::auto()).unwrap();
        let labels = prep.uid_labels().unwrap().unwrap();
        assert_eq!(
            labels
                .str()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }

    #[test]
    fn missing_uid_column_yields_a_single_time_ordered_group() {
        let df = df![
            "datetime" => [2_000i64, 0, 1_000],
            "lat" => [0.0, 0.0, 0.0],
            "lng" => [3.0, 0.0, 1.0],
        ]
        .unwrap();
        let prep = prepare(&df, Cols::auto()).unwrap();
        assert_eq!(prep.num_users(), 1);
        assert_eq!(prep.ends(), &[3]);
        assert_eq!(prep.lng(), &[0.0, 1.0, 3.0]);
        assert!(prep.uid_labels().unwrap().is_none());
    }

    #[test]
    fn null_and_non_finite_coordinates_are_dropped() {
        let df = df![
            "uid" => [1i64, 1, 1, 1],
            "datetime" => [0i64, 1, 2, 3],
            "lat" => [Some(0.0), None, Some(f64::NAN), Some(1.0)],
            "lng" => [Some(0.0), Some(1.0), Some(2.0), Some(1.0)],
        ]
        .unwrap();
        let prep = prepare(&df, Cols::auto()).unwrap();
        assert_eq!(prep.height(), 2);
        assert_eq!(prep.lat(), &[0.0, 1.0]);
    }

    #[test]
    fn filter_preserves_the_arrangement_and_rebuilds_boundaries() {
        let prep = prepare(&sample(), Cols::auto()).unwrap();
        // Drop every row of user "a" except the first.
        let mask = BooleanChunked::new("mask".into(), [true, false, false, true, true, true]);
        let filtered = prep.filter(&mask).unwrap();
        assert_eq!(filtered.height(), 4);
        assert_eq!(filtered.ends(), &[1, 4]);
        assert_eq!(filtered.starts(), &[0, 1]);
        let labels = filtered.uid_labels().unwrap().unwrap();
        assert_eq!(
            labels
                .str()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
    }

    #[test]
    fn filtering_out_an_entire_user_removes_its_group() {
        let prep = prepare(&sample(), Cols::auto()).unwrap();
        let mask = BooleanChunked::new("mask".into(), [false, false, false, true, true, true]);
        let filtered = prep.filter(&mask).unwrap();
        assert_eq!(filtered.num_users(), 1);
        assert_eq!(filtered.ends(), &[3]);
    }

    #[test]
    fn extra_columns_are_carried_through_the_arrangement() {
        let df = df![
            "uid" => ["a", "a"],
            "datetime" => [1_000i64, 0],
            "lat" => [0.0, 0.0],
            "lng" => [1.0, 0.0],
            "location_id" => ["x", "y"],
        ]
        .unwrap();
        let prep = prepare_keeping(&df, Cols::auto(), &["location_id"]).unwrap();
        let location = prep.frame().column("location_id").unwrap();
        assert_eq!(
            location
                .as_materialized_series()
                .str()
                .unwrap()
                .into_no_null_iter()
                .collect::<Vec<_>>(),
            vec!["y", "x"]
        );
    }

    #[test]
    fn empty_input_prepares_to_an_empty_trajectory() {
        let df = df![
            "uid" => Vec::<i64>::new(),
            "datetime" => Vec::<i64>::new(),
            "lat" => Vec::<f64>::new(),
            "lng" => Vec::<f64>::new(),
        ]
        .unwrap();
        let prep = prepare(&df, Cols::auto()).unwrap();
        assert!(prep.is_empty());
        assert_eq!(prep.num_users(), 0);
    }
}
