//! Trajectory column resolution.
//!
//! The candidate lists mirror the priority order documented in fastmob's
//! `CLAUDE.md` and implemented by the Python `_detect_trajectory_columns`, so a
//! frame that auto-detects in Python auto-detects identically here.

use polars::prelude::*;

use crate::error::FastmobRsError;

pub(crate) const DATETIME_CANDIDATES: &[&str] = &["datetime", "check-in_time", "timestamp", "time"];
pub(crate) const LAT_CANDIDATES: &[&str] = &["lat", "latitude"];
pub(crate) const LNG_CANDIDATES: &[&str] = &["lng", "lon", "longitude", "long"];
pub(crate) const UID_CANDIDATES: &[&str] = &["uid", "user", "user_id"];

/// Explicit column overrides; any field left `None` is auto-detected.
#[derive(Debug, Clone, Default)]
pub struct Cols {
    pub datetime: Option<String>,
    pub lat: Option<String>,
    pub lng: Option<String>,
    pub uid: Option<String>,
}

impl Cols {
    /// Auto-detect every column from the standard fastmob name priority lists.
    pub fn auto() -> Self {
        Self::default()
    }

    pub fn datetime(mut self, name: impl Into<String>) -> Self {
        self.datetime = Some(name.into());
        self
    }

    pub fn lat(mut self, name: impl Into<String>) -> Self {
        self.lat = Some(name.into());
        self
    }

    pub fn lng(mut self, name: impl Into<String>) -> Self {
        self.lng = Some(name.into());
        self
    }

    pub fn uid(mut self, name: impl Into<String>) -> Self {
        self.uid = Some(name.into());
        self
    }
}

/// What a preparation needs from the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Requires {
    /// Coordinates must be present; rows with unusable ones are dropped.
    Coordinates,
    /// Only time matters. Coordinates are carried through if present but never
    /// required, and never used as a reason to drop a row -- a fix with a bad
    /// latitude still happened at a real time.
    TimeOnly,
}

/// Column names after resolution.
///
/// `uid` stays optional: a frame without a user column is treated as a single
/// individual, matching the Python API. `lat`/`lng` are optional only for
/// time-only preparations.
#[derive(Debug, Clone)]
pub struct ResolvedColumns {
    pub datetime: String,
    pub lat: Option<String>,
    pub lng: Option<String>,
    pub uid: Option<String>,
}

fn pick_existing(columns: &[String], candidates: &[&str]) -> Option<String> {
    candidates
        .iter()
        .find(|candidate| columns.iter().any(|column| column == **candidate))
        .map(|value| (*value).to_string())
}

pub(crate) fn resolve(
    df: &DataFrame,
    cols: &Cols,
    requires: Requires,
) -> Result<ResolvedColumns, FastmobRsError> {
    let columns = df
        .get_column_names()
        .iter()
        .map(|name| name.as_str().to_string())
        .collect::<Vec<_>>();

    let datetime = cols
        .datetime
        .clone()
        .or_else(|| pick_existing(&columns, DATETIME_CANDIDATES));
    let lat = cols
        .lat
        .clone()
        .or_else(|| pick_existing(&columns, LAT_CANDIDATES));
    let lng = cols
        .lng
        .clone()
        .or_else(|| pick_existing(&columns, LNG_CANDIDATES));
    let uid = cols
        .uid
        .clone()
        .or_else(|| pick_existing(&columns, UID_CANDIDATES));

    let mut required: Vec<(&str, Option<&String>)> = vec![("datetime", datetime.as_ref())];
    if requires == Requires::Coordinates {
        required.push(("latitude", lat.as_ref()));
        required.push(("longitude", lng.as_ref()));
    }
    let missing = required
        .into_iter()
        .filter_map(|(name, value)| value.is_none().then_some(name))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        return Err(FastmobRsError::MissingColumns(missing.join(", "), columns));
    }

    Ok(ResolvedColumns {
        datetime: datetime.unwrap(),
        lat,
        lng,
        uid,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_detects_standard_names() {
        let df = df![
            "uid" => [1i64],
            "datetime" => [0i64],
            "lat" => [0.0],
            "lng" => [0.0],
        ]
        .unwrap();
        let resolved = resolve(&df, &Cols::auto(), Requires::Coordinates).unwrap();
        assert_eq!(resolved.datetime, "datetime");
        assert_eq!(resolved.lat.as_deref(), Some("lat"));
        assert_eq!(resolved.lng.as_deref(), Some("lng"));
        assert_eq!(resolved.uid.as_deref(), Some("uid"));
    }

    #[test]
    fn auto_detects_alternate_names_in_priority_order() {
        let df = df![
            "user_id" => [1i64],
            "check-in_time" => [0i64],
            "latitude" => [0.0],
            "longitude" => [0.0],
        ]
        .unwrap();
        let resolved = resolve(&df, &Cols::auto(), Requires::Coordinates).unwrap();
        assert_eq!(resolved.datetime, "check-in_time");
        assert_eq!(resolved.lat.as_deref(), Some("latitude"));
        assert_eq!(resolved.lng.as_deref(), Some("longitude"));
        assert_eq!(resolved.uid.as_deref(), Some("user_id"));
    }

    #[test]
    fn missing_uid_is_allowed_missing_coordinates_are_not() {
        let df = df!["datetime" => [0i64], "lat" => [0.0], "lng" => [0.0]].unwrap();
        assert!(
            resolve(&df, &Cols::auto(), Requires::Coordinates)
                .unwrap()
                .uid
                .is_none()
        );

        let df = df!["datetime" => [0i64], "lat" => [0.0]].unwrap();
        assert!(matches!(
            resolve(&df, &Cols::auto(), Requires::Coordinates),
            Err(FastmobRsError::MissingColumns(_, _))
        ));
    }

    #[test]
    fn time_only_resolution_tolerates_absent_coordinates() {
        let df = df!["uid" => [1i64], "datetime" => [0i64]].unwrap();
        let resolved = resolve(&df, &Cols::auto(), Requires::TimeOnly).unwrap();
        assert_eq!(resolved.datetime, "datetime");
        assert!(resolved.lat.is_none());
        assert!(resolved.lng.is_none());
        // The datetime column is still mandatory.
        let df = df!["uid" => [1i64]].unwrap();
        assert!(resolve(&df, &Cols::auto(), Requires::TimeOnly).is_err());
    }

    #[test]
    fn explicit_overrides_win_over_detection() {
        let df = df![
            "uid" => [1i64],
            "datetime" => [0i64],
            "lat" => [0.0],
            "lng" => [0.0],
            "other_lat" => [1.0],
        ]
        .unwrap();
        let resolved = resolve(&df, &Cols::auto().lat("other_lat"), Requires::Coordinates).unwrap();
        assert_eq!(resolved.lat.as_deref(), Some("other_lat"));
    }
}
