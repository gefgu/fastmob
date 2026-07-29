//! Haversine distance as a Polars expression.
//!
//! `fastmob-core` exposes Haversine as a scalar and as batch kernels over
//! slices, which is what every measure in this crate uses. Neither fits inside
//! a lazy Polars pipeline, where the distance has to be computed as part of a
//! larger query without materialising columns first.
//!
//! This exists so that need has one sanctioned implementation rather than being
//! re-derived by each caller that hits it. Prefer the kernels whenever the data
//! is already in hand: only reach for this when the computation genuinely has
//! to stay inside a `LazyFrame`.

use polars::prelude::*;

/// Mean Earth radius in kilometres, matching `fastmob-core`'s scalar kernel.
const EARTH_RADIUS_KM: f64 = 6371.0088;

/// Great-circle distance in kilometres between two coordinate pairs.
///
/// Nulls propagate: a null in any input yields a null distance, rather than
/// being silently treated as a value. That matters because the first row of a
/// per-group `shift()` is always null, and quietly clamping it produces a
/// spurious antipodal jump of roughly 20,015 km on every group.
pub fn haversine_km_expr(lat1: Expr, lng1: Expr, lat2: Expr, lng2: Expr) -> Expr {
    let lat1_rad = lat1.radians();
    let lng1_rad = lng1.radians();
    let lat2_rad = lat2.radians();
    let lng2_rad = lng2.radians();

    let dlat = lat2_rad.clone() - lat1_rad.clone();
    let dlng = lng2_rad - lng1_rad;
    let a = (dlat / lit(2.0)).sin().pow(2)
        + lat1_rad.cos() * lat2_rad.cos() * (dlng / lit(2.0)).sin().pow(2);

    lit(EARTH_RADIUS_KM) * lit(2.0) * a.sqrt().clip_max(lit(1.0)).arcsin()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn distance(lat1: f64, lng1: f64, lat2: f64, lng2: f64) -> Option<f64> {
        let df = df!["lat1" => [lat1], "lng1" => [lng1], "lat2" => [lat2], "lng2" => [lng2]]
            .unwrap()
            .lazy()
            .select([
                haversine_km_expr(col("lat1"), col("lng1"), col("lat2"), col("lng2")).alias("km"),
            ])
            .collect()
            .unwrap();
        df.column("km").unwrap().f64().unwrap().get(0)
    }

    #[test]
    fn a_point_is_zero_distance_from_itself() {
        assert_eq!(distance(48.85, 2.35, 48.85, 2.35), Some(0.0));
    }

    #[test]
    fn it_agrees_with_the_scalar_kernel() {
        let expected = fastmob_core::utils::haversine::haversine_km(48.85, 2.35, 40.71, -74.01);
        let actual = distance(48.85, 2.35, 40.71, -74.01).unwrap();
        assert!(
            (actual - expected).abs() < 1e-9,
            "expr {actual} vs kernel {expected}"
        );
    }

    #[test]
    fn one_degree_of_latitude_is_about_111_km() {
        let value = distance(0.0, 0.0, 1.0, 0.0).unwrap();
        assert!((value - 111.19).abs() < 0.01, "got {value}");
    }

    /// A null input must stay null. Clamping it instead -- the bug this
    /// expression exists to avoid -- turns every group's first row into a
    /// ~20,015 km jump.
    #[test]
    fn nulls_propagate_instead_of_becoming_an_antipodal_jump() {
        let df = df![
            "lat1" => [None, Some(48.85)],
            "lng1" => [None, Some(2.35)],
            "lat2" => [Some(48.86), Some(48.86)],
            "lng2" => [Some(2.36), Some(2.36)],
        ]
        .unwrap()
        .lazy()
        .select([haversine_km_expr(col("lat1"), col("lng1"), col("lat2"), col("lng2")).alias("km")])
        .collect()
        .unwrap();
        let km = df.column("km").unwrap().f64().unwrap();
        assert_eq!(km.get(0), None, "null input must yield null, not ~20015 km");
        assert!(km.get(1).unwrap() > 0.0);
    }
}
