//! Douglas-Peucker simplification, delegating the actual RDP computation to
//! `geo::SimplifyIdx` over locally-projected planar coordinates.

use geo::{LineString, SimplifyIdx};

use crate::utils::haversine::project_local_planar_km;

/// Return the 0-based local indices retained by the Ramer-Douglas-Peucker
/// algorithm for a single user's trajectory.
///
/// Coordinates are projected to local planar kilometre coordinates via
/// [`project_local_planar_km`] first, so `epsilon_km` is a genuine
/// perpendicular-distance tolerance in kilometres rather than a raw
/// lat/lng-degree tolerance.
///
/// @usedBy `fastmob-core/src/preprocessing/simplify/mod.rs::simplify_user_slice`
/// (method = `douglas_peucker`).
pub fn douglas_peucker_indices(lats: &[f64], lngs: &[f64], epsilon_km: f64) -> Vec<usize> {
    let coords = project_local_planar_km(lats, lngs);
    let line = LineString::from(coords);
    line.simplify_idx(epsilon_km)
}
