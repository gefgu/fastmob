//! Batch lat/lng -> H3 cell conversion.
//!
//! `h3o`'s per-point `LatLng::to_cell` is the same algorithm h3-py's scalar
//! `latlng_to_cell` uses, but calling it from a Python `for`/list-comprehension
//! loop pays Python-call overhead per row. Trajectory datasets with tens to
//! hundreds of millions of rows need this converted in bulk, so it's run here
//! across all rows in one call, in parallel.

use h3o::{CellIndex, LatLng, Resolution};
use rayon::prelude::*;

/// `u64::MAX` is not a valid H3 cell index (the top reserved bits are never
/// all-1 for a valid cell), so it doubles as the "invalid input" sentinel for
/// non-finite/out-of-range lat/lng or an explicitly-masked-out row -- callers
/// are expected to treat it the same way they already treat a missing/NaN
/// location.
pub const INVALID_CELL: u64 = u64::MAX;

/// Converts `(lat, lng)` pairs (degrees) to H3 cell indices at `resolution`,
/// in parallel. Invalid coordinates map to [`INVALID_CELL`] rather than
/// failing the whole batch, since real-world mobility data routinely has a
/// few bad rows mixed into an otherwise valid column.
///
/// `valid_rows`, when provided, marks rows that must be treated as invalid
/// regardless of their (lat, lng) bit pattern -- this lets the Arrow binding
/// honor real Arrow null slots rather than relying on NaN sentinel bytes,
/// which a null Arrow slot's underlying buffer is not guaranteed to contain.
pub fn batch_latlng_to_cells(
    lats: &[f64],
    lngs: &[f64],
    resolution: Resolution,
    valid_rows: Option<&[bool]>,
) -> Vec<u64> {
    lats.par_iter()
        .zip(lngs.par_iter())
        .enumerate()
        .map(|(idx, (&lat, &lng))| {
            if valid_rows.is_some_and(|valid| !valid[idx]) {
                return INVALID_CELL;
            }
            LatLng::new(lat, lng)
                .map(|ll| u64::from(ll.to_cell(resolution)))
                .unwrap_or(INVALID_CELL)
        })
        .collect()
}

/// Converts `(lat, lng)` pairs directly to `(cell, center_lat, center_lng)`
/// in one parallel pass.
///
/// Equivalent to calling [`batch_latlng_to_cells`] followed by
/// [`batch_cells_to_latlng`] on its output, but callers that need both the
/// cell index and its center coordinates (rather than just one or the
/// other) avoid a second full traversal of the data and a second rayon
/// dispatch by computing the center from the cell within the same per-row
/// closure. Invalid coordinates map to [`INVALID_CELL`] and NaN centers,
/// matching the two-call behavior exactly.
pub fn batch_latlng_to_h3_centered(
    lats: &[f64],
    lngs: &[f64],
    resolution: Resolution,
    valid_rows: Option<&[bool]>,
) -> (Vec<u64>, Vec<f64>, Vec<f64>) {
    let rows: Vec<(u64, f64, f64)> = lats
        .par_iter()
        .zip(lngs.par_iter())
        .enumerate()
        .map(|(idx, (&lat, &lng))| {
            if valid_rows.is_some_and(|valid| !valid[idx]) {
                return (INVALID_CELL, f64::NAN, f64::NAN);
            }
            match LatLng::new(lat, lng) {
                Ok(ll) => {
                    let cell = ll.to_cell(resolution);
                    let center = LatLng::from(cell);
                    (u64::from(cell), center.lat(), center.lng())
                }
                Err(_) => (INVALID_CELL, f64::NAN, f64::NAN),
            }
        })
        .collect();

    let mut out_cells = Vec::with_capacity(rows.len());
    let mut out_lats = Vec::with_capacity(rows.len());
    let mut out_lngs = Vec::with_capacity(rows.len());
    for (cell, lat, lng) in rows {
        out_cells.push(cell);
        out_lats.push(lat);
        out_lngs.push(lng);
    }
    (out_cells, out_lats, out_lngs)
}

/// Convert H3 cell indices to their fixed geographic centers in parallel.
/// Invalid or masked cells produce NaN coordinates.
pub fn batch_cells_to_latlng(cells: &[u64], valid_rows: Option<&[bool]>) -> (Vec<f64>, Vec<f64>) {
    let centers: Vec<(f64, f64)> = cells
        .par_iter()
        .enumerate()
        .map(|(idx, &cell)| {
            if valid_rows.is_some_and(|valid| !valid[idx]) || cell == INVALID_CELL {
                return (f64::NAN, f64::NAN);
            }
            CellIndex::try_from(cell)
                .map(LatLng::from)
                .map(|center| (center.lat(), center.lng()))
                .unwrap_or((f64::NAN, f64::NAN))
        })
        .collect();
    centers.into_iter().unzip()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_known_cell() {
        // Cross-checked against h3-py: h3.latlng_to_cell(37.769377, -122.388519, 9)
        // == '89283082e73ffff'.
        let cells = batch_latlng_to_cells(&[37.769377], &[-122.388519], Resolution::Nine, None);
        assert_eq!(cells, vec![0x89283082e73ffffu64]);
    }

    #[test]
    fn invalid_coordinates_map_to_sentinel() {
        let cells = batch_latlng_to_cells(
            &[f64::NAN, 10.0],
            &[20.0, f64::INFINITY],
            Resolution::Nine,
            None,
        );
        assert_eq!(cells, vec![INVALID_CELL, INVALID_CELL]);
    }

    #[test]
    fn batch_matches_scalar_one_at_a_time() {
        let lats = [37.769377, -33.865143, 51.507351];
        let lngs = [-122.388519, 151.209900, -0.127758];
        let batch = batch_latlng_to_cells(&lats, &lngs, Resolution::Nine, None);
        for i in 0..lats.len() {
            let single = batch_latlng_to_cells(&lats[i..=i], &lngs[i..=i], Resolution::Nine, None);
            assert_eq!(batch[i], single[0]);
        }
    }

    #[test]
    fn masked_row_maps_to_sentinel_even_with_valid_coordinates() {
        let lats = [37.769377, 51.507351];
        let lngs = [-122.388519, -0.127758];
        let valid = [true, false];
        let cells = batch_latlng_to_cells(&lats, &lngs, Resolution::Nine, Some(&valid));
        assert_eq!(cells[0], 0x89283082e73ffffu64);
        assert_eq!(cells[1], INVALID_CELL);
    }

    #[test]
    fn empty_input_returns_empty_output() {
        let cells = batch_latlng_to_cells(&[], &[], Resolution::Nine, None);
        assert!(cells.is_empty());
    }

    #[test]
    fn cell_centers_are_stable_and_invalid_cells_are_nan() {
        let cell = batch_latlng_to_cells(&[37.769377], &[-122.388519], Resolution::Nine, None)[0];
        let (lats, lngs) = batch_cells_to_latlng(&[cell, INVALID_CELL], None);
        assert!(lats[0].is_finite() && lngs[0].is_finite());
        assert!(lats[1].is_nan() && lngs[1].is_nan());
    }

    #[test]
    fn fused_encode_decode_matches_the_two_separate_calls() {
        let lats = [37.769377, 40.712776, f64::NAN];
        let lngs = [-122.388519, -74.005974, -122.0];
        let expected_cells = batch_latlng_to_cells(&lats, &lngs, Resolution::Nine, None);
        let (expected_lats, expected_lngs) = batch_cells_to_latlng(&expected_cells, None);

        let (cells, centered_lats, centered_lngs) =
            batch_latlng_to_h3_centered(&lats, &lngs, Resolution::Nine, None);

        assert_eq!(cells, expected_cells);
        for index in 0..lats.len() {
            let (a, b) = (centered_lats[index], expected_lats[index]);
            assert!(a == b || (a.is_nan() && b.is_nan()), "lat[{index}]: {a} vs {b}");
            let (a, b) = (centered_lngs[index], expected_lngs[index]);
            assert!(a == b || (a.is_nan() && b.is_nan()), "lng[{index}]: {a} vs {b}");
        }
    }

    #[test]
    fn fused_encode_decode_honors_masked_rows() {
        let (cells, lats, lngs) =
            batch_latlng_to_h3_centered(&[37.769377], &[-122.388519], Resolution::Nine, Some(&[false]));
        assert_eq!(cells, [INVALID_CELL]);
        assert!(lats[0].is_nan() && lngs[0].is_nan());
    }
}
