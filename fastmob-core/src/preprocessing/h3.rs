//! Batch lat/lng -> H3 cell conversion.
//!
//! `h3o`'s per-point `LatLng::to_cell` is the same algorithm h3-py's scalar
//! `latlng_to_cell` uses, but calling it from a Python `for`/list-comprehension
//! loop pays Python-call overhead per row. Trajectory datasets with tens to
//! hundreds of millions of rows need this converted in bulk, so it's run here
//! across all rows in one call, in parallel.

use h3o::{LatLng, Resolution};
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
}
