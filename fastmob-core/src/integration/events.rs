//! Trajectory/event spatiotemporal join: for each trajectory point, find the
//! nearest (by Haversine distance) reference event whose timestamp falls
//! within a symmetric time window -- the PyMove `join_with_events` semantics
//! (`utils.integration.join_with_events`): among events within
//! `[t - window, t + window]`, pick the spatially nearest one, not the
//! temporally nearest one.

use crate::utils::haversine::haversine_km;
use rayon::prelude::*;

/// Reference events to search, already sorted ascending by time (callers
/// sort once outside this function, since one sorted reference set serves
/// every query in the batch). `original_index[j]` maps sorted position `j`
/// back to the reference dataframe's original row, so the Python wrapper
/// can gather id/category columns after the fact without carrying them
/// through Rust.
pub struct SortedReferenceEvents<'a> {
    pub lat: &'a [f64],
    pub lng: &'a [f64],
    pub time_sorted: &'a [i64],
    pub original_index: &'a [usize],
}

/// For each query point `(query_lat[i], query_lng[i], query_time[i])`, finds
/// the nearest reference event whose time lies within `[time -
/// window_seconds, time + window_seconds]`.
///
/// Returns `(nearest_idx, dist_m)`: `nearest_idx[i]` is the reference's
/// *original* row index for query `i`'s nearest in-window event, or `-1` if
/// none qualify; `dist_m[i]` is that event's Haversine distance in metres,
/// or `f64::INFINITY` if none qualify. Ties (equal distance) keep the
/// event encountered first in sorted-time order.
pub fn nearest_event_within_window(
    query_lat: &[f64],
    query_lng: &[f64],
    query_time: &[i64],
    reference: &SortedReferenceEvents,
    window_seconds: i64,
) -> (Vec<i64>, Vec<f64>) {
    let n = query_lat.len();
    (0..n)
        .into_par_iter()
        .map(|i| {
            let t = query_time[i];
            let lo_time = t - window_seconds;
            let hi_time = t + window_seconds;
            let lo = reference.time_sorted.partition_point(|&rt| rt < lo_time);
            let hi = reference.time_sorted.partition_point(|&rt| rt <= hi_time);
            if lo >= hi {
                return (-1i64, f64::INFINITY);
            }
            let mut best_idx = -1i64;
            let mut best_dist = f64::INFINITY;
            for j in lo..hi {
                let d = haversine_km(query_lat[i], query_lng[i], reference.lat[j], reference.lng[j]) * 1000.0;
                if d < best_dist {
                    best_dist = d;
                    best_idx = reference.original_index[j] as i64;
                }
            }
            (best_idx, best_dist)
        })
        .unzip()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn picks_spatially_nearest_event_within_window() {
        // Two events at the same time (within window of the query), the
        // second closer spatially -- must pick the closer one, not the
        // first-in-time one.
        let query_lat = vec![0.0];
        let query_lng = vec![0.0];
        let query_time = vec![1000];
        let ref_lat = vec![0.0, 0.01];
        let ref_lng = vec![1.0, 0.0];
        let ref_time_sorted = vec![1000, 1000];
        let ref_original_index = vec![0, 1];
        let (idx, dist) = nearest_event_within_window(
            &query_lat,
            &query_lng,
            &query_time,
            &SortedReferenceEvents {
                lat: &ref_lat,
                lng: &ref_lng,
                time_sorted: &ref_time_sorted,
                original_index: &ref_original_index,
            },
            900,
        );
        assert_eq!(idx, vec![1]);
        assert!(dist[0] < 2000.0);
    }

    #[test]
    fn event_outside_time_window_is_not_matched() {
        let query_lat = vec![0.0];
        let query_lng = vec![0.0];
        let query_time = vec![1000];
        let ref_lat = vec![0.0];
        let ref_lng = vec![0.0];
        let ref_time_sorted = vec![5000]; // 4000s away, window is 900s
        let ref_original_index = vec![0];
        let (idx, dist) = nearest_event_within_window(
            &query_lat,
            &query_lng,
            &query_time,
            &SortedReferenceEvents {
                lat: &ref_lat,
                lng: &ref_lng,
                time_sorted: &ref_time_sorted,
                original_index: &ref_original_index,
            },
            900,
        );
        assert_eq!(idx, vec![-1]);
        assert_eq!(dist, vec![f64::INFINITY]);
    }

    #[test]
    fn window_boundary_is_inclusive() {
        let query_lat = vec![0.0];
        let query_lng = vec![0.0];
        let query_time = vec![1000];
        let ref_lat = vec![0.0];
        let ref_lng = vec![0.0];
        let ref_time_sorted = vec![1900]; // exactly 900s away
        let ref_original_index = vec![0];
        let (idx, _dist) = nearest_event_within_window(
            &query_lat,
            &query_lng,
            &query_time,
            &SortedReferenceEvents {
                lat: &ref_lat,
                lng: &ref_lng,
                time_sorted: &ref_time_sorted,
                original_index: &ref_original_index,
            },
            900,
        );
        assert_eq!(idx, vec![0]);
    }

    #[test]
    fn original_index_is_preserved_after_sort() {
        // Reference events sorted by time ascending; original_index maps
        // back to unsorted row 2, not sorted position 0.
        let query_lat = vec![0.0];
        let query_lng = vec![0.0];
        let query_time = vec![1000];
        let ref_lat = vec![0.0];
        let ref_lng = vec![0.0];
        let ref_time_sorted = vec![1000];
        let ref_original_index = vec![2];
        let (idx, _dist) = nearest_event_within_window(
            &query_lat,
            &query_lng,
            &query_time,
            &SortedReferenceEvents {
                lat: &ref_lat,
                lng: &ref_lng,
                time_sorted: &ref_time_sorted,
                original_index: &ref_original_index,
            },
            900,
        );
        assert_eq!(idx, vec![2]);
    }
}
