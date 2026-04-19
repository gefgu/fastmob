use geo::{Distance, Haversine, Point};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let p1 = Point::new(lon1, lat1);
    let p2 = Point::new(lon2, lat2);
    Haversine.distance(p1, p2) / 1000.0
}

#[pyfunction]
fn jump_lengths_km(latitudes: Vec<f64>, longitudes: Vec<f64>) -> PyResult<Vec<f64>> {
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }

    if latitudes.is_empty() {
        return Ok(Vec::new());
    }

    // Zip coordinates together into a single slice of pairs for easier windowing
    let coords: Vec<(f64, f64)> = latitudes
        .into_iter()
        .zip(longitudes.into_iter())
        .collect();

    // Use par_windows to process segments in parallel
    let lengths: Vec<f64> = coords
        .par_windows(2)
        .map(|window| {
            let (lat1, lon1) = window[0];
            let (lat2, lon2) = window[1];
            haversine_km(lat1, lon1, lat2, lon2)
        })
        .collect();

    Ok(lengths)
}

/// Compute the radius of gyration (in km) for a single user's trajectory.
///
/// Accepts a list of (lat, lng) pairs. Returns the RMS Haversine distance
/// from each point to the center of mass (mean lat/lng).
///
/// Returns 0.0 for empty or single-point trajectories.
#[pyfunction]
fn radius_of_gyration_km(coords: Vec<(f64, f64)>) -> PyResult<f64> {
    Ok(rog_for_slice(&coords))
}

/// Compute radius of gyration for a contiguous slice of coordinates.
fn rog_for_slice(coords: &[(f64, f64)]) -> f64 {
    let n = coords.len();
    if n == 0 {
        return 0.0;
    }

    // Center of mass: arithmetic mean of lat and lng
    let (lat_sum, lng_sum) = coords.iter().fold((0.0f64, 0.0f64), |(ls, ns), &(lat, lng)| {
        (ls + lat, ns + lng)
    });
    let cm_lat = lat_sum / n as f64;
    let cm_lng = lng_sum / n as f64;
    let cm = Point::new(cm_lng, cm_lat);

    // RMS distance from center of mass
    let sum_sq: f64 = coords
        .iter()
        .map(|&(lat, lng)| {
            let p = Point::new(lng, lat);
            let d = Haversine.distance(p, cm) / 1000.0;
            d * d
        })
        .sum();

    (sum_sq / n as f64).sqrt()
}

/// Batch version: compute radius of gyration for all users in one call.
///
/// Accepts the full sorted latitude and longitude arrays, plus a list of
/// (start, end) index ranges — one range per user, in any order.
/// Returns one RoG value per range in the same order.
///
/// This eliminates the per-user Python↔Rust boundary crossing overhead when
/// there are many users.
#[pyfunction]
fn radius_of_gyration_batch_km(
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }

    // Interleave into (lat, lng) pairs for cache-friendly access
    let coords: Vec<(f64, f64)> = latitudes
        .into_iter()
        .zip(longitudes.into_iter())
        .collect();

    // Process each user's range in parallel
    let results: Vec<f64> = ranges
        .par_iter()
        .map(|&(start, end)| rog_for_slice(&coords[start..end]))
        .collect();

    Ok(results)
}

/// Compute the maximum consecutive-pair Haversine distance (km) for each user range.
///
/// Accepts the full sorted latitude and longitude arrays, plus a list of
/// (start, end) index ranges — one range per user, in any order.
/// Returns one maximum-jump-length value per range in the same order.
///
/// A range with fewer than 2 points returns 0.0.
///
/// Called from skmob2/measures/spatial/maximum_distance.py.
#[pyfunction]
fn maximum_distance_batch_km(
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }

    let coords: Vec<(f64, f64)> = latitudes
        .into_iter()
        .zip(longitudes.into_iter())
        .collect();

    let results: Vec<f64> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let slice = &coords[start..end];
            if slice.len() < 2 {
                return 0.0;
            }
            slice
                .windows(2)
                .map(|w| haversine_km(w[0].0, w[0].1, w[1].0, w[1].1))
                .fold(0.0f64, f64::max)
        })
        .collect();

    Ok(results)
}

/// Compute the total trajectory length (km) — sum of consecutive Haversine distances —
/// for each user range.
///
/// Accepts the full sorted latitude and longitude arrays, plus a list of
/// (start, end) index ranges — one range per user, in any order.
/// Returns one total-distance value per range in the same order.
///
/// A range with fewer than 2 points returns 0.0.
///
/// Called from skmob2/measures/spatial/distance_straight_line.py.
#[pyfunction]
fn total_distance_batch_km(
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }

    let coords: Vec<(f64, f64)> = latitudes
        .into_iter()
        .zip(longitudes.into_iter())
        .collect();

    let results: Vec<f64> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let slice = &coords[start..end];
            if slice.len() < 2 {
                return 0.0;
            }
            slice
                .windows(2)
                .map(|w| haversine_km(w[0].0, w[0].1, w[1].0, w[1].1))
                .sum()
        })
        .collect();

    Ok(results)
}

/// Compute the k-radius of gyration (km) for a single user's top-k locations.
///
/// Accepts the unique location coordinates (lat, lng) and the corresponding
/// visit counts for a single user, together with k (the number of top-visited
/// locations to consider).  Returns the weighted RoG over the top-k locations.
///
/// The weighted RoG is defined as:
///   sqrt( sum_i(w_i * haversine(r_i, r_cm)^2) / sum_i(w_i) )
/// where the sum is over the top-k locations sorted by visit count descending,
/// and r_cm is the weighted center of mass (weighted mean lat/lng).
///
/// Returns 0.0 when coords is empty or k == 0.
///
/// Called from skmob2/measures/spatial/k_radius_of_gyration.py.
#[pyfunction]
fn k_radius_of_gyration_km(
    coords: Vec<(f64, f64)>,
    visit_counts: Vec<u64>,
    k: usize,
) -> PyResult<f64> {
    if coords.len() != visit_counts.len() {
        return Err(PyValueError::new_err(
            "coords and visit_counts must have the same length",
        ));
    }

    if coords.is_empty() || k == 0 {
        return Ok(0.0);
    }

    // Pair each location with its visit count, sort descending by count, take top-k.
    let mut pairs: Vec<((f64, f64), u64)> = coords
        .into_iter()
        .zip(visit_counts.into_iter())
        .collect();
    pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    let top_k = &pairs[..k.min(pairs.len())];

    let total_weight: f64 = top_k.iter().map(|(_, w)| *w as f64).sum();
    if total_weight == 0.0 {
        return Ok(0.0);
    }

    // Weighted center of mass (arithmetic mean weighted by visit count).
    let (lat_sum, lng_sum) = top_k.iter().fold((0.0f64, 0.0f64), |(ls, ns), &((lat, lng), w)| {
        (ls + lat * w as f64, ns + lng * w as f64)
    });
    let cm_lat = lat_sum / total_weight;
    let cm_lng = lng_sum / total_weight;
    let cm = Point::new(cm_lng, cm_lat);

    // Weighted RMS distance from center of mass.
    let sum_sq: f64 = top_k
        .iter()
        .map(|&((lat, lng), w)| {
            let p = Point::new(lng, lat);
            let d = Haversine.distance(p, cm) / 1000.0;
            w as f64 * d * d
        })
        .sum();

    Ok((sum_sq / total_weight).sqrt())
}

/// Compute the maximum Haversine distance (km) from a fixed home point to any
/// trajectory point, for each user.
///
/// Accepts a parallel list of home coordinates (one per user), the full sorted
/// latitude/longitude arrays, and per-user (start, end) ranges.  Returns one
/// max-distance value per user in the same order as ``ranges``.
///
/// A range with zero points returns 0.0.
///
/// Called from skmob2/measures/spatial/max_distance_from_home.py.
#[pyfunction]
fn max_distance_from_point_batch_km(
    home_lats: Vec<f64>,
    home_lngs: Vec<f64>,
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    if home_lats.len() != home_lngs.len() {
        return Err(PyValueError::new_err(
            "home_lats and home_lngs must have the same length",
        ));
    }
    if home_lats.len() != ranges.len() {
        return Err(PyValueError::new_err(
            "home coordinates and ranges must have the same length",
        ));
    }
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }

    let coords: Vec<(f64, f64)> = latitudes
        .into_iter()
        .zip(longitudes.into_iter())
        .collect();

    let results: Vec<f64> = ranges
        .par_iter()
        .enumerate()
        .map(|(i, &(start, end))| {
            let slice = &coords[start..end];
            if slice.is_empty() {
                return 0.0;
            }
            let home_lat = home_lats[i];
            let home_lng = home_lngs[i];
            slice
                .iter()
                .map(|&(lat, lng)| haversine_km(home_lat, home_lng, lat, lng))
                .fold(0.0f64, f64::max)
        })
        .collect();

    Ok(results)
}

/// Compute consecutive waiting times (seconds) for each user range.
///
/// Accepts a flat array of Unix timestamps in seconds (one entry per trajectory
/// row, sorted by [uid, datetime]), plus a list of (start, end) ranges — one
/// per user.  Returns a Vec of Vecs: each inner Vec contains the time differences
/// between consecutive points for one user.
///
/// A range with fewer than 2 points returns an empty inner Vec.
///
/// Called from skmob2/measures/spatial/waiting_times.py.
#[pyfunction]
fn waiting_times_seconds(
    timestamps_s: Vec<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<Vec<f64>>> {
    let results: Vec<Vec<f64>> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let slice = &timestamps_s[start..end];
            if slice.len() < 2 {
                return Vec::new();
            }
            slice.windows(2).map(|w| w[1] - w[0]).collect()
        })
        .collect();

    Ok(results)
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(haversine_km, m)?)?;
    m.add_function(wrap_pyfunction!(jump_lengths_km, m)?)?;
    m.add_function(wrap_pyfunction!(radius_of_gyration_km, m)?)?;
    m.add_function(wrap_pyfunction!(radius_of_gyration_batch_km, m)?)?;
    m.add_function(wrap_pyfunction!(maximum_distance_batch_km, m)?)?;
    m.add_function(wrap_pyfunction!(total_distance_batch_km, m)?)?;
    m.add_function(wrap_pyfunction!(k_radius_of_gyration_km, m)?)?;
    m.add_function(wrap_pyfunction!(max_distance_from_point_batch_km, m)?)?;
    m.add_function(wrap_pyfunction!(waiting_times_seconds, m)?)?;
    Ok(())
}