use std::collections::HashMap;

use geo::{Distance, Haversine, Point};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use rayon::prelude::*;

use crate::utils::{
    arrow_values, as_f64_array, f64_results_into_arrow, ranges_from_starts_ends,
    validate_indexed_coord_ranges,
};

type LocationKey = (u64, u64);
type LocationStats = (f64, f64, u64, f64, usize);

#[pyfunction]
pub(crate) fn k_radius_of_gyration_km(
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

    let mut pairs: Vec<((f64, f64), u64)> = coords.into_iter().zip(visit_counts).collect();
    pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    let top_k = &pairs[..k.min(pairs.len())];

    let total_weight: f64 = top_k.iter().map(|(_, w)| *w as f64).sum();
    if total_weight == 0.0 {
        return Ok(0.0);
    }

    let (lat_sum, lng_sum) = top_k
        .iter()
        .fold((0.0f64, 0.0f64), |(ls, ns), &((lat, lng), w)| {
            (ls + lat * w as f64, ns + lng * w as f64)
        });
    let cm_lat = lat_sum / total_weight;
    let cm_lng = lng_sum / total_weight;
    let cm = Point::new(cm_lng, cm_lat);

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

fn k_radius_for_weighted_locations(coords: &[(f64, f64)], visit_counts: &[u64], k: usize) -> f64 {
    if coords.is_empty() || k == 0 {
        return 0.0;
    }

    let top_k_len = k.min(coords.len());
    let total_weight: f64 = visit_counts[..top_k_len].iter().map(|w| *w as f64).sum();
    if total_weight == 0.0 {
        return 0.0;
    }

    let (lat_sum, lng_sum) = coords[..top_k_len]
        .iter()
        .zip(&visit_counts[..top_k_len])
        .fold((0.0f64, 0.0f64), |(ls, ns), (&(lat, lng), &w)| {
            (ls + lat * w as f64, ns + lng * w as f64)
        });
    let cm_lat = lat_sum / total_weight;
    let cm_lng = lng_sum / total_weight;
    let cm = Point::new(cm_lng, cm_lat);

    let sum_sq: f64 = coords[..top_k_len]
        .iter()
        .zip(&visit_counts[..top_k_len])
        .map(|(&(lat, lng), &w)| {
            let p = Point::new(lng, lat);
            let d = Haversine.distance(p, cm) / 1000.0;
            w as f64 * d * d
        })
        .sum();

    (sum_sq / total_weight).sqrt()
}

fn k_radius_of_gyration_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
    k: usize,
) -> PyResult<Vec<f64>> {
    validate_indexed_coord_ranges(latitudes, longitudes, indices, ranges)?;
    if timestamps.len() != latitudes.len() {
        return Err(PyValueError::new_err(
            "timestamps, latitudes, and longitudes must have the same length",
        ));
    }

    Ok(ranges
        .par_iter()
        .map(|&(start, end)| {
            let mut stats: HashMap<LocationKey, LocationStats> = HashMap::new();
            for &idx in indices.iter().take(end).skip(start) {
                let lat = latitudes[idx];
                let lng = longitudes[idx];
                let entry = stats.entry((lat.to_bits(), lng.to_bits())).or_insert((
                    lat,
                    lng,
                    0,
                    timestamps[idx],
                    idx,
                ));
                entry.2 += 1;
                if timestamps[idx].total_cmp(&entry.3).is_lt()
                    || (timestamps[idx].total_cmp(&entry.3).is_eq() && idx < entry.4)
                {
                    entry.3 = timestamps[idx];
                    entry.4 = idx;
                }
            }

            let mut locations: Vec<LocationStats> = stats.into_values().collect();
            locations.sort_by(|left, right| {
                right
                    .2
                    .cmp(&left.2)
                    .then(left.3.total_cmp(&right.3))
                    .then(left.4.cmp(&right.4))
            });

            let top_len = k.min(locations.len());
            let coords: Vec<(f64, f64)> = locations[..top_len]
                .iter()
                .map(|&(lat, lng, _, _, _)| (lat, lng))
                .collect();
            let counts: Vec<u64> = locations[..top_len]
                .iter()
                .map(|&(_, _, count, _, _)| count)
                .collect();
            k_radius_for_weighted_locations(&coords, &counts, k)
        })
        .collect())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn k_radius_of_gyration_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    timestamps: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    starts: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    k: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    Ok(k_radius_of_gyration_indexed_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        timestamps.as_slice()?,
        indices.as_slice()?,
        &ranges,
        k,
    )?
    .into_pyarray(py))
}

#[pyfunction]
pub(crate) fn k_radius_of_gyration_indexed_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    timestamps: PyArray,
    indices: PyReadonlyArray1<usize>,
    starts: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
    k: usize,
) -> PyResult<PyArray> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps = as_f64_array(timestamps, "timestamps")?;
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    Ok(f64_results_into_arrow(k_radius_of_gyration_indexed_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        arrow_values(&timestamps),
        indices.as_slice()?,
        &ranges,
        k,
    )?))
}
