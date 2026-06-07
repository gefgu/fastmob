use geo::{Distance, Haversine, Point};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::utils::{validate_coord_ends, validate_coord_ranges, validate_indexed_coord_ends};

type LocationKey = (u64, u64);
type LocationStats = (f64, f64, u64, f64, usize);

pub fn k_radius_of_gyration_km(
    coords: Vec<(f64, f64)>,
    visit_counts: Vec<u64>,
    k: usize,
) -> Result<f64, String> {
    if coords.len() != visit_counts.len() {
        return Err("coords and visit_counts must have the same length".to_string());
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

pub fn k_radius_of_gyration_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps: &[f64],
    indices: &[usize],
    ends: &[usize],
    k: usize,
    valid_rows: Option<&[bool]>,
) -> Result<Vec<f64>, String> {
    validate_indexed_coord_ends(latitudes, longitudes, indices, ends)?;
    if timestamps.len() != latitudes.len() {
        return Err("timestamps, latitudes, and longitudes must have the same length".to_string());
    }

    Ok((0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let mut stats: FxHashMap<LocationKey, LocationStats> = FxHashMap::default();
            for &idx in indices.iter().take(end).skip(start) {
                if !valid_rows.is_none_or(|v| v[idx])
                    || !latitudes[idx].is_finite()
                    || !longitudes[idx].is_finite()
                    || !timestamps[idx].is_finite()
                {
                    continue;
                }
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

pub fn k_radius_of_gyration_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps: &[f64],
    ranges: &[(usize, usize)],
    k: usize,
) -> Result<Vec<f64>, String> {
    validate_coord_ranges(latitudes, longitudes, ranges)?;
    if timestamps.len() != latitudes.len() {
        return Err("timestamps, latitudes, and longitudes must have the same length".to_string());
    }

    Ok(ranges
        .par_iter()
        .map(|&(start, end)| {
            let mut stats: FxHashMap<LocationKey, LocationStats> = FxHashMap::default();
            for idx in start..end {
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

pub fn k_radius_of_gyration_from_ends_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps: &[f64],
    ends: &[usize],
    k: usize,
) -> Result<Vec<f64>, String> {
    validate_coord_ends(latitudes, longitudes, ends)?;
    if timestamps.len() != latitudes.len() {
        return Err("timestamps, latitudes, and longitudes must have the same length".to_string());
    }

    Ok((0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let mut stats: FxHashMap<LocationKey, LocationStats> = FxHashMap::default();
            for idx in start..end {
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
