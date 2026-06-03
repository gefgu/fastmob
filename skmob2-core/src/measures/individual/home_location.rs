use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::utils::{validate_coord_ranges, validate_indexed_coord_ranges};

type HomeResults = (Vec<f64>, Vec<f64>);

struct HomeInputs<'a> {
    latitudes: &'a [f64],
    longitudes: &'a [f64],
    hours: &'a [f64],
}

struct NightWindow {
    start: f64,
    end: f64,
}

fn best_location_for_indices(
    inputs: &HomeInputs<'_>,
    indices: &[usize],
    start: usize,
    end: usize,
    night: &NightWindow,
) -> (f64, f64) {
    let has_night = indices[start..end]
        .iter()
        .any(|&idx| inputs.hours[idx] >= night.start || inputs.hours[idx] < night.end);
    let mut counts: FxHashMap<(u64, u64), (f64, f64, u64)> = FxHashMap::default();

    for &idx in &indices[start..end] {
        let is_night = inputs.hours[idx] >= night.start || inputs.hours[idx] < night.end;
        if has_night && !is_night {
            continue;
        }
        let lat = inputs.latitudes[idx];
        let lng = inputs.longitudes[idx];
        let entry = counts
            .entry((lat.to_bits(), lng.to_bits()))
            .or_insert((lat, lng, 0));
        entry.2 += 1;
    }

    counts
        .into_values()
        .max_by(|left, right| {
            left.2
                .cmp(&right.2)
                .then_with(|| right.0.total_cmp(&left.0))
                .then_with(|| right.1.total_cmp(&left.1))
        })
        .map(|(lat, lng, _)| (lat, lng))
        .unwrap_or((f64::NAN, f64::NAN))
}

fn best_location_for_range(
    inputs: &HomeInputs<'_>,
    start: usize,
    end: usize,
    night: &NightWindow,
) -> (f64, f64) {
    let has_night =
        (start..end).any(|idx| inputs.hours[idx] >= night.start || inputs.hours[idx] < night.end);
    let mut counts: FxHashMap<(u64, u64), (f64, f64, u64)> = FxHashMap::default();

    for idx in start..end {
        let is_night = inputs.hours[idx] >= night.start || inputs.hours[idx] < night.end;
        if has_night && !is_night {
            continue;
        }
        let lat = inputs.latitudes[idx];
        let lng = inputs.longitudes[idx];
        let entry = counts
            .entry((lat.to_bits(), lng.to_bits()))
            .or_insert((lat, lng, 0));
        entry.2 += 1;
    }

    counts
        .into_values()
        .max_by(|left, right| {
            left.2
                .cmp(&right.2)
                .then_with(|| right.0.total_cmp(&left.0))
                .then_with(|| right.1.total_cmp(&left.1))
        })
        .map(|(lat, lng, _)| (lat, lng))
        .unwrap_or((f64::NAN, f64::NAN))
}

pub fn home_location_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    hours: &[f64],
    ranges: &[(usize, usize)],
    start_night: f64,
    end_night: f64,
) -> Result<HomeResults, String> {
    validate_coord_ranges(latitudes, longitudes, ranges)?;
    if hours.len() != latitudes.len() {
        return Err("hours, latitudes, and longitudes must have the same length".to_string());
    }

    let inputs = HomeInputs {
        latitudes,
        longitudes,
        hours,
    };
    let night = NightWindow {
        start: start_night,
        end: end_night,
    };
    let homes: Vec<(f64, f64)> = ranges
        .par_iter()
        .map(|&(start, end)| best_location_for_range(&inputs, start, end, &night))
        .collect();
    Ok(homes.into_iter().unzip())
}

pub fn home_location_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    hours: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
    start_night: f64,
    end_night: f64,
) -> Result<HomeResults, String> {
    validate_indexed_coord_ranges(latitudes, longitudes, indices, ranges)?;
    if hours.len() != latitudes.len() {
        return Err("hours, latitudes, and longitudes must have the same length".to_string());
    }

    let inputs = HomeInputs {
        latitudes,
        longitudes,
        hours,
    };
    let night = NightWindow {
        start: start_night,
        end: end_night,
    };
    let homes: Vec<(f64, f64)> = ranges
        .par_iter()
        .map(|&(start, end)| best_location_for_indices(&inputs, indices, start, end, &night))
        .collect();
    Ok(homes.into_iter().unzip())
}
