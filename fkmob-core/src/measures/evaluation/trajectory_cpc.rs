use h3o::{LatLng, Resolution};
use rayon::iter::{ParallelBridge, ParallelIterator};
use rayon::prelude::*;
use rustc_hash::FxHashMap;

type Edge = (u64, u64);
type FlowMap = FxHashMap<Edge, u64>;

fn validate_inputs(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
) -> Result<(), String> {
    if latitudes.len() != longitudes.len() {
        return Err(format!(
            "latitude and longitude arrays must have the same length, got {} and {}",
            latitudes.len(),
            longitudes.len()
        ));
    }

    let mut previous = 0usize;
    for &end in ends {
        if end < previous {
            return Err("range ends must be monotonically non-decreasing".to_string());
        }
        if end > indices.len() {
            return Err("range end must be within index array bounds".to_string());
        }
        previous = end;
    }

    for &idx in indices {
        if idx >= latitudes.len() {
            return Err("index must be within coordinate array bounds".to_string());
        }
    }

    Ok(())
}

fn ordered_ranges_from_ends(ends: &[usize]) -> Vec<(usize, usize)> {
    let mut start = 0usize;
    ends.iter()
        .map(|&end| {
            let range = (start, end);
            start = end;
            range
        })
        .collect()
}

fn coordinate_to_cell(lat: f64, lng: f64, resolution: Resolution) -> Option<u64> {
    if !lat.is_finite()
        || !lng.is_finite()
        || !(-90.0..=90.0).contains(&lat)
        || !(-180.0..=180.0).contains(&lng)
    {
        return None;
    }

    LatLng::new(lat, lng)
        .ok()
        .map(|coord| u64::from(coord.to_cell(resolution)))
}

fn trajectory_od_flows(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
    resolution: Resolution,
) -> Result<(FlowMap, u64), String> {
    validate_inputs(latitudes, longitudes, indices, ends)?;
    let ranges = ordered_ranges_from_ends(ends);

    let partials: Vec<(FlowMap, u64)> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let mut flows = FlowMap::default();
            let mut total = 0u64;
            let mut previous_cell = None;

            for &row_idx in &indices[start..end] {
                let Some(cell) =
                    coordinate_to_cell(latitudes[row_idx], longitudes[row_idx], resolution)
                else {
                    continue;
                };

                if let Some(origin) = previous_cell
                    && origin != cell
                {
                    *flows.entry((origin, cell)).or_insert(0) += 1;
                    total += 1;
                }
                previous_cell = Some(cell);
            }

            (flows, total)
        })
        .collect();

    let mut flows = FlowMap::default();
    let mut total = 0u64;
    for (partial_flows, partial_total) in partials {
        total += partial_total;
        for (edge, count) in partial_flows {
            *flows.entry(edge).or_insert(0) += count;
        }
    }

    Ok((flows, total))
}

#[allow(clippy::too_many_arguments)]
pub fn trajectory_common_part_of_commuters_impl(
    latitudes_a: &[f64],
    longitudes_a: &[f64],
    indices_a: &[usize],
    ends_a: &[usize],
    latitudes_b: &[f64],
    longitudes_b: &[f64],
    indices_b: &[usize],
    ends_b: &[usize],
    resolution: u8,
) -> Result<f64, String> {
    let resolution = Resolution::try_from(resolution)
        .map_err(|_| format!("H3 resolution must be between 0 and 15, got {resolution}"))?;
    let (flows_a, total_a) =
        trajectory_od_flows(latitudes_a, longitudes_a, indices_a, ends_a, resolution)?;
    let (flows_b, total_b) =
        trajectory_od_flows(latitudes_b, longitudes_b, indices_b, ends_b, resolution)?;

    let total = total_a + total_b;
    if total == 0 {
        return Ok(0.0);
    }

    let (smaller, larger) = if flows_a.len() <= flows_b.len() {
        (&flows_a, &flows_b)
    } else {
        (&flows_b, &flows_a)
    };
    let common: u64 = smaller
        .iter()
        .par_bridge()
        .map(|(edge, &count)| count.min(*larger.get(edge).unwrap_or(&0)))
        .sum();

    Ok(2.0 * common as f64 / total as f64)
}
