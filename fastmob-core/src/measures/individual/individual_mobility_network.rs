use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::utils::{validate_coord_ends, validate_indexed_coord_ends};

type MobilityNetworkData = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<u64>, Vec<usize>);

#[derive(Clone, Copy, Eq, Hash, PartialEq)]
struct EdgeKey {
    lat_origin: u64,
    lng_origin: u64,
    lat_dest: u64,
    lng_dest: u64,
}

struct Edge {
    lat_origin: f64,
    lng_origin: f64,
    lat_dest: f64,
    lng_dest: f64,
    count: u64,
}

fn is_valid_row(
    latitudes: &[f64],
    longitudes: &[f64],
    valid_rows: Option<&[bool]>,
    idx: usize,
) -> bool {
    valid_rows.is_none_or(|v| v[idx]) && latitudes[idx].is_finite() && longitudes[idx].is_finite()
}

fn push_transition(
    latitudes: &[f64],
    longitudes: &[f64],
    self_loops: bool,
    origin_idx: usize,
    dest_idx: usize,
    edges: &mut Vec<Edge>,
    positions: &mut FxHashMap<EdgeKey, usize>,
) {
    let key = EdgeKey {
        lat_origin: latitudes[origin_idx].to_bits(),
        lng_origin: longitudes[origin_idx].to_bits(),
        lat_dest: latitudes[dest_idx].to_bits(),
        lng_dest: longitudes[dest_idx].to_bits(),
    };
    if !self_loops && key.lat_origin == key.lat_dest && key.lng_origin == key.lng_dest {
        return;
    }

    if let Some(&pos) = positions.get(&key) {
        edges[pos].count += 1;
        return;
    }

    let pos = edges.len();
    positions.insert(key, pos);
    edges.push(Edge {
        lat_origin: latitudes[origin_idx],
        lng_origin: longitudes[origin_idx],
        lat_dest: latitudes[dest_idx],
        lng_dest: longitudes[dest_idx],
        count: 1,
    });
}

fn network_for_indexed_range(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    start: usize,
    end: usize,
    self_loops: bool,
    valid_rows: Option<&[bool]>,
) -> Vec<Edge> {
    let mut edges = Vec::new();
    let mut positions: FxHashMap<EdgeKey, usize> =
        FxHashMap::with_capacity_and_hasher(end.saturating_sub(start), Default::default());
    let mut previous_valid: Option<usize> = None;

    for &idx in &indices[start..end] {
        if !is_valid_row(latitudes, longitudes, valid_rows, idx) {
            continue;
        }
        if let Some(prev_idx) = previous_valid {
            push_transition(
                latitudes,
                longitudes,
                self_loops,
                prev_idx,
                idx,
                &mut edges,
                &mut positions,
            );
        }
        previous_valid = Some(idx);
    }

    edges
}

fn network_for_presorted_range(
    latitudes: &[f64],
    longitudes: &[f64],
    start: usize,
    end: usize,
    self_loops: bool,
    valid_rows: Option<&[bool]>,
) -> Vec<Edge> {
    let mut edges = Vec::new();
    let mut positions: FxHashMap<EdgeKey, usize> =
        FxHashMap::with_capacity_and_hasher(end.saturating_sub(start), Default::default());
    let mut previous_valid: Option<usize> = None;

    for idx in start..end {
        if !is_valid_row(latitudes, longitudes, valid_rows, idx) {
            continue;
        }
        if let Some(prev_idx) = previous_valid {
            push_transition(
                latitudes,
                longitudes,
                self_loops,
                prev_idx,
                idx,
                &mut edges,
                &mut positions,
            );
        }
        previous_valid = Some(idx);
    }

    edges
}

fn flatten_user_edges(per_user: Vec<Vec<Edge>>) -> MobilityNetworkData {
    let total_edges: usize = per_user.iter().map(Vec::len).sum();
    let mut lat_origins = Vec::with_capacity(total_edges);
    let mut lng_origins = Vec::with_capacity(total_edges);
    let mut lat_dests = Vec::with_capacity(total_edges);
    let mut lng_dests = Vec::with_capacity(total_edges);
    let mut n_trips = Vec::with_capacity(total_edges);
    let mut user_indices = Vec::with_capacity(total_edges);

    for (user_idx, edges) in per_user.into_iter().enumerate() {
        for edge in edges {
            lat_origins.push(edge.lat_origin);
            lng_origins.push(edge.lng_origin);
            lat_dests.push(edge.lat_dest);
            lng_dests.push(edge.lng_dest);
            n_trips.push(edge.count);
            user_indices.push(user_idx);
        }
    }

    (
        lat_origins,
        lng_origins,
        lat_dests,
        lng_dests,
        n_trips,
        user_indices,
    )
}

pub fn individual_mobility_network_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
    self_loops: bool,
    valid_rows: Option<&[bool]>,
) -> Result<MobilityNetworkData, String> {
    validate_indexed_coord_ends(latitudes, longitudes, indices, ends)?;
    if let Some(valid_rows) = valid_rows
        && valid_rows.len() != latitudes.len()
    {
        return Err("valid_rows and coordinates must have the same length".to_string());
    }

    let per_user = (0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            network_for_indexed_range(
                latitudes, longitudes, indices, start, end, self_loops, valid_rows,
            )
        })
        .collect();

    Ok(flatten_user_edges(per_user))
}

pub fn individual_mobility_network_presorted_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ends: &[usize],
    self_loops: bool,
    valid_rows: Option<&[bool]>,
) -> Result<MobilityNetworkData, String> {
    validate_coord_ends(latitudes, longitudes, ends)?;
    if let Some(valid_rows) = valid_rows
        && valid_rows.len() != latitudes.len()
    {
        return Err("valid_rows and coordinates must have the same length".to_string());
    }

    let per_user = (0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            network_for_presorted_range(latitudes, longitudes, start, end, self_loops, valid_rows)
        })
        .collect();

    Ok(flatten_user_edges(per_user))
}
