use rayon::prelude::*;
use rustc_hash::FxHashSet;

use crate::utils::{validate_coord_ends, validate_indexed_coord_ends};

type RecencyRankData = (Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>);
type RecencyRankValuesData = (Vec<f64>, Vec<f64>, Vec<u64>, Vec<usize>);

pub fn recency_rank_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
) -> Result<RecencyRankData, String> {
    validate_indexed_coord_ends(latitudes, longitudes, indices, ends)?;

    let per_user: Vec<Vec<(f64, f64)>> = (0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let mut seen: FxHashSet<(u64, u64)> =
                FxHashSet::with_capacity_and_hasher(end.saturating_sub(start), Default::default());
            let mut locs: Vec<(f64, f64)> = Vec::new();
            for &idx in indices[start..end].iter().rev() {
                if !valid_rows.is_none_or(|v| v[idx])
                    || !latitudes[idx].is_finite()
                    || !longitudes[idx].is_finite()
                {
                    continue;
                }
                let lat = latitudes[idx];
                let lng = longitudes[idx];
                let key = (lat.to_bits(), lng.to_bits());
                if seen.insert(key) {
                    locs.push((lat, lng));
                }
            }
            locs
        })
        .collect();

    let total_locs: usize = per_user.iter().map(|v| v.len()).sum();
    let mut out_lats = Vec::with_capacity(total_locs);
    let mut out_lngs = Vec::with_capacity(total_locs);
    let mut out_user_starts = Vec::with_capacity(ends.len());
    let mut out_user_ends = Vec::with_capacity(ends.len());

    let mut offset = 0usize;
    for locs in per_user {
        out_user_starts.push(offset);
        for (lat, lng) in locs {
            out_lats.push(lat);
            out_lngs.push(lng);
        }
        offset = out_lats.len();
        out_user_ends.push(offset);
    }

    Ok((out_lats, out_lngs, out_user_starts, out_user_ends))
}

fn recency_rank_locs_for_indexed(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
) -> Result<Vec<Vec<(f64, f64)>>, String> {
    validate_indexed_coord_ends(latitudes, longitudes, indices, ends)?;

    Ok((0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let mut seen: FxHashSet<(u64, u64)> =
                FxHashSet::with_capacity_and_hasher(end.saturating_sub(start), Default::default());
            let mut locs: Vec<(f64, f64)> = Vec::new();
            for &idx in indices[start..end].iter().rev() {
                if !valid_rows.is_none_or(|v| v[idx])
                    || !latitudes[idx].is_finite()
                    || !longitudes[idx].is_finite()
                {
                    continue;
                }
                let lat = latitudes[idx];
                let lng = longitudes[idx];
                let key = (lat.to_bits(), lng.to_bits());
                if seen.insert(key) {
                    locs.push((lat, lng));
                }
            }
            locs
        })
        .collect())
}

fn recency_rank_locs_for_presorted(
    latitudes: &[f64],
    longitudes: &[f64],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
) -> Result<Vec<Vec<(f64, f64)>>, String> {
    validate_coord_ends(latitudes, longitudes, ends)?;

    Ok((0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let mut seen: FxHashSet<(u64, u64)> =
                FxHashSet::with_capacity_and_hasher(end.saturating_sub(start), Default::default());
            let mut locs: Vec<(f64, f64)> = Vec::new();
            for idx in (start..end).rev() {
                if !valid_rows.is_none_or(|v| v[idx])
                    || !latitudes[idx].is_finite()
                    || !longitudes[idx].is_finite()
                {
                    continue;
                }
                let lat = latitudes[idx];
                let lng = longitudes[idx];
                let key = (lat.to_bits(), lng.to_bits());
                if seen.insert(key) {
                    locs.push((lat, lng));
                }
            }
            locs
        })
        .collect())
}

fn recency_rank_from_locs(per_user: Vec<Vec<(f64, f64)>>) -> RecencyRankValuesData {
    let total_locs: usize = per_user.iter().map(|v| v.len()).sum();
    let mut out_lats = Vec::with_capacity(total_locs);
    let mut out_lngs = Vec::with_capacity(total_locs);
    let mut out_ranks = Vec::with_capacity(total_locs);
    let mut out_user_indices = Vec::with_capacity(total_locs);

    for (user_idx, locs) in per_user.into_iter().enumerate() {
        for (rank_idx, (lat, lng)) in locs.into_iter().enumerate() {
            out_lats.push(lat);
            out_lngs.push(lng);
            out_ranks.push((rank_idx + 1) as u64);
            out_user_indices.push(user_idx);
        }
    }

    (out_lats, out_lngs, out_ranks, out_user_indices)
}

pub fn recency_rank_indexed_values_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
) -> Result<RecencyRankValuesData, String> {
    Ok(recency_rank_from_locs(recency_rank_locs_for_indexed(
        latitudes, longitudes, indices, ends, valid_rows,
    )?))
}

pub fn recency_rank_presorted_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
) -> Result<RecencyRankValuesData, String> {
    Ok(recency_rank_from_locs(recency_rank_locs_for_presorted(
        latitudes, longitudes, ends, valid_rows,
    )?))
}
