use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::utils::{validate_coord_ends, validate_indexed_coord_ends};

type LocFreqData = (Vec<f64>, Vec<f64>, Vec<u64>, Vec<usize>, Vec<usize>);
type LocFreqValuesData = (
    Vec<f64>,
    Vec<f64>,
    Vec<f64>,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    Vec<f64>,
);

fn location_frequency_data_from_locs(per_user: Vec<Vec<(f64, f64, u64)>>) -> LocFreqData {
    let total_locs: usize = per_user.iter().map(|v| v.len()).sum();
    let mut out_lats = Vec::with_capacity(total_locs);
    let mut out_lngs = Vec::with_capacity(total_locs);
    let mut out_counts = Vec::with_capacity(total_locs);
    let mut out_user_starts = Vec::with_capacity(per_user.len());
    let mut out_user_ends = Vec::with_capacity(per_user.len());

    let mut offset = 0usize;
    for locs in per_user {
        out_user_starts.push(offset);
        for (lat, lng, count) in locs {
            out_lats.push(lat);
            out_lngs.push(lng);
            out_counts.push(count);
        }
        offset = out_lats.len();
        out_user_ends.push(offset);
    }

    (
        out_lats,
        out_lngs,
        out_counts,
        out_user_starts,
        out_user_ends,
    )
}

fn sorted_locs_from_counts(counts: FxHashMap<(u64, u64), (f64, f64, u64)>) -> Vec<(f64, f64, u64)> {
    let mut locs: Vec<(f64, f64, u64)> = counts.into_values().collect();
    locs.sort_unstable_by(|a, b| {
        b.2.cmp(&a.2)
            .then(a.0.total_cmp(&b.0))
            .then(a.1.total_cmp(&b.1))
    });
    locs
}

pub fn location_frequency_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
) -> Result<LocFreqData, String> {
    Ok(location_frequency_data_from_locs(
        location_frequency_locs_for_indexed(latitudes, longitudes, indices, ends, valid_rows)?,
    ))
}

pub fn location_frequency_indexed_with_row_validity_impl<F>(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
    is_valid_row: F,
) -> Result<LocFreqData, String>
where
    F: Fn(usize) -> bool + Sync,
{
    Ok(location_frequency_data_from_locs(
        location_frequency_locs_for_indexed_with_row_validity_impl(
            latitudes,
            longitudes,
            indices,
            ends,
            is_valid_row,
        )?,
    ))
}

fn location_frequency_from_locs(
    per_user: Vec<Vec<(f64, f64, u64)>>,
    normalize: bool,
) -> LocFreqValuesData {
    let total_locs: usize = per_user.iter().map(|v| v.len()).sum();
    let max_locs: usize = per_user.iter().map(|v| v.len()).max().unwrap_or(0);
    let mut out_lats = Vec::with_capacity(total_locs);
    let mut out_lngs = Vec::with_capacity(total_locs);
    let mut out_values = Vec::with_capacity(total_locs);
    let mut out_user_indices = Vec::with_capacity(total_locs);
    let mut out_user_starts = Vec::with_capacity(per_user.len());
    let mut out_user_ends = Vec::with_capacity(per_user.len());
    let mut rank_sums = vec![0.0f64; max_locs];
    let mut rank_counts = vec![0usize; max_locs];

    let mut offset = 0usize;
    for (user_idx, locs) in per_user.into_iter().enumerate() {
        out_user_starts.push(offset);
        let total: u64 = locs.iter().map(|loc| loc.2).sum();
        for (rank_idx, (lat, lng, count)) in locs.into_iter().enumerate() {
            let value = if normalize {
                if total == 0 {
                    0.0
                } else {
                    count as f64 / total as f64
                }
            } else {
                count as f64
            };
            out_lats.push(lat);
            out_lngs.push(lng);
            out_values.push(value);
            out_user_indices.push(user_idx);
            rank_sums[rank_idx] += value;
            rank_counts[rank_idx] += 1;
        }
        offset = out_lats.len();
        out_user_ends.push(offset);
    }

    let rank_means = rank_sums
        .into_iter()
        .zip(rank_counts)
        .filter_map(|(sum, count)| (count > 0).then_some(sum / count as f64))
        .collect();

    (
        out_lats,
        out_lngs,
        out_values,
        out_user_indices,
        out_user_starts,
        out_user_ends,
        rank_means,
    )
}

fn location_frequency_locs_for_indexed(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
) -> Result<Vec<Vec<(f64, f64, u64)>>, String> {
    match valid_rows {
        Some(valid_rows) => location_frequency_locs_for_indexed_valid_rows(
            latitudes, longitudes, indices, ends, valid_rows,
        ),
        None => {
            location_frequency_locs_for_indexed_no_validity(latitudes, longitudes, indices, ends)
        }
    }
}

fn location_frequency_locs_for_indexed_no_validity(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
) -> Result<Vec<Vec<(f64, f64, u64)>>, String> {
    validate_indexed_coord_ends(latitudes, longitudes, indices, ends)?;

    Ok((0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let mut counts: FxHashMap<(u64, u64), (f64, f64, u64)> =
                FxHashMap::with_capacity_and_hasher(end.saturating_sub(start), Default::default());
            for &idx in &indices[start..end] {
                if !latitudes[idx].is_finite() || !longitudes[idx].is_finite() {
                    continue;
                }
                let lat = latitudes[idx];
                let lng = longitudes[idx];
                let key = (lat.to_bits(), lng.to_bits());
                let entry = counts.entry(key).or_insert((lat, lng, 0));
                entry.2 += 1;
            }
            sorted_locs_from_counts(counts)
        })
        .collect())
}

fn location_frequency_locs_for_indexed_valid_rows(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
    valid_rows: &[bool],
) -> Result<Vec<Vec<(f64, f64, u64)>>, String> {
    validate_indexed_coord_ends(latitudes, longitudes, indices, ends)?;
    if valid_rows.len() != latitudes.len() {
        return Err("valid_rows and coordinates must have the same length".to_string());
    }

    Ok((0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let mut counts: FxHashMap<(u64, u64), (f64, f64, u64)> =
                FxHashMap::with_capacity_and_hasher(end.saturating_sub(start), Default::default());
            for &idx in &indices[start..end] {
                if !valid_rows[idx] || !latitudes[idx].is_finite() || !longitudes[idx].is_finite() {
                    continue;
                }
                let lat = latitudes[idx];
                let lng = longitudes[idx];
                let key = (lat.to_bits(), lng.to_bits());
                let entry = counts.entry(key).or_insert((lat, lng, 0));
                entry.2 += 1;
            }
            sorted_locs_from_counts(counts)
        })
        .collect())
}

pub fn location_frequency_locs_for_indexed_with_row_validity_impl<F>(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
    is_valid_row: F,
) -> Result<Vec<Vec<(f64, f64, u64)>>, String>
where
    F: Fn(usize) -> bool + Sync,
{
    validate_indexed_coord_ends(latitudes, longitudes, indices, ends)?;

    Ok((0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let mut counts: FxHashMap<(u64, u64), (f64, f64, u64)> =
                FxHashMap::with_capacity_and_hasher(end.saturating_sub(start), Default::default());
            for &idx in &indices[start..end] {
                if !is_valid_row(idx) || !latitudes[idx].is_finite() || !longitudes[idx].is_finite()
                {
                    continue;
                }
                let lat = latitudes[idx];
                let lng = longitudes[idx];
                let key = (lat.to_bits(), lng.to_bits());
                let entry = counts.entry(key).or_insert((lat, lng, 0));
                entry.2 += 1;
            }
            sorted_locs_from_counts(counts)
        })
        .collect())
}

fn location_frequency_locs_for_presorted(
    latitudes: &[f64],
    longitudes: &[f64],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
) -> Result<Vec<Vec<(f64, f64, u64)>>, String> {
    match valid_rows {
        Some(valid_rows) => location_frequency_locs_for_presorted_valid_rows(
            latitudes, longitudes, ends, valid_rows,
        ),
        None => location_frequency_locs_for_presorted_no_validity(latitudes, longitudes, ends),
    }
}

fn location_frequency_locs_for_presorted_no_validity(
    latitudes: &[f64],
    longitudes: &[f64],
    ends: &[usize],
) -> Result<Vec<Vec<(f64, f64, u64)>>, String> {
    validate_coord_ends(latitudes, longitudes, ends)?;

    Ok((0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let mut counts: FxHashMap<(u64, u64), (f64, f64, u64)> =
                FxHashMap::with_capacity_and_hasher(end.saturating_sub(start), Default::default());
            for idx in start..end {
                if !latitudes[idx].is_finite() || !longitudes[idx].is_finite() {
                    continue;
                }
                let lat = latitudes[idx];
                let lng = longitudes[idx];
                let key = (lat.to_bits(), lng.to_bits());
                let entry = counts.entry(key).or_insert((lat, lng, 0));
                entry.2 += 1;
            }
            sorted_locs_from_counts(counts)
        })
        .collect())
}

fn location_frequency_locs_for_presorted_valid_rows(
    latitudes: &[f64],
    longitudes: &[f64],
    ends: &[usize],
    valid_rows: &[bool],
) -> Result<Vec<Vec<(f64, f64, u64)>>, String> {
    validate_coord_ends(latitudes, longitudes, ends)?;
    if valid_rows.len() != latitudes.len() {
        return Err("valid_rows and coordinates must have the same length".to_string());
    }

    Ok((0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let mut counts: FxHashMap<(u64, u64), (f64, f64, u64)> =
                FxHashMap::with_capacity_and_hasher(end.saturating_sub(start), Default::default());
            for idx in start..end {
                if !valid_rows[idx] || !latitudes[idx].is_finite() || !longitudes[idx].is_finite() {
                    continue;
                }
                let lat = latitudes[idx];
                let lng = longitudes[idx];
                let key = (lat.to_bits(), lng.to_bits());
                let entry = counts.entry(key).or_insert((lat, lng, 0));
                entry.2 += 1;
            }
            sorted_locs_from_counts(counts)
        })
        .collect())
}

pub fn location_frequency_locs_for_presorted_with_row_validity_impl<F>(
    latitudes: &[f64],
    longitudes: &[f64],
    ends: &[usize],
    is_valid_row: F,
) -> Result<Vec<Vec<(f64, f64, u64)>>, String>
where
    F: Fn(usize) -> bool + Sync,
{
    validate_coord_ends(latitudes, longitudes, ends)?;

    Ok((0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let mut counts: FxHashMap<(u64, u64), (f64, f64, u64)> =
                FxHashMap::with_capacity_and_hasher(end.saturating_sub(start), Default::default());
            for idx in start..end {
                if !is_valid_row(idx) || !latitudes[idx].is_finite() || !longitudes[idx].is_finite()
                {
                    continue;
                }
                let lat = latitudes[idx];
                let lng = longitudes[idx];
                let key = (lat.to_bits(), lng.to_bits());
                let entry = counts.entry(key).or_insert((lat, lng, 0));
                entry.2 += 1;
            }
            sorted_locs_from_counts(counts)
        })
        .collect())
}

pub fn location_frequency_indexed_values_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
    normalize: bool,
    valid_rows: Option<&[bool]>,
) -> Result<LocFreqValuesData, String> {
    Ok(location_frequency_from_locs(
        location_frequency_locs_for_indexed(latitudes, longitudes, indices, ends, valid_rows)?,
        normalize,
    ))
}

pub fn location_frequency_indexed_values_with_row_validity_impl<F>(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
    normalize: bool,
    is_valid_row: F,
) -> Result<LocFreqValuesData, String>
where
    F: Fn(usize) -> bool + Sync,
{
    Ok(location_frequency_from_locs(
        location_frequency_locs_for_indexed_with_row_validity_impl(
            latitudes,
            longitudes,
            indices,
            ends,
            is_valid_row,
        )?,
        normalize,
    ))
}

pub fn location_frequency_presorted_values_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ends: &[usize],
    normalize: bool,
    valid_rows: Option<&[bool]>,
) -> Result<LocFreqValuesData, String> {
    Ok(location_frequency_from_locs(
        location_frequency_locs_for_presorted(latitudes, longitudes, ends, valid_rows)?,
        normalize,
    ))
}

pub fn location_frequency_presorted_values_with_row_validity_impl<F>(
    latitudes: &[f64],
    longitudes: &[f64],
    ends: &[usize],
    normalize: bool,
    is_valid_row: F,
) -> Result<LocFreqValuesData, String>
where
    F: Fn(usize) -> bool + Sync,
{
    Ok(location_frequency_from_locs(
        location_frequency_locs_for_presorted_with_row_validity_impl(
            latitudes,
            longitudes,
            ends,
            is_valid_row,
        )?,
        normalize,
    ))
}

type FrequencyRankData = (Vec<f64>, Vec<f64>, Vec<u64>, Vec<usize>);

fn frequency_rank_from_locs(per_user: Vec<Vec<(f64, f64, u64)>>) -> FrequencyRankData {
    let total_locs: usize = per_user.iter().map(|v| v.len()).sum();
    let mut out_lats = Vec::with_capacity(total_locs);
    let mut out_lngs = Vec::with_capacity(total_locs);
    let mut out_ranks = Vec::with_capacity(total_locs);
    let mut out_user_indices = Vec::with_capacity(total_locs);

    for (user_idx, locs) in per_user.into_iter().enumerate() {
        for (rank_idx, (lat, lng, _count)) in locs.into_iter().enumerate() {
            out_lats.push(lat);
            out_lngs.push(lng);
            out_ranks.push((rank_idx + 1) as u64);
            out_user_indices.push(user_idx);
        }
    }

    (out_lats, out_lngs, out_ranks, out_user_indices)
}

pub fn frequency_rank_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
) -> Result<FrequencyRankData, String> {
    Ok(frequency_rank_from_locs(
        location_frequency_locs_for_indexed(latitudes, longitudes, indices, ends, valid_rows)?,
    ))
}

pub fn frequency_rank_indexed_with_row_validity_impl<F>(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
    is_valid_row: F,
) -> Result<FrequencyRankData, String>
where
    F: Fn(usize) -> bool + Sync,
{
    Ok(frequency_rank_from_locs(
        location_frequency_locs_for_indexed_with_row_validity_impl(
            latitudes,
            longitudes,
            indices,
            ends,
            is_valid_row,
        )?,
    ))
}

pub fn frequency_rank_presorted_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
) -> Result<FrequencyRankData, String> {
    Ok(frequency_rank_from_locs(
        location_frequency_locs_for_presorted(latitudes, longitudes, ends, valid_rows)?,
    ))
}

pub fn frequency_rank_presorted_with_row_validity_impl<F>(
    latitudes: &[f64],
    longitudes: &[f64],
    ends: &[usize],
    is_valid_row: F,
) -> Result<FrequencyRankData, String>
where
    F: Fn(usize) -> bool + Sync,
{
    Ok(frequency_rank_from_locs(
        location_frequency_locs_for_presorted_with_row_validity_impl(
            latitudes,
            longitudes,
            ends,
            is_valid_row,
        )?,
    ))
}
