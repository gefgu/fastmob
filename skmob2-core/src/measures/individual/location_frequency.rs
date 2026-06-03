use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::utils::validate_indexed_coord_ranges;

type LocFreqData = (Vec<f64>, Vec<f64>, Vec<u64>, Vec<usize>, Vec<usize>);

pub fn location_frequency_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> Result<LocFreqData, String> {
    validate_indexed_coord_ranges(latitudes, longitudes, indices, ranges)?;

    let per_user: Vec<Vec<(f64, f64, u64)>> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let mut counts: FxHashMap<(u64, u64), (f64, f64, u64)> =
                FxHashMap::with_capacity_and_hasher(end.saturating_sub(start), Default::default());
            for &idx in &indices[start..end] {
                let lat = latitudes[idx];
                let lng = longitudes[idx];
                let key = (lat.to_bits(), lng.to_bits());
                let entry = counts.entry(key).or_insert((lat, lng, 0));
                entry.2 += 1;
            }
            let mut locs: Vec<(f64, f64, u64)> = counts.into_values().collect();
            locs.sort_unstable_by(|a, b| {
                b.2.cmp(&a.2)
                    .then(a.0.total_cmp(&b.0))
                    .then(a.1.total_cmp(&b.1))
            });
            locs
        })
        .collect();

    let total_locs: usize = per_user.iter().map(|v| v.len()).sum();
    let mut out_lats = Vec::with_capacity(total_locs);
    let mut out_lngs = Vec::with_capacity(total_locs);
    let mut out_counts = Vec::with_capacity(total_locs);
    let mut out_user_starts = Vec::with_capacity(ranges.len());
    let mut out_user_ends = Vec::with_capacity(ranges.len());

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

    Ok((
        out_lats,
        out_lngs,
        out_counts,
        out_user_starts,
        out_user_ends,
    ))
}
