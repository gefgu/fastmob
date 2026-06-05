use rayon::prelude::*;
use rustc_hash::FxHashSet;

use crate::utils::validate_indexed_coord_ends;

type RecencyRankData = (Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>);

pub fn recency_rank_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
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
