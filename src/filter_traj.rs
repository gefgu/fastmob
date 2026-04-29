use pyo3::prelude::*;
use rayon::prelude::*;

use crate::haversine::haversine_km;

#[allow(clippy::too_many_arguments)]
fn filter_user_slice(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    max_speed_kmh: f64,
    include_loops: bool,
    speed_kmh: f64,
    max_loop: usize,
    ratio_max: f64,
) -> Vec<bool> {
    let n = lats.len();
    if n == 0 {
        return Vec::new();
    }
    let mut keep = vec![true; n];

    // --- Pass 1: high-speed filter ---
    // Simulates the original's while loop with list deletion.
    // We use a keep-mask and track the "previous kept index" manually.
    let mut i = 0usize;
    loop {
        // Find the next kept index after i
        let mut j = i + 1;
        while j < n && !keep[j] {
            j += 1;
        }
        if j >= n {
            break;
        }

        let dt = times[j] - times[i];
        let dist = haversine_km(lats[i], lngs[i], lats[j], lngs[j]);

        let should_remove = if dt == 0.0 {
            // ZeroDivisionError: remove the later point
            true
        } else {
            dist / dt * 3600.0 > max_speed_kmh
        };

        if should_remove {
            keep[j] = false;
            // Stay at i, re-check against the new next kept point
        } else {
            i = j;
        }
    }

    if !include_loops {
        return keep;
    }

    // --- Pass 2: loop detection ---
    // Rebuild the compacted index list of kept points
    let kept_idx: Vec<usize> = (0..n).filter(|&k| keep[k]).collect();
    let m = kept_idx.len();

    let mut drop_original_idx: Vec<bool> = vec![false; n];

    let mut ci = 0usize; // index into kept_idx (current compacted position)
    while ci + 1 < m {
        let ahead = (max_loop).min(m - ci - 1);
        if ahead == 0 {
            ci += 1;
            continue;
        }

        let orig_i = kept_idx[ci];

        // Build DrDt for the next `ahead` kept points
        let dr_dt: Vec<(f64, f64)> = (1..=ahead)
            .map(|j| {
                let orig_j = kept_idx[ci + j];
                let dist = haversine_km(lats[orig_i], lngs[orig_i], lats[orig_j], lngs[orig_j]);
                let dt = times[orig_j] - times[orig_i];
                (dist, dt)
            })
            .collect();

        // imax: index (0-based into dr_dt) of max distance
        let imax = dr_dt
            .iter()
            .enumerate()
            .max_by(|a, b| a.1 .0.partial_cmp(&b.1 .0).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let max_dist = dr_dt[imax].0;
        let threshold = max_dist * ratio_max;

        // Find the first index after imax where distance drops back below threshold
        let inside: Vec<usize> = (imax..dr_dt.len())
            .filter(|&k| dr_dt[k].0 < threshold)
            .collect();

        if inside.is_empty() {
            ci += 1;
            continue;
        }

        let imin = inside[0];

        // Compute sum of Dr and Dt for the loop portion [0..imin]
        let (total_dr, total_dt) = dr_dt[..imin]
            .iter()
            .fold((0.0, 0.0), |acc, &(d, t)| (acc.0 + d, acc.1 + t));

        if total_dt > 0.0 && total_dr / total_dt * 3600.0 > speed_kmh {
            // Delete the point at kept_idx[ci + 1 + imax]
            let to_drop = kept_idx[ci + 1 + imax];
            drop_original_idx[to_drop] = true;
            // Rebuild kept_idx would be expensive; instead just set keep=false
            // and continue (we don't restart from ci, matching original's `del` at imax)
        } else {
            ci += 1;
        }
        ci += 1; // always advance after loop check (original does `i += 1` in else branch)
    }

    for (i, d) in drop_original_idx.iter().enumerate() {
        if *d {
            keep[i] = false;
        }
    }

    keep
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn filter_trajectory_batch(
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    timestamps_s: Vec<f64>,
    ranges: Vec<(usize, usize)>,
    max_speed_kmh: f64,
    include_loops: bool,
    speed_kmh: f64,
    max_loop: usize,
    ratio_max: f64,
) -> PyResult<Vec<bool>> {
    let n = latitudes.len();
    let mut mask = vec![true; n];

    let per_user: Vec<(usize, Vec<bool>)> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let user_mask = filter_user_slice(
                &latitudes[start..end],
                &longitudes[start..end],
                &timestamps_s[start..end],
                max_speed_kmh,
                include_loops,
                speed_kmh,
                max_loop,
                ratio_max,
            );
            (start, user_mask)
        })
        .collect();

    for (start, user_mask) in per_user {
        for (j, &keep) in user_mask.iter().enumerate() {
            mask[start + j] = keep;
        }
    }

    Ok(mask)
}
