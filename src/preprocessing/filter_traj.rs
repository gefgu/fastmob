use pyo3::prelude::*;
use rayon::prelude::*;

use crate::haversine::haversine_km;

#[pyclass(from_py_object)]
#[derive(Clone, Copy)]
pub struct FilterConfig {
    #[pyo3(get, set)]
    pub max_speed_kmh: f64,
    #[pyo3(get, set)]
    pub include_loops: bool,
    #[pyo3(get, set)]
    pub speed_kmh: f64,
    #[pyo3(get, set)]
    pub max_loop: usize,
    #[pyo3(get, set)]
    pub ratio_max: f64,
}

#[pymethods]
impl FilterConfig {
    #[new]
    #[pyo3(signature = (max_speed_kmh=500.0, include_loops=false, speed_kmh=5.0, max_loop=6, ratio_max=0.25))]
    fn new(
        max_speed_kmh: f64,
        include_loops: bool,
        speed_kmh: f64,
        max_loop: usize,
        ratio_max: f64,
    ) -> Self {
        FilterConfig {
            max_speed_kmh,
            include_loops,
            speed_kmh,
            max_loop,
            ratio_max,
        }
    }
}

fn filter_user_slice_into(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    config: &FilterConfig,
    keep: &mut [bool],
) {
    let n = lats.len();
    debug_assert_eq!(lngs.len(), n);
    debug_assert_eq!(times.len(), n);
    debug_assert_eq!(keep.len(), n);

    keep.fill(true);
    if n == 0 {
        return;
    }

    // --- Pass 1: high-speed filter ---
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
            dist / dt * 3600.0 > config.max_speed_kmh
        };

        if should_remove {
            keep[j] = false;
            // Stay at i, re-check against the new next kept point
        } else {
            i = j;
        }
    }

    if !config.include_loops {
        return;
    }

    // --- Pass 2: loop detection ---
    // Rebuild the compacted index list of kept points
    let kept_idx: Vec<usize> = (0..n).filter(|&k| keep[k]).collect();
    let m = kept_idx.len();

    let mut dr_dt: Vec<(f64, f64)> = Vec::with_capacity(config.max_loop);
    let mut drop_original_idx = vec![false; n];

    let mut ci = 0usize; // index into kept_idx (current compacted position)
    while ci + 1 < m {
        let ahead = (config.max_loop).min(m - ci - 1);
        if ahead == 0 {
            ci += 1;
            continue;
        }

        let orig_i = kept_idx[ci];

        // Build DrDt for the next `ahead` kept points.
        // with_capacity avoids a realloc: the iterator size_hint is known.
        dr_dt.clear();
        dr_dt.extend((1..=ahead).map(|j| {
            let orig_j = kept_idx[ci + j];
            let dist = haversine_km(lats[orig_i], lngs[orig_i], lats[orig_j], lngs[orig_j]);
            let dt = times[orig_j] - times[orig_i];
            (dist, dt)
        }));

        // imax: index (0-based into dr_dt) of max distance
        let imax = dr_dt
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.0.total_cmp(&b.1.0))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let max_dist = dr_dt[imax].0;
        let threshold = max_dist * config.ratio_max;

        // Find the first index after imax where distance drops back below threshold.
        // Use iterator .find() to avoid allocating a Vec<usize> for the inside set.
        let imin = match (imax..dr_dt.len()).find(|&k| dr_dt[k].0 < threshold) {
            Some(k) => k,
            None => {
                ci += 1;
                continue;
            }
        };

        // Compute sum of Dr and Dt for the loop portion [0..imin]
        let (total_dr, total_dt) = dr_dt[..imin]
            .iter()
            .fold((0.0, 0.0), |acc, &(d, t)| (acc.0 + d, acc.1 + t));

        if total_dt > 0.0 && total_dr / total_dt * 3600.0 > config.speed_kmh {
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

    for (idx, &should_drop) in drop_original_idx.iter().enumerate() {
        if should_drop {
            keep[idx] = false;
        }
    }
}

// Returns a boolean mask of which points to keep, for the entire batch of trajectories.
pub(super) fn filter_trajectory_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps_s: &[f64],
    ranges: &[(usize, usize)],
    config: &FilterConfig,
) -> Vec<bool> {
    let n = latitudes.len();
    let mut mask = vec![true; n];

    let mask_addr = mask.as_mut_ptr() as usize;
    ranges.par_iter().for_each(|&(start, end)| {
        let user_len = end - start;
        // Safety: user ranges are non-overlapping contiguous slices of the
        // trajectory, so each Rayon worker writes a distinct mask region.
        let user_mask = unsafe {
            std::slice::from_raw_parts_mut((mask_addr as *mut bool).add(start), user_len)
        };
        filter_user_slice_into(
            &latitudes[start..end],
            &longitudes[start..end],
            &timestamps_s[start..end],
            config,
            user_mask,
        );
    });

    mask
}

fn filter_user_slice_indexed(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    user_indices: &[usize], // Map of chronological indices for *this user*
    config: &FilterConfig,
) -> Vec<bool> {
    let n = user_indices.len();
    if n == 0 {
        return Vec::new();
    }
    // Track keeping/dropping based on the localized user_indices offset
    let mut keep = vec![true; n];

    // --- Pass 1: high-speed filter ---
    let mut i = 0usize;
    loop {
        let mut j = i + 1;
        while j < n && !keep[j] {
            j += 1;
        }
        if j >= n {
            break;
        }

        // Indirection mapping: Lookup via the sorted index list
        let orig_i = user_indices[i];
        let orig_j = user_indices[j];

        let dt = times[orig_j] - times[orig_i];
        let dist = haversine_km(lats[orig_i], lngs[orig_i], lats[orig_j], lngs[orig_j]);

        let should_remove = if dt == 0.0 {
            true
        } else {
            dist / dt * 3600.0 > config.max_speed_kmh
        };

        if should_remove {
            keep[j] = false;
        } else {
            i = j;
        }
    }

    if !config.include_loops {
        return keep;
    }

    // --- Pass 2: loop detection ---
    // Extract local positions (offsets inside user_indices) that are still kept
    let kept_idx: Vec<usize> = (0..n).filter(|&k| keep[k]).collect();
    let m = kept_idx.len();

    let mut dr_dt: Vec<(f64, f64)> = Vec::with_capacity(config.max_loop);
    let mut drop_local_idx = vec![false; n];

    let mut ci = 0usize;
    while ci + 1 < m {
        let ahead = (config.max_loop).min(m - ci - 1);
        if ahead == 0 {
            ci += 1;
            continue;
        }

        let local_i = kept_idx[ci];
        let orig_i = user_indices[local_i];

        dr_dt.clear();
        dr_dt.extend((1..=ahead).map(|j| {
            let local_j = kept_idx[ci + j];
            let orig_j = user_indices[local_j];
            let dist = haversine_km(lats[orig_i], lngs[orig_i], lats[orig_j], lngs[orig_j]);
            let dt = times[orig_j] - times[orig_i];
            (dist, dt)
        }));

        let imax = dr_dt
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.0.total_cmp(&b.1.0))
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let max_dist = dr_dt[imax].0;
        let threshold = max_dist * config.ratio_max;

        let imin = match (imax..dr_dt.len()).find(|&k| dr_dt[k].0 < threshold) {
            Some(k) => k,
            None => {
                ci += 1;
                continue;
            }
        };

        let (total_dr, total_dt) = dr_dt[..imin]
            .iter()
            .fold((0.0, 0.0), |acc, &(d, t)| (acc.0 + d, acc.1 + t));

        if total_dt > 0.0 && total_dr / total_dt * 3600.0 > config.speed_kmh {
            let local_to_drop = kept_idx[ci + 1 + imax];
            drop_local_idx[local_to_drop] = true;
        } else {
            ci += 1;
        }
        ci += 1;
    }

    for (idx, &should_drop) in drop_local_idx.iter().enumerate() {
        if should_drop {
            keep[idx] = false;
        }
    }

    keep
}

// Global orchestration function exposing index mapping to Rayon threads
pub(super) fn filter_trajectory_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps_s: &[f64],
    sorted_indices: &[usize],
    ranges: &[(usize, usize)],
    config: &FilterConfig,
) -> Vec<bool> {
    let n = latitudes.len();
    // This mask directly references the memory order of the original unsorted arrays
    let mut global_mask = vec![true; n];

    let mask_addr = global_mask.as_mut_ptr() as usize;
    ranges.par_iter().for_each(|&(start, end)| {
        let user_indices = &sorted_indices[start..end];
        let user_mask =
            filter_user_slice_indexed(latitudes, longitudes, timestamps_s, user_indices, config);
        // Safety: each original row belongs to exactly one user range in
        // `sorted_indices`, so these writes target disjoint mask elements.
        let mask_ptr = mask_addr as *mut bool;
        for (local_idx, &keep) in user_mask.iter().enumerate() {
            let absolute_original_idx = user_indices[local_idx];
            unsafe {
                *mask_ptr.add(absolute_original_idx) = keep;
            }
        }
    });

    global_mask
}
