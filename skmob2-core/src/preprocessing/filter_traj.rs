use rayon::prelude::*;

use crate::utils::haversine::haversine_km;

#[derive(Clone, Copy)]
pub struct FilterConfig {
    pub max_speed_kmh: f64,
    pub include_loops: bool,
    pub speed_kmh: f64,
    pub max_loop: usize,
    pub ratio_max: f64,
}

impl FilterConfig {
    pub fn new(
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

    let mut i = 0usize;
    loop {
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
        return;
    }

    let kept_idx: Vec<usize> = (0..n).filter(|&k| keep[k]).collect();
    let m = kept_idx.len();

    let mut dr_dt: Vec<(f64, f64)> = Vec::with_capacity(config.max_loop);
    let mut drop_original_idx = vec![false; n];

    let mut ci = 0usize;
    while ci + 1 < m {
        let ahead = (config.max_loop).min(m - ci - 1);
        if ahead == 0 {
            ci += 1;
            continue;
        }

        let orig_i = kept_idx[ci];

        dr_dt.clear();
        dr_dt.extend((1..=ahead).map(|j| {
            let orig_j = kept_idx[ci + j];
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
            let to_drop = kept_idx[ci + 1 + imax];
            drop_original_idx[to_drop] = true;
        } else {
            ci += 1;
        }
        ci += 1;
    }

    for (idx, &should_drop) in drop_original_idx.iter().enumerate() {
        if should_drop {
            keep[idx] = false;
        }
    }
}

pub fn filter_trajectory_impl(
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
    user_indices: &[usize],
    config: &FilterConfig,
    keep: &mut Vec<bool>,
    kept_idx: &mut Vec<usize>,
    dr_dt: &mut Vec<(f64, f64)>,
    drop_local_idx: &mut Vec<bool>,
) {
    let n = user_indices.len();
    keep.clear();
    keep.resize(n, true);
    if n == 0 {
        return;
    }

    let mut i = 0usize;
    loop {
        let mut j = i + 1;
        while j < n && !keep[j] {
            j += 1;
        }
        if j >= n {
            break;
        }

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
        return;
    }

    kept_idx.clear();
    kept_idx.extend((0..n).filter(|&k| keep[k]));
    let m = kept_idx.len();

    drop_local_idx.clear();
    drop_local_idx.resize(n, false);

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
}

pub fn filter_trajectory_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps_s: &[f64],
    sorted_indices: &[usize],
    ends: &[usize],
    config: &FilterConfig,
) -> Vec<bool> {
    let n = latitudes.len();
    let mut global_mask = vec![true; n];

    let mask_addr = global_mask.as_mut_ptr() as usize;
    (0..ends.len()).into_par_iter().for_each_init(
        || {
            (
                Vec::<bool>::new(),
                Vec::<usize>::new(),
                Vec::<(f64, f64)>::with_capacity(config.max_loop),
                Vec::<bool>::new(),
            )
        },
        |(keep_buf, kept_idx_buf, dr_dt_buf, drop_local_idx_buf), i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            if start >= end {
                return;
            }

            let user_indices = &sorted_indices[start..end];
            filter_user_slice_indexed(
                latitudes,
                longitudes,
                timestamps_s,
                user_indices,
                config,
                keep_buf,
                kept_idx_buf,
                dr_dt_buf,
                drop_local_idx_buf,
            );

            let mask_ptr = mask_addr as *mut bool;
            for (local_idx, &keep) in keep_buf.iter().enumerate() {
                let absolute_original_idx = user_indices[local_idx];
                unsafe {
                    *mask_ptr.add(absolute_original_idx) = keep;
                }
            }
        },
    );

    global_mask
}
