//! Trajectory-pair similarity/distance metrics.
//!
//! `fastmob.trajectory.trajectory_distance` compares exactly two whole
//! point-sequences (no per-user grouping/batching -- see the project plan:
//! these algorithms are inherently defined between two sequences, not
//! aggregated over many users). All four share the same
//! `Result<f64, String>` shape and reuse [`haversine_km`] as the
//! point-distance metric, matching every other fastmob measure's geographic-
//! distance convention.

use rayon::prelude::*;

use crate::utils::haversine::haversine_km;

fn validate_two_sequences(
    lats_a: &[f64],
    lngs_a: &[f64],
    lats_b: &[f64],
    lngs_b: &[f64],
) -> Result<(), String> {
    if lats_a.len() != lngs_a.len() {
        return Err("latitudes_a and longitudes_a must have the same length".to_string());
    }
    if lats_b.len() != lngs_b.len() {
        return Err("latitudes_b and longitudes_b must have the same length".to_string());
    }
    if lats_a.is_empty() || lats_b.is_empty() {
        return Err("both sequences must contain at least one point".to_string());
    }
    Ok(())
}

#[inline]
fn point_distance_km(lat_a: f64, lng_a: f64, lat_b: f64, lng_b: f64) -> f64 {
    haversine_km(lat_a, lng_a, lat_b, lng_b)
}

/// Dynamic Time Warping distance (km), classic O(n*m) DP.
pub fn dtw_distance_impl(
    lats_a: &[f64],
    lngs_a: &[f64],
    lats_b: &[f64],
    lngs_b: &[f64],
) -> Result<f64, String> {
    validate_two_sequences(lats_a, lngs_a, lats_b, lngs_b)?;
    let (n, m) = (lats_a.len(), lats_b.len());
    let mut dp = vec![vec![f64::INFINITY; m + 1]; n + 1];
    dp[0][0] = 0.0;
    for i in 1..=n {
        for j in 1..=m {
            let cost = point_distance_km(lats_a[i - 1], lngs_a[i - 1], lats_b[j - 1], lngs_b[j - 1]);
            let best_prev = dp[i - 1][j].min(dp[i][j - 1]).min(dp[i - 1][j - 1]);
            dp[i][j] = cost + best_prev;
        }
    }
    Ok(dp[n][m])
}

/// Discrete Fréchet distance (km), bottom-up DP (iterative to avoid
/// recursion-depth issues on long trajectories).
pub fn frechet_distance_impl(
    lats_a: &[f64],
    lngs_a: &[f64],
    lats_b: &[f64],
    lngs_b: &[f64],
) -> Result<f64, String> {
    validate_two_sequences(lats_a, lngs_a, lats_b, lngs_b)?;
    let (n, m) = (lats_a.len(), lats_b.len());
    let mut ca = vec![vec![0.0_f64; m]; n];
    for i in 0..n {
        for j in 0..m {
            let d = point_distance_km(lats_a[i], lngs_a[i], lats_b[j], lngs_b[j]);
            ca[i][j] = if i == 0 && j == 0 {
                d
            } else if i == 0 {
                ca[0][j - 1].max(d)
            } else if j == 0 {
                ca[i - 1][0].max(d)
            } else {
                ca[i - 1][j].min(ca[i - 1][j - 1]).min(ca[i][j - 1]).max(d)
            };
        }
    }
    Ok(ca[n - 1][m - 1])
}

/// Directed Hausdorff distance (km): for each point in `from`, the distance
/// to its nearest neighbor in `to`, then the max over `from`. Parallelized
/// over the outer (`from`) sequence -- the O(n*m) structure benefits from
/// rayon here, unlike the sequential DP metrics above.
fn directed_hausdorff_km(
    from_lats: &[f64],
    from_lngs: &[f64],
    to_lats: &[f64],
    to_lngs: &[f64],
) -> f64 {
    from_lats
        .par_iter()
        .zip(from_lngs.par_iter())
        .map(|(&lat_a, &lng_a)| {
            to_lats
                .iter()
                .zip(to_lngs.iter())
                .map(|(&lat_b, &lng_b)| point_distance_km(lat_a, lng_a, lat_b, lng_b))
                .fold(f64::INFINITY, f64::min)
        })
        .reduce(|| 0.0_f64, f64::max)
}

/// Symmetric (undirected) Hausdorff distance (km): the max of the two
/// directed distances.
pub fn hausdorff_distance_impl(
    lats_a: &[f64],
    lngs_a: &[f64],
    lats_b: &[f64],
    lngs_b: &[f64],
) -> Result<f64, String> {
    validate_two_sequences(lats_a, lngs_a, lats_b, lngs_b)?;
    let a_to_b = directed_hausdorff_km(lats_a, lngs_a, lats_b, lngs_b);
    let b_to_a = directed_hausdorff_km(lats_b, lngs_b, lats_a, lngs_a);
    Ok(a_to_b.max(b_to_a))
}

/// Longest Common Subsequence spatial similarity (Vlachos et al. 2002):
/// unlike the other three metrics, this returns a **similarity** in `[0, 1]`
/// (1.0 = identical shape), not a distance -- two points "match" when their
/// haversine distance is within `epsilon_km`.
pub fn lcss_similarity_impl(
    lats_a: &[f64],
    lngs_a: &[f64],
    lats_b: &[f64],
    lngs_b: &[f64],
    epsilon_km: f64,
) -> Result<f64, String> {
    validate_two_sequences(lats_a, lngs_a, lats_b, lngs_b)?;
    if epsilon_km < 0.0 {
        return Err("epsilon_km must be non-negative".to_string());
    }

    let (n, m) = (lats_a.len(), lats_b.len());
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for i in 1..=n {
        for j in 1..=m {
            let d = point_distance_km(lats_a[i - 1], lngs_a[i - 1], lats_b[j - 1], lngs_b[j - 1]);
            dp[i][j] = if d <= epsilon_km {
                dp[i - 1][j - 1] + 1
            } else {
                dp[i - 1][j].max(dp[i][j - 1])
            };
        }
    }

    let lcss_len = dp[n][m] as f64;
    let normalizer = n.min(m) as f64;
    Ok(if normalizer > 0.0 {
        lcss_len / normalizer
    } else {
        0.0
    })
}
