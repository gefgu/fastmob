//! Sparse log-domain Sinkhorn over a [`CandidateGraph`], replicating
//! `wass::sinkhorn_log`/`sinkhorn_log_with_convergence`'s exact weight
//! normalization, log-mass handling, and dual-update math -- but summing
//! each `f[i]`/`g[j]` update only over `i`'s/`j`'s graph-search candidates
//! (`stvd_candidate_graph.rs`) instead of all `m`/`n` potential partners,
//! and parallelized with rayon (the `wass` loops this replaces are
//! single-threaded).
//!
//! Not yet wired into `stvd_emd_impl`'s production path -- see the plan
//! (`/mnt/raid5/gustavo/.claude/plans/clean-here-s-the-finding-cheeky-beaver.md`,
//! stage 4): this lands as new, independently-tested code first, and only
//! becomes the default path once differential-tested against the dense
//! `wass`-backed implementation.

use rayon::prelude::*;

use super::stvd_candidate_graph::{build_candidate_graph, CandidateGraph, CandidateGraphConfig};
use super::stvd_emd::SinkhornConfig;

/// Matches `wass::EPSILON` exactly, for weight-normalization parity.
const EPSILON: f32 = 1e-7;

/// Log-domain-stable `logsumexp` over one point's candidate edges, reading
/// the *other* side's current dual values. Identical structure to `wass`'s
/// private `logsumexp_by` (max-subtract stabilization, `-inf`/NaN-safe
/// early return when every term is `-inf`), just iterating a candidate
/// slice instead of `0..len`.
fn logsumexp_over_edges(other_dual: &[f32], edges: &[(u32, f32)], reg: f32) -> f32 {
    if edges.is_empty() {
        return f32::NEG_INFINITY;
    }
    let mut max_val = f32::NEG_INFINITY;
    for &(k, cost) in edges {
        let value = (other_dual[k as usize] - cost) / reg;
        if value > max_val {
            max_val = value;
        }
    }
    if !max_val.is_finite() {
        // Every term is -inf (all reachable candidates are zero-mass) or NaN: propagate.
        return max_val;
    }
    let mut sum_exp = 0.0f32;
    for &(k, cost) in edges {
        let value = (other_dual[k as usize] - cost) / reg;
        sum_exp += (value - max_val).exp();
    }
    max_val + sum_exp.ln()
}

fn normalize_weights(ws: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let sum: f32 = ws.iter().sum();
    let normalized: Vec<f32> = ws.iter().map(|&w| w / (sum + EPSILON)).collect();
    let log_mass: Vec<f32> = normalized
        .iter()
        .map(|&x| if x <= 0.0 { f32::NEG_INFINITY } else { x.ln() })
        .collect();
    (normalized, log_mass)
}

/// One dual-update pass (used for both the `f` update, reading `g`, and the
/// `g` update, reading `f` -- the two are structurally identical, only
/// which CSR view and which log-mass vector differ).
fn update_duals(
    other_dual: &[f32],
    own_offsets: &[u32],
    own_edges: &[(u32, f32)],
    own_log_mass: &[f32],
    reg: f32,
) -> Vec<f32> {
    let own_count = own_offsets.len() - 1;
    (0..own_count)
        .into_par_iter()
        .map(|own_idx| {
            let start = own_offsets[own_idx] as usize;
            let end = own_offsets[own_idx + 1] as usize;
            let lse = logsumexp_over_edges(other_dual, &own_edges[start..end], reg);
            reg * (own_log_mass[own_idx] - lse)
        })
        .collect()
}

/// Max marginal-constraint error on one side (`Σ_candidates P - own_mass`),
/// used only by the convergence-checking variant. Mirrors
/// `sinkhorn_log_with_convergence`'s per-row/column error computation.
fn max_marginal_error(
    own_dual: &[f32],
    other_dual: &[f32],
    own_offsets: &[u32],
    own_edges: &[(u32, f32)],
    own_mass: &[f32],
    reg: f32,
) -> f32 {
    let own_count = own_offsets.len() - 1;
    (0..own_count)
        .into_par_iter()
        .map(|own_idx| {
            let start = own_offsets[own_idx] as usize;
            let end = own_offsets[own_idx + 1] as usize;
            let lse = logsumexp_over_edges(other_dual, &own_edges[start..end], reg);
            let row_sum = (own_dual[own_idx] / reg).exp() * lse.exp();
            (row_sum - own_mass[own_idx]).abs()
        })
        .reduce(|| 0.0f32, f32::max)
}

fn compute_distance(graph: &CandidateGraph, f: &[f32], g: &[f32], reg: f32) -> f32 {
    let n = graph.offsets_a.len() - 1;
    (0..n)
        .into_par_iter()
        .map(|i| {
            let start = graph.offsets_a[i] as usize;
            let end = graph.offsets_a[i + 1] as usize;
            graph.edges_a[start..end]
                .iter()
                .map(|&(j, cost)| {
                    let log_p = (f[i] + g[j as usize] - cost) / reg;
                    log_p.exp() * cost
                })
                .sum::<f32>()
        })
        .sum()
}

/// Fixed-iteration sparse Sinkhorn, mirroring `wass::sinkhorn_log` (no
/// convergence check, always runs exactly `max_iter` rounds).
pub fn sparse_sinkhorn_log(graph: &CandidateGraph, ws_a: &[f32], ws_b: &[f32], reg: f32, max_iter: usize) -> f32 {
    let (_, log_a) = normalize_weights(ws_a);
    let (_, log_b) = normalize_weights(ws_b);

    let mut f = vec![0.0f32; ws_a.len()];
    let mut g = vec![0.0f32; ws_b.len()];

    for _ in 0..max_iter {
        f = update_duals(&g, &graph.offsets_a, &graph.edges_a, &log_a, reg);
        g = update_duals(&f, &graph.offsets_b, &graph.edges_b, &log_b, reg);
    }

    compute_distance(graph, &f, &g, reg)
}

/// Sparse Sinkhorn with a marginal-error convergence check, mirroring
/// `wass::sinkhorn_log_with_convergence` exactly: checks every 10
/// iterations (and on the last), returns `Ok((distance, iterations))` once
/// the max marginal error drops below `tol`, or the same
/// `"Sinkhorn did not converge in {max_iter} iterations"` error `wass`
/// would raise if it never does.
pub fn sparse_sinkhorn_log_with_convergence(
    graph: &CandidateGraph,
    ws_a: &[f32],
    ws_b: &[f32],
    reg: f32,
    max_iter: usize,
    tol: f32,
) -> Result<(f32, usize), String> {
    let (a, log_a) = normalize_weights(ws_a);
    let (b, log_b) = normalize_weights(ws_b);

    let mut f = vec![0.0f32; ws_a.len()];
    let mut g = vec![0.0f32; ws_b.len()];

    const CHECK_EVERY: usize = 10;
    for iter in 0..max_iter {
        f = update_duals(&g, &graph.offsets_a, &graph.edges_a, &log_a, reg);
        g = update_duals(&f, &graph.offsets_b, &graph.edges_b, &log_b, reg);

        if (iter + 1) % CHECK_EVERY == 0 || iter + 1 == max_iter {
            let err_a = max_marginal_error(&f, &g, &graph.offsets_a, &graph.edges_a, &a, reg);
            let err_b = max_marginal_error(&g, &f, &graph.offsets_b, &graph.edges_b, &b, reg);
            if err_a.max(err_b) < tol {
                return Ok((compute_distance(graph, &f, &g, reg), iter + 1));
            }
        }
    }
    Err(format!("Sinkhorn did not converge in {max_iter} iterations"))
}

/// End-to-end sparse `stvd_emd`: builds the candidate graph, then solves
/// over it. Same input shape as `stvd_emd::stvd_emd_impl` (minus the dense
/// path's own validation, which callers -- currently only stage-4
/// differential tests -- are expected to have already run) so the two are
/// directly comparable.
#[allow(clippy::too_many_arguments)]
pub fn sparse_stvd_emd_impl(
    lats_a: &[f64],
    lngs_a: &[f64],
    ts_a: &[f64],
    ws_a: &[f64],
    lats_b: &[f64],
    lngs_b: &[f64],
    ts_b: &[f64],
    ws_b: &[f64],
    alpha: f64,
    cyclical_period: f64,
    sinkhorn: SinkhornConfig,
    candidate_config: CandidateGraphConfig,
) -> Result<f64, String> {
    use std::f64::consts::PI;
    let r = (alpha * cyclical_period) / (2.0 * PI);

    let graph = build_candidate_graph(
        lats_a, lngs_a, ts_a, lats_b, lngs_b, ts_b, cyclical_period, r, candidate_config,
    );

    let a: Vec<f32> = ws_a.iter().map(|&w| w as f32).collect();
    let b: Vec<f32> = ws_b.iter().map(|&w| w as f32).collect();
    let reg = sinkhorn.reg as f32;

    let distance = if let Some(tol) = sinkhorn.tol {
        let (distance, _iterations) =
            sparse_sinkhorn_log_with_convergence(&graph, &a, &b, reg, sinkhorn.max_iter, tol as f32)?;
        distance
    } else {
        sparse_sinkhorn_log(&graph, &a, &b, reg, sinkhorn.max_iter)
    };
    Ok(distance as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_single_point_is_zero_distance() {
        let value = sparse_stvd_emd_impl(
            &[10.0], &[20.0], &[480.0], &[1.0],
            &[10.0], &[20.0], &[480.0], &[1.0],
            10.0, 1440.0, SinkhornConfig::default(), CandidateGraphConfig::default(),
        )
        .unwrap();
        assert!(value.abs() < 1e-6, "got {value}");
    }

    #[test]
    fn spatial_displacement_is_positive_and_roughly_haversine() {
        use crate::utils::haversine::haversine_km;
        let lat1 = 0.0;
        let lng1 = 0.0;
        let lat2 = 0.0;
        let lng2 = 0.01; // ~1.1 km, well within default candidate search radius
        let expected_m = haversine_km(lat1, lng1, lat2, lng2) * 1000.0;

        let value = sparse_stvd_emd_impl(
            &[lat1], &[lng1], &[480.0], &[1.0],
            &[lat2], &[lng2], &[480.0], &[1.0],
            10.0, 1440.0, SinkhornConfig::default(), CandidateGraphConfig::default(),
        )
        .unwrap();
        assert!(
            (value - expected_m).abs() / expected_m < 0.05,
            "got {value}, expected close to {expected_m}"
        );
    }

    #[test]
    fn convergence_check_reports_iterations_and_matches_fixed_iteration_result() {
        let lats_a = [0.0, 0.0, 0.1];
        let lngs_a = [0.0, 0.002, 0.05];
        let ts_a = [480.0, 560.0, 720.0];
        let ws_a = [0.3, 0.7, 0.4];
        let lats_b = [0.0, 0.0, 0.12];
        let lngs_b = [0.001, 0.003, 0.06];
        let ts_b = [500.0, 600.0, 780.0];
        let ws_b = [0.5, 0.5, 0.4];

        let graph = build_candidate_graph(
            &lats_a, &lngs_a, &ts_a, &lats_b, &lngs_b, &ts_b, 1440.0, 10.0 / (2.0 * std::f64::consts::PI),
            CandidateGraphConfig::default(),
        );
        let a: Vec<f32> = ws_a.iter().map(|&w| w as f32).collect();
        let b: Vec<f32> = ws_b.iter().map(|&w| w as f32).collect();

        let fixed = sparse_sinkhorn_log(&graph, &a, &b, 10.0, 200);
        let (converged, iterations) =
            sparse_sinkhorn_log_with_convergence(&graph, &a, &b, 10.0, 200, 1e-3).unwrap();
        assert!(iterations <= 200);
        assert!(
            (fixed - converged).abs() / fixed.abs().max(1.0) < 1e-2,
            "fixed={fixed} converged={converged}"
        );
    }

    #[test]
    fn nonconvergence_raises_matching_wass_error_format() {
        // A single source/sink pair converges trivially in one iteration
        // (there's only one possible transport plan), so this needs several
        // points and a larger reg -- same reasoning as the dense-path
        // equivalent in stvd_emd.rs's Python-level test suite: at the
        // default reg=0.01 small problems already saturate in one round.
        let err = sparse_stvd_emd_impl(
            &[0.0, 0.0, 0.1],
            &[0.0, 0.002, 0.05],
            &[480.0, 560.0, 720.0],
            &[0.3, 0.7, 0.4],
            &[0.0, 0.0, 0.12],
            &[0.001, 0.003, 0.06],
            &[500.0, 600.0, 780.0],
            &[0.5, 0.5, 0.4],
            10.0,
            1440.0,
            SinkhornConfig { reg: 10.0, max_iter: 1, tol: Some(1e-12) },
            CandidateGraphConfig::default(),
        )
        .unwrap_err();
        assert_eq!(err, "Sinkhorn did not converge in 1 iterations");
    }
}
