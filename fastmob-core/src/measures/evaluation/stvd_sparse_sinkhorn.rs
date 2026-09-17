//! Sparse log-domain Sinkhorn over a [`CandidateGraph`], replicating
//! `wass::sinkhorn_log`/`sinkhorn_log_with_convergence`'s exact weight
//! normalization, log-mass handling, and dual-update math -- but summing
//! each `f[i]`/`g[j]` update only over `i`'s/`j`'s graph-search candidates
//! (`stvd_candidate_graph.rs`) instead of all `m`/`n` potential partners,
//! and parallelized with rayon (the `wass` loops this replaces are
//! single-threaded).
//!
//! `stvd_emd_impl` dispatches here whenever the dense complete-bipartite
//! path's cost matrix would exceed its memory budget (see
//! `stvd_emd.rs::DENSE_COST_MEMORY_BUDGET_BYTES`); below that threshold the
//! dense `wass`-backed path stays the default, validated to agree with this
//! module within 1e-2 relative by the differential tests below (the plan's
//! stage 4 gate: `/mnt/raid5/gustavo/.claude/plans/clean-here-s-the-finding-cheeky-beaver.md`).

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
    use crate::measures::evaluation::stvd_emd::stvd_emd_impl;

    /// Deterministic xorshift PRNG -- same dependency-free pattern as
    /// `../collective/stvd.rs`'s differential test.
    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn range_f64(&mut self, lo: f64, hi: f64) -> f64 {
            let frac = (self.next() % 1_000_000) as f64 / 1_000_000.0;
            lo + frac * (hi - lo)
        }
    }

    fn random_distribution(rng: &mut XorShift, count: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
        let mut lats = Vec::with_capacity(count);
        let mut lngs = Vec::with_capacity(count);
        let mut ts = Vec::with_capacity(count);
        let mut ws = Vec::with_capacity(count);
        for _ in 0..count {
            lats.push(rng.range_f64(48.80, 48.90));
            lngs.push(rng.range_f64(2.25, 2.40));
            ts.push(rng.range_f64(0.0, 1439.0));
            ws.push(rng.range_f64(0.1, 1.0));
        }
        (lats, lngs, ts, ws)
    }

    /// The core "prove it's safe" gate: sparse must reproduce the dense
    /// (`wass`-backed) reference within a small relative tolerance, across
    /// many random small distributions (dense stays trivially cheap to
    /// compute here, so it's usable as ground truth) and across several
    /// `reg` values including the library's actual default (0.01).
    ///
    /// Measured empirically before picking this tolerance (see the plan's
    /// Open Risk #1 discussion): max relative difference over 100 trials
    /// per reg was ~0 at reg=0.01 (consistent with the near-degeneracy
    /// argument -- the dense computation is already effectively
    /// nearest-neighbor at this reg), ~4e-6 at reg=1/10, and ~6e-3 at
    /// reg=50 (larger reg genuinely spreads mass over more candidates, so
    /// the bounded-radius search has more to miss) -- 1e-2 gives headroom
    /// above the observed worst case without being so loose it would miss
    /// a real regression.
    #[test]
    fn matches_dense_reference_on_random_inputs_reg_sweep() {
        let mut rng = XorShift(0xC0FFEE_u64);
        let regs = [0.01_f64, 1.0, 10.0, 50.0];

        for &reg in &regs {
            for trial in 0..100 {
                let n = 1 + (rng.next() % 15) as usize;
                let m = 1 + (rng.next() % 15) as usize;
                let (lats_a, lngs_a, ts_a, ws_a) = random_distribution(&mut rng, n);
                let (lats_b, lngs_b, ts_b, ws_b) = random_distribution(&mut rng, m);

                let config = SinkhornConfig { reg, max_iter: 200, tol: None };
                let dense = stvd_emd_impl(
                    &lats_a, &lngs_a, &ts_a, &ws_a, &lats_b, &lngs_b, &ts_b, &ws_b, 10.0, 1440.0, config,
                )
                .unwrap();
                let sparse = sparse_stvd_emd_impl(
                    &lats_a, &lngs_a, &ts_a, &ws_a, &lats_b, &lngs_b, &ts_b, &ws_b, 10.0, 1440.0, config,
                    CandidateGraphConfig::default(),
                )
                .unwrap();

                let rel_diff = (dense - sparse).abs() / dense.abs().max(1.0);
                assert!(
                    rel_diff < 1e-2,
                    "reg={reg} trial={trial}: dense={dense} sparse={sparse} rel_diff={rel_diff}"
                );
            }
        }
    }

    /// Exercises the bidirectional-search fix (module docs, and
    /// `stvd_candidate_graph.rs`'s `every_point_has_at_least_one_candidate`
    /// test) specifically: a dense cluster plus one deliberately distant
    /// orphan on each side. A one-directional search could silently starve
    /// the orphan's marginal constraint; this checks the full distance
    /// still matches dense, not just that the orphan gets *some* candidate.
    #[test]
    fn matches_dense_reference_with_skewed_density() {
        let mut rng = XorShift(0xBADA55_u64);
        let (mut lats_a, mut lngs_a, mut ts_a, mut ws_a) = random_distribution(&mut rng, 10);
        let (mut lats_b, mut lngs_b, mut ts_b, mut ws_b) = random_distribution(&mut rng, 10);
        // Orphans: far from the rest of both distributions.
        lats_a.push(59.33);
        lngs_a.push(18.06);
        ts_a.push(200.0);
        ws_a.push(0.8);
        lats_b.push(35.68);
        lngs_b.push(139.65);
        ts_b.push(1000.0);
        ws_b.push(0.6);

        let config = SinkhornConfig::default();
        let dense = stvd_emd_impl(
            &lats_a, &lngs_a, &ts_a, &ws_a, &lats_b, &lngs_b, &ts_b, &ws_b, 10.0, 1440.0, config,
        )
        .unwrap();
        let sparse = sparse_stvd_emd_impl(
            &lats_a, &lngs_a, &ts_a, &ws_a, &lats_b, &lngs_b, &ts_b, &ws_b, 10.0, 1440.0, config,
            CandidateGraphConfig::default(),
        )
        .unwrap();

        let rel_diff = (dense - sparse).abs() / dense.abs().max(1.0);
        assert!(rel_diff < 1e-2, "dense={dense} sparse={sparse} rel_diff={rel_diff}");
    }

    #[test]
    fn matches_dense_reference_with_all_identical_timestamps() {
        let mut rng = XorShift(0x5EED_u64);
        let (lats_a, lngs_a, ts_a_random, ws_a) = random_distribution(&mut rng, 8);
        let (lats_b, lngs_b, ts_b_random, ws_b) = random_distribution(&mut rng, 8);
        let ts_a = vec![480.0; ts_a_random.len()];
        let ts_b = vec![480.0; ts_b_random.len()];

        let config = SinkhornConfig::default();
        let dense = stvd_emd_impl(
            &lats_a, &lngs_a, &ts_a, &ws_a, &lats_b, &lngs_b, &ts_b, &ws_b, 10.0, 1440.0, config,
        )
        .unwrap();
        let sparse = sparse_stvd_emd_impl(
            &lats_a, &lngs_a, &ts_a, &ws_a, &lats_b, &lngs_b, &ts_b, &ws_b, 10.0, 1440.0, config,
            CandidateGraphConfig::default(),
        )
        .unwrap();

        let rel_diff = (dense - sparse).abs() / dense.abs().max(1.0);
        assert!(rel_diff < 1e-2, "dense={dense} sparse={sparse} rel_diff={rel_diff}");
    }

    #[test]
    fn matches_dense_reference_with_extreme_size_skew() {
        let mut rng = XorShift(0xF00D_u64);
        let (lats_a, lngs_a, ts_a, ws_a) = random_distribution(&mut rng, 1);
        let (lats_b, lngs_b, ts_b, ws_b) = random_distribution(&mut rng, 25);

        let config = SinkhornConfig::default();
        let dense = stvd_emd_impl(
            &lats_a, &lngs_a, &ts_a, &ws_a, &lats_b, &lngs_b, &ts_b, &ws_b, 10.0, 1440.0, config,
        )
        .unwrap();
        let sparse = sparse_stvd_emd_impl(
            &lats_a, &lngs_a, &ts_a, &ws_a, &lats_b, &lngs_b, &ts_b, &ws_b, 10.0, 1440.0, config,
            CandidateGraphConfig::default(),
        )
        .unwrap();

        let rel_diff = (dense - sparse).abs() / dense.abs().max(1.0);
        assert!(rel_diff < 1e-2, "dense={dense} sparse={sparse} rel_diff={rel_diff}");
    }

    #[test]
    fn matches_dense_reference_for_isolated_single_point_distributions() {
        // Paris vs. Tokyo: forces the candidate graph's brute-force
        // nearest-cell fallback (stvd_candidate_graph.rs), not just ring
        // search, since no ring search bridges antipodal-ish distances.
        let lats_a = [48.8566];
        let lngs_a = [2.3522];
        let ts_a = [480.0];
        let ws_a = [1.0];
        let lats_b = [35.6762];
        let lngs_b = [139.6503];
        let ts_b = [900.0];
        let ws_b = [1.0];

        let config = SinkhornConfig::default();
        let dense = stvd_emd_impl(
            &lats_a, &lngs_a, &ts_a, &ws_a, &lats_b, &lngs_b, &ts_b, &ws_b, 10.0, 1440.0, config,
        )
        .unwrap();
        let sparse = sparse_stvd_emd_impl(
            &lats_a, &lngs_a, &ts_a, &ws_a, &lats_b, &lngs_b, &ts_b, &ws_b, 10.0, 1440.0, config,
            CandidateGraphConfig::default(),
        )
        .unwrap();

        assert!(dense.is_finite() && sparse.is_finite());
        let rel_diff = (dense - sparse).abs() / dense.abs().max(1.0);
        assert!(rel_diff < 1e-2, "dense={dense} sparse={sparse} rel_diff={rel_diff}");
    }

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

    /// Capstone scale validation (plan's verification section): synthetic
    /// data shaped like the real production report that motivated this
    /// rewrite -- gparis's ~126,892-row / ~78,918-row (location, 10-min-bin)
    /// distributions over only 1,808 distinct H3-8 locations, which OOM'd
    /// the dense path outright (n*m ~= 1e10 cells, ~40GB+). Run on demand
    /// (`cargo test --release -- --ignored`), not in the default suite,
    /// matching how `ae1009e`'s large-scale `mean_area_volume` validation
    /// was kept out of the fast test loop.
    #[test]
    #[ignore = "capstone scale validation, run with --release -- --ignored"]
    fn handles_real_world_gparis_scale_in_bounded_memory_and_time() {
        let mut rng = XorShift(0xC17_u64);
        let num_locations = 1_808usize;
        let mut loc_lats = Vec::with_capacity(num_locations);
        let mut loc_lngs = Vec::with_capacity(num_locations);
        for _ in 0..num_locations {
            loc_lats.push(rng.range_f64(48.80, 48.90));
            loc_lngs.push(rng.range_f64(2.25, 2.40));
        }

        let mut gen_side = |target_rows: usize| -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
            let mut lats = Vec::with_capacity(target_rows);
            let mut lngs = Vec::with_capacity(target_rows);
            let mut ts = Vec::with_capacity(target_rows);
            let mut ws = Vec::with_capacity(target_rows);
            while lats.len() < target_rows {
                let loc = (rng.next() % num_locations as u64) as usize;
                lats.push(loc_lats[loc]);
                lngs.push(loc_lngs[loc]);
                ts.push((rng.next() % 144) as f64 * 10.0);
                ws.push(rng.range_f64(0.1, 1.0));
            }
            (lats, lngs, ts, ws)
        };
        let (lats_a, lngs_a, ts_a, ws_a) = gen_side(126_892);
        let (lats_b, lngs_b, ts_b, ws_b) = gen_side(78_918);

        let n = lats_a.len();
        let m = lats_b.len();
        let dense_bytes = (n as u64) * (m as u64) * 8;
        assert!(dense_bytes > 4 * 1024 * 1024 * 1024, "test setup: should exceed the dense budget");

        let start = std::time::Instant::now();
        let graph = build_candidate_graph(
            &lats_a, &lngs_a, &ts_a, &lats_b, &lngs_b, &ts_b, 1440.0, 10.0 / (2.0 * std::f64::consts::PI),
            CandidateGraphConfig::default(),
        );
        let graph_elapsed = start.elapsed();

        // Bounded memory, directly: edge count must stay near-linear in
        // (n+m), nowhere near the n*m ~= 1e10 the dense path would need.
        assert!(
            graph.edge_count() < 50 * (n + m),
            "edge_count={} n+m={} -- candidate graph is not staying sparse",
            graph.edge_count(),
            n + m
        );

        let a: Vec<f32> = ws_a.iter().map(|&w| w as f32).collect();
        let b: Vec<f32> = ws_b.iter().map(|&w| w as f32).collect();
        let solve_start = std::time::Instant::now();
        let distance = sparse_sinkhorn_log(&graph, &a, &b, 0.01, 200);
        let solve_elapsed = solve_start.elapsed();

        eprintln!(
            "[capstone] n={n} m={m} edges={} avg_candidates_per_point={:.1} graph_build={graph_elapsed:?} solve={solve_elapsed:?} distance={distance}",
            graph.edge_count(),
            graph.edge_count() as f64 / (n + m) as f64,
        );

        assert!(distance.is_finite() && distance >= 0.0);
        assert!(
            graph_elapsed + solve_elapsed < std::time::Duration::from_secs(300),
            "took {:?}, expected well under 5 minutes",
            graph_elapsed + solve_elapsed
        );
    }
}
