//! Truncated power-law fitting: `y = c * (x + r0)^-beta * exp(-x / kappa)`.
//!
//! Fits the log-density of a log-spaced histogram by a deterministic
//! coarse-to-fine grid search over `(r0, beta, kappa)`, solving the optimal `c`
//! in closed form in log space for every candidate. The search is
//! parallelized across `r0` candidates with rayon.
//!
//! This is fastmob's sole production estimator for
//! `fit_values_to_truncated_powerlaw` -- it replaced an earlier scipy
//! `curve_fit`-based path. It is a grid search, not a gradient-based
//! nonlinear least squares solver, so treat it as an approximate method with
//! its own error profile rather than something that should converge on
//! scipy's answer to the last bit.

use rayon::prelude::*;

/// Fitted parameters `[c, r0, beta, kappa]` plus the histogram that produced
/// them: geometric bin centres and their densities.
#[derive(Debug, Clone, PartialEq)]
pub struct TruncatedPowerlawFit {
    pub parameters: [f64; 4],
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

/// Log-spaced histogram of positive values, as densities at geometric bin
/// centres. Empty bins are dropped.
pub fn log_density_histogram(
    values: &[f64],
    n_bins: usize,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let mut clean: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .collect();
    if clean.len() < 2 {
        return Err("at least two positive finite values are required".to_string());
    }
    clean.sort_by(|a, b| a.total_cmp(b));

    let min_x = clean[0].max(1.0e-9);
    let max_x = *clean.last().expect("non-empty after the length check");
    if max_x <= min_x {
        return Err("positive values must span a non-zero range".to_string());
    }

    let n_bins = n_bins.min(clean.len()).max(2);
    let log_min = min_x.log10();
    let log_max = max_x.log10();
    let edges: Vec<f64> = (0..=n_bins)
        .map(|index| 10f64.powf(log_min + (log_max - log_min) * index as f64 / n_bins as f64))
        .collect();

    let mut x = Vec::new();
    let mut y = Vec::new();
    for index in 0..n_bins {
        let (low, high) = (edges[index], edges[index + 1]);
        let is_last = index == n_bins - 1;
        let count = clean
            .iter()
            .filter(|&&value| {
                if is_last {
                    value >= low && value <= high
                } else {
                    value >= low && value < high
                }
            })
            .count();
        let width = high - low;
        if count == 0 || width <= 0.0 {
            continue;
        }
        x.push((low * high).sqrt());
        y.push(count as f64 / (clean.len() as f64 * width));
    }
    Ok((x, y))
}

fn geometric_grid(low: f64, high: f64, n: usize) -> Vec<f64> {
    let (log_low, log_high) = (low.ln(), high.ln());
    (0..n)
        .map(|index| (log_low + (log_high - log_low) * index as f64 / (n - 1).max(1) as f64).exp())
        .collect()
}

fn linear_grid(low: f64, high: f64, n: usize) -> Vec<f64> {
    (0..n)
        .map(|index| low + (high - low) * index as f64 / (n - 1).max(1) as f64)
        .collect()
}

/// Best `(sse, c, r0, beta, kappa)` over the cartesian product of the grids.
///
/// Parallelized across `r0` candidates: each rayon task owns its slice of the
/// `beta`/`kappa` cartesian product sequentially, then the per-`r0` bests are
/// reduced to a single global best by minimum `sse`.
fn search(
    x: &[f64],
    log_y: &[f64],
    r0_values: &[f64],
    beta_values: &[f64],
    kappa_values: &[f64],
) -> (f64, f64, f64, f64, f64) {
    let default_best = (f64::INFINITY, 1.0, 1.0, 1.5, 100.0);
    r0_values
        .par_iter()
        .map(|&r0| {
            let mut best = default_best;
            for &beta in beta_values {
                for &kappa in kappa_values {
                    let shape_log: Vec<f64> = x
                        .iter()
                        .map(|&xi| -beta * (xi + r0).ln() - xi / kappa)
                        .collect();
                    // With the shape fixed, the optimal log(c) is the mean residual.
                    let log_c = log_y
                        .iter()
                        .zip(&shape_log)
                        .map(|(ly, sl)| ly - sl)
                        .sum::<f64>()
                        / log_y.len() as f64;
                    let sse = log_y
                        .iter()
                        .zip(&shape_log)
                        .map(|(ly, sl)| (ly - (log_c + sl)).powi(2))
                        .sum::<f64>();
                    if sse < best.0 {
                        best = (sse, log_c.exp(), r0, beta, kappa);
                    }
                }
            }
            best
        })
        .reduce(|| default_best, |a, b| if a.0 <= b.0 { a } else { b })
}

/// Fit a truncated power-law to positive values via a coarse-to-fine grid.
pub fn fit_truncated_powerlaw_grid(
    values: &[f64],
    n_bins: usize,
) -> Result<TruncatedPowerlawFit, String> {
    let (x, y) = log_density_histogram(values, n_bins)?;
    if x.len() < 2 {
        return Err("not enough occupied histogram bins to fit".to_string());
    }
    let max_x = values
        .iter()
        .copied()
        .filter(|value| value.is_finite() && *value > 0.0)
        .fold(f64::NEG_INFINITY, f64::max);

    let log_y: Vec<f64> = y.iter().map(|value| value.ln()).collect();

    let (_, _, r0_coarse, beta_coarse, kappa_coarse) = search(
        &x,
        &log_y,
        &geometric_grid(0.01, max_x.max(1.0), 10),
        &linear_grid(0.2, 4.0, 16),
        &geometric_grid(1.0, (max_x * 20.0).max(10.0), 14),
    );

    let r0_low = (r0_coarse / 3.0).max(0.001);
    let r0_high = (r0_coarse * 3.0).max(r0_low * 1.01);
    let beta_low = (beta_coarse - 0.6).max(0.01);
    let beta_high = beta_coarse + 0.6;
    let kappa_low = (kappa_coarse / 3.0).max(0.1);
    let kappa_high = (kappa_coarse * 3.0).max(kappa_low * 1.01);

    let (_, c, r0, beta, kappa) = search(
        &x,
        &log_y,
        &geometric_grid(r0_low, r0_high, 12),
        &linear_grid(beta_low, beta_high, 14),
        &geometric_grid(kappa_low, kappa_high, 12),
    );

    Ok(TruncatedPowerlawFit {
        parameters: [c, r0, beta, kappa],
        x,
        y,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Samples whose density follows `x^-beta` over a wide range.
    fn power_law_samples(beta: f64, count: usize) -> Vec<f64> {
        (1..=count)
            .map(|index| {
                let u = index as f64 / (count as f64 + 1.0);
                (1.0 - u).powf(-1.0 / (beta - 1.0))
            })
            .collect()
    }

    #[test]
    fn a_power_law_sample_recovers_a_plausible_exponent() {
        let fit = fit_truncated_powerlaw_grid(&power_law_samples(2.5, 2000), 30).unwrap();
        let beta = fit.parameters[2];
        assert!(
            (0.5..4.0).contains(&beta),
            "beta {beta} outside the searched range"
        );
        assert!(fit.parameters[0] > 0.0, "c must be positive");
    }

    #[test]
    fn the_histogram_drops_empty_bins_and_stays_positive() {
        let (x, y) = log_density_histogram(&[1.0, 1.2, 1.5, 2.0, 100.0], 30).unwrap();
        assert_eq!(x.len(), y.len());
        assert!(x.iter().all(|value| *value > 0.0));
        assert!(y.iter().all(|value| *value > 0.0));
        assert!(x.len() < 30, "most log bins are empty for this input");
    }

    #[test]
    fn the_fit_is_deterministic() {
        let values = power_law_samples(2.0, 500);
        let first = fit_truncated_powerlaw_grid(&values, 30).unwrap();
        let second = fit_truncated_powerlaw_grid(&values, 30).unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn non_positive_and_non_finite_values_are_ignored() {
        let mut values = power_law_samples(2.5, 500);
        values.extend([0.0, -1.0, f64::NAN, f64::INFINITY]);
        assert!(fit_truncated_powerlaw_grid(&values, 30).is_ok());
    }

    #[test]
    fn too_few_usable_values_is_an_error() {
        assert!(fit_truncated_powerlaw_grid(&[1.0], 30).is_err());
        assert!(fit_truncated_powerlaw_grid(&[-1.0, 0.0], 30).is_err());
    }

    #[test]
    fn a_degenerate_range_is_an_error() {
        assert!(fit_truncated_powerlaw_grid(&[2.0, 2.0, 2.0], 30).is_err());
    }
}
