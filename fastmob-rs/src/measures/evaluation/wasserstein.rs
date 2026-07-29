//! Empirical 1-D Wasserstein distance between two samples.

use fastmob_core::measures::evaluation::wasserstein::empirical_wasserstein_1d;

/// Wasserstein distance between two samples, ignoring non-finite values.
///
/// Returns `NaN` when either sample is empty after filtering, which is how the
/// comparison layer signals "no distance to report" rather than an error.
pub fn wasserstein_distance(values1: &[f64], values2: &[f64]) -> f64 {
    let left: Vec<f64> = values1.iter().copied().filter(|v| v.is_finite()).collect();
    let right: Vec<f64> = values2.iter().copied().filter(|v| v.is_finite()).collect();
    if left.is_empty() || right.is_empty() {
        return f64::NAN;
    }
    empirical_wasserstein_1d(&left, &right).unwrap_or(f64::NAN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_samples_have_zero_distance() {
        assert_eq!(
            wasserstein_distance(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]),
            0.0
        );
    }

    #[test]
    fn a_constant_shift_equals_the_shift() {
        let value = wasserstein_distance(&[1.0, 2.0, 3.0], &[2.0, 3.0, 4.0]);
        assert!((value - 1.0).abs() < 1e-12, "got {value}");
    }

    #[test]
    fn non_finite_values_are_ignored() {
        let with_noise = wasserstein_distance(&[1.0, f64::NAN, 2.0, f64::INFINITY], &[1.0, 2.0]);
        assert_eq!(with_noise, 0.0);
    }

    #[test]
    fn an_empty_sample_yields_nan() {
        assert!(wasserstein_distance(&[], &[1.0]).is_nan());
        assert!(wasserstein_distance(&[f64::NAN], &[1.0]).is_nan());
    }
}
