//! Jensen-Shannon divergence.
//!
//! `jensen_shannon_divergence_impl` is the sole implementation backing
//! `fastmob.measures.evaluation.jensen_shannon_divergence` (see
//! `fastmob/measures/evaluation/metrics.py`), exposed to Python via the
//! `jensen_shannon` PyO3 binding in `fastmob-py/src/measures/evaluation/jsd.rs`.

/// Normalise a distribution to sum to 1, treating non-finite entries as zero.
///
/// An all-zero (or empty) input is returned unchanged rather than divided by
/// zero, so the divergence of two empty distributions is 0 rather than NaN.
pub fn normalize_distribution(values: &[f64]) -> Vec<f64> {
    let mut out: Vec<f64> = values
        .iter()
        .map(|value| if value.is_finite() { *value } else { 0.0 })
        .collect();
    let total: f64 = out.iter().sum();
    if total > 0.0 {
        for value in &mut out {
            *value /= total;
        }
    }
    out
}

/// Jensen-Shannon divergence between two distributions.
///
/// Inputs are normalised first, so raw counts and probabilities both work.
pub fn jensen_shannon_divergence_impl(
    distribution1: &[f64],
    distribution2: &[f64],
) -> Result<f64, String> {
    if distribution1.len() != distribution2.len() {
        return Err(format!(
            "distribution shapes must match, got {} and {}",
            distribution1.len(),
            distribution2.len()
        ));
    }

    let p = normalize_distribution(distribution1);
    let q = normalize_distribution(distribution2);
    if p.is_empty() || (p.iter().sum::<f64>() == 0.0 && q.iter().sum::<f64>() == 0.0) {
        return Ok(0.0);
    }

    let mut left = 0.0;
    let mut right = 0.0;
    for (pi, qi) in p.iter().zip(&q) {
        let m = 0.5 * (pi + qi);
        if *pi > 0.0 {
            left += pi * (pi / m).ln();
        }
        if *qi > 0.0 {
            right += qi * (qi / m).ln();
        }
    }
    Ok(0.5 * (left + right))
}

/// Mean per-column Jensen-Shannon divergence between two `[category, time_bin]`
/// matrices, given in column-major slices of equal length.
///
/// Columns where both sides are entirely zero contribute nothing and are
/// skipped rather than counted as a zero divergence, matching the Python
/// implementation: an hour nobody was active in should not dilute the mean.
pub fn time_bin_matrix_jsd_impl(matrix1: &[Vec<f64>], matrix2: &[Vec<f64>]) -> Result<f64, String> {
    if matrix1.len() != matrix2.len() {
        return Err("matrix1/matrix2 must have the same number of category rows".to_string());
    }
    let bins = matrix1.first().map_or(0, Vec::len);
    if matrix2.first().map_or(0, Vec::len) != bins {
        return Err("Number of time bins must match".to_string());
    }

    let nan_to_zero = |value: f64| if value.is_nan() { 0.0 } else { value };
    let mut values = Vec::new();
    for column in 0..bins {
        let left: Vec<f64> = matrix1.iter().map(|row| nan_to_zero(row[column])).collect();
        let right: Vec<f64> = matrix2.iter().map(|row| nan_to_zero(row[column])).collect();
        if left.iter().sum::<f64>() == 0.0 && right.iter().sum::<f64>() == 0.0 {
            continue;
        }
        values.push(jensen_shannon_divergence_impl(&left, &right)?);
    }

    if values.is_empty() {
        return Ok(0.0);
    }
    Ok(values.iter().sum::<f64>() / values.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_distributions_have_zero_divergence() {
        let p = [0.2, 0.3, 0.5];
        assert_eq!(jensen_shannon_divergence_impl(&p, &p).unwrap(), 0.0);
    }

    #[test]
    fn disjoint_distributions_reach_the_maximum() {
        let value = jensen_shannon_divergence_impl(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert!((value - 2.0f64.ln()).abs() < 1e-12, "got {value}");
    }

    #[test]
    fn raw_counts_are_normalised_before_comparison() {
        let from_counts = jensen_shannon_divergence_impl(&[2.0, 6.0], &[3.0, 1.0]).unwrap();
        let from_probs = jensen_shannon_divergence_impl(&[0.25, 0.75], &[0.75, 0.25]).unwrap();
        assert!((from_counts - from_probs).abs() < 1e-15);
    }

    #[test]
    fn non_finite_entries_are_treated_as_zero() {
        let with_nan = jensen_shannon_divergence_impl(&[1.0, f64::NAN], &[0.0, 1.0]).unwrap();
        let without = jensen_shannon_divergence_impl(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert_eq!(with_nan, without);
    }

    #[test]
    fn empty_and_all_zero_inputs_are_zero_not_nan() {
        assert_eq!(jensen_shannon_divergence_impl(&[], &[]).unwrap(), 0.0);
        assert_eq!(
            jensen_shannon_divergence_impl(&[0.0, 0.0], &[0.0, 0.0]).unwrap(),
            0.0
        );
    }

    #[test]
    fn mismatched_lengths_are_rejected() {
        assert!(jensen_shannon_divergence_impl(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn time_bin_matrix_averages_over_non_empty_columns_only() {
        // Column 1 is empty on both sides and must not dilute the mean.
        let a = vec![vec![1.0, 0.0], vec![0.0, 0.0]];
        let b = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let value = time_bin_matrix_jsd_impl(&a, &b).unwrap();
        assert!((value - 2.0f64.ln()).abs() < 1e-12, "got {value}");
    }

    #[test]
    fn time_bin_matrix_with_no_activity_is_zero() {
        let a = vec![vec![0.0, 0.0]];
        assert_eq!(time_bin_matrix_jsd_impl(&a, &a).unwrap(), 0.0);
    }

    #[test]
    fn time_bin_matrix_rejects_shape_mismatches() {
        let a = vec![vec![1.0, 0.0]];
        let b = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        assert!(time_bin_matrix_jsd_impl(&a, &b).is_err());
        let c = vec![vec![1.0]];
        assert!(time_bin_matrix_jsd_impl(&a, &c).is_err());
    }
}
