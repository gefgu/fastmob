//! Jensen-Shannon divergence between distributions and time-bin matrices.
//!
//! Thin wrappers over the `fastmob-core` kernel so Rust callers and the Python
//! layer share one definition. See that kernel's docs for how closely this
//! tracks Python's SIMD path.

use fastmob_core::measures::evaluation::jsd;

use crate::error::FastmobRsError;

/// Jensen-Shannon divergence between two distributions, normalising first.
pub fn jensen_shannon_divergence(
    distribution1: &[f64],
    distribution2: &[f64],
) -> Result<f64, FastmobRsError> {
    jsd::jensen_shannon_divergence_impl(distribution1, distribution2).map_err(FastmobRsError::Core)
}

/// Mean per-time-bin Jensen-Shannon divergence between two
/// `[category][time_bin]` matrices.
///
/// Both matrices must already be aligned to the same category rows.
pub fn time_bin_matrix_jsd(
    matrix1: &[Vec<f64>],
    matrix2: &[Vec<f64>],
) -> Result<f64, FastmobRsError> {
    jsd::time_bin_matrix_jsd_impl(matrix1, matrix2).map_err(FastmobRsError::Core)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_distributions_have_zero_divergence() {
        assert_eq!(
            jensen_shannon_divergence(&[0.2, 0.8], &[0.2, 0.8]).unwrap(),
            0.0
        );
    }

    #[test]
    fn mismatched_lengths_surface_as_an_error() {
        assert!(jensen_shannon_divergence(&[1.0], &[1.0, 2.0]).is_err());
    }

    #[test]
    fn time_bin_matrix_skips_columns_empty_on_both_sides() {
        let a = vec![vec![1.0, 0.0], vec![0.0, 0.0]];
        let b = vec![vec![0.0, 0.0], vec![1.0, 0.0]];
        let value = time_bin_matrix_jsd(&a, &b).unwrap();
        assert!((value - 2.0f64.ln()).abs() < 1e-12, "got {value}");
    }
}
