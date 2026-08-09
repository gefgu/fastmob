//! Kullback-Leibler divergence for paired discrete distributions.

/// Return ``D_KL(p || q)`` after independently normalizing raw counts.
///
/// The numerical behavior intentionally follows the historical
/// ``scipy.stats.entropy(p, q)`` use: zero reference mass contributes zero,
/// positive reference mass over zero predicted mass is infinite, and invalid
/// totals/values naturally propagate as ``NaN``.
pub fn kullback_leibler_divergence_impl(p: &[f64], q: &[f64]) -> Result<f64, String> {
    if p.len() != q.len() {
        return Err(format!(
            "true and pred must have the same length, got {} and {}",
            p.len(),
            q.len()
        ));
    }
    let p_total: f64 = p.iter().sum();
    let q_total: f64 = q.iter().sum();
    let mut divergence = 0.0;
    for (&p_value, &q_value) in p.iter().zip(q) {
        let p_normalized = p_value / p_total;
        if p_normalized != 0.0 {
            let q_normalized = q_value / q_total;
            divergence += p_normalized * (p_normalized / q_normalized).ln();
        }
    }
    Ok(divergence)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_raw_counts() {
        let probabilities = kullback_leibler_divergence_impl(&[0.4, 0.6], &[0.5, 0.5]).unwrap();
        let counts = kullback_leibler_divergence_impl(&[4.0, 6.0], &[5.0, 5.0]).unwrap();
        assert!((probabilities - counts).abs() < 1e-15);
    }

    #[test]
    fn zero_reference_mass_contributes_nothing() {
        assert_eq!(
            kullback_leibler_divergence_impl(&[0.0, 2.0], &[0.0, 2.0]).unwrap(),
            0.0
        );
    }

    #[test]
    fn missing_predicted_support_is_infinite() {
        assert!(
            kullback_leibler_divergence_impl(&[1.0, 1.0], &[0.0, 1.0])
                .unwrap()
                .is_infinite()
        );
    }

    #[test]
    fn mismatched_lengths_are_rejected() {
        assert!(kullback_leibler_divergence_impl(&[1.0], &[1.0, 2.0]).is_err());
    }
}
