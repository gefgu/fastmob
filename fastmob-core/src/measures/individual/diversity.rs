use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::utils::validate_ranges;

/// Builds a suffix array via the O(n log^2 n) prefix-doubling algorithm.
fn build_suffix_array(seq: &[usize]) -> Vec<usize> {
    let n = seq.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    let mut sa: Vec<usize> = (0..n).collect();
    let mut rank: Vec<i64> = seq.iter().map(|&v| v as i64).collect();
    let mut next_rank = vec![0i64; n];
    let mut k = 1usize;

    #[inline]
    fn key_at(rank: &[i64], n: usize, k: usize, i: usize) -> (i64, i64) {
        let second = if i + k < n { rank[i + k] } else { -1 };
        (rank[i], second)
    }

    while k < n {
        sa.sort_unstable_by_key(|&a| key_at(&rank, n, k, a));

        next_rank[sa[0]] = 0;
        for i in 1..n {
            let same = key_at(&rank, n, k, sa[i - 1]) == key_at(&rank, n, k, sa[i]);
            next_rank[sa[i]] = next_rank[sa[i - 1]] + if same { 0 } else { 1 };
        }
        rank.copy_from_slice(&next_rank);

        if rank[sa[n - 1]] as usize == n - 1 {
            break;
        }
        k <<= 1;
    }

    sa
}

/// Kasai's O(n) algorithm: `lcp[rank[i]]` is the longest common prefix between
/// the suffix starting at `i` and the suffix immediately before it in `sa` order.
/// `lcp[0]` stays 0 by convention (the first suffix in SA order has no predecessor).
fn kasai_lcp(seq: &[usize], sa: &[usize]) -> Vec<usize> {
    let n = seq.len();
    let mut rank = vec![0usize; n];
    for (i, &s) in sa.iter().enumerate() {
        rank[s] = i;
    }

    let mut lcp = vec![0usize; n];
    let mut h = 0usize;
    for i in 0..n {
        if rank[i] > 0 {
            let j = sa[rank[i] - 1];
            while i + h < n && j + h < n && seq[i + h] == seq[j + h] {
                h += 1;
            }
            lcp[rank[i]] = h;
            h = h.saturating_sub(1);
        } else {
            h = 0;
        }
    }

    lcp
}

/// Suffix-array entropy: ratio of distinct substrings to total substrings.
///
/// Distinct substring count follows the standard suffix-array + LCP formula:
/// `sum((n - sa[i]) - lcp[i])`, matching `pydivsufsort`'s `divsufsort`/`kasai`
/// convention (`lcp[0] == 0`).
fn suffix_array_diversity(sequence: &[usize]) -> f64 {
    let n = sequence.len();
    if n <= 1 {
        return 0.0;
    }

    let sa = build_suffix_array(sequence);
    let lcp = kasai_lcp(sequence, &sa);

    let distinct_substrings: usize = (0..n).map(|i| (n - sa[i]) - lcp[i]).sum();
    let total_substrings = n * (n + 1) / 2;

    distinct_substrings as f64 / total_substrings as f64
}

fn encode_tokens(tokens: &[String]) -> Vec<usize> {
    let mut ids_by_token: FxHashMap<&str, usize> = FxHashMap::default();
    let mut ids = Vec::with_capacity(tokens.len());
    for token in tokens {
        let token = token.as_str();
        let id = match ids_by_token.get(token) {
            Some(&id) => id,
            None => {
                let id = ids_by_token.len();
                ids_by_token.insert(token, id);
                id
            }
        };
        ids.push(id);
    }
    ids
}

pub fn diversity_batch(
    tokens: Vec<String>,
    ranges: Vec<(usize, usize)>,
) -> Result<Vec<f64>, String> {
    validate_ranges(tokens.len(), &ranges)?;

    let token_ids = encode_tokens(&tokens);
    let diversities = ranges
        .par_iter()
        .map(|&(start, end)| suffix_array_diversity(&token_ids[start..end]))
        .collect();

    Ok(diversities)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_sequence_lower_than_varied() {
        let constant = suffix_array_diversity(&[0, 0, 0, 0]);
        let varied = suffix_array_diversity(&[0, 1, 2, 3]);
        assert!(constant < varied);
    }

    #[test]
    fn constant_sequence_matches_known_ratio() {
        // "AAAA" has 4 distinct substrings (A, AA, AAA, AAAA) out of 10 total.
        let diversity = suffix_array_diversity(&[0, 0, 0, 0]);
        assert!((diversity - 0.4).abs() < 1e-12);
    }

    #[test]
    fn fully_unique_sequence_is_one() {
        let diversity = suffix_array_diversity(&[0, 1, 2, 3, 4]);
        assert!((diversity - 1.0).abs() < 1e-12);
    }

    #[test]
    fn short_sequences_are_zero() {
        assert_eq!(suffix_array_diversity(&[]), 0.0);
        assert_eq!(suffix_array_diversity(&[0]), 0.0);
    }

    #[test]
    fn two_element_sequences_ordered_correctly() {
        let same = suffix_array_diversity(&[0, 0]);
        let diff = suffix_array_diversity(&[0, 1]);
        assert!(diff >= same);
    }

    #[test]
    fn batch_respects_ranges() {
        let tokens: Vec<String> = ["A", "A", "A", "A", "A", "B", "C", "A", "B"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let ranges = vec![(0, 5), (5, 9)];
        let result = diversity_batch(tokens, ranges).unwrap();
        assert_eq!(result.len(), 2);
        assert!(result[0] < result[1]);
    }

    #[test]
    fn batch_rejects_invalid_ranges() {
        let tokens = vec!["A".to_string(), "B".to_string()];
        let result = diversity_batch(tokens, vec![(0, 3)]);
        assert!(result.is_err());
    }
}
