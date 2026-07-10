use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::utils::validate_indexed_coord_ends;

type PredictabilityBatchResult = (Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>);

// skmob-compatible LZ77 entropy estimator — matches scikit-mobility's _true_entropy.
// Distinct from the LZ78 DP estimator used by trajectory_entropy_batch.
fn skmob_lz77_entropy<T: PartialEq>(sequence: &[T]) -> f64 {
    let n = sequence.len();
    if n <= 1 {
        return 0.0;
    }

    // 3.0 accounts for the boundary positions (i=0 and i=n-1) that the loop skips.
    let mut sum_lambda = 3.0f64;

    for i in 1..(n - 1) {
        let mut j = i + 1;
        loop {
            if j >= n {
                j += 1; // reached sequence end; extend by 1 per skmob convention
                break;
            }
            let candidate = &sequence[i..j];
            let prefix = &sequence[..i];
            let clen = candidate.len();
            let found = if clen > prefix.len() {
                false
            } else {
                (0..=(prefix.len() - clen)).any(|k| prefix[k..k + clen] == *candidate)
            };
            if found {
                j += 1;
            } else {
                break;
            }
        }
        sum_lambda += (j - i) as f64;
    }

    (n as f64) * (n as f64).log2() / sum_lambda
}

pub fn real_entropy_batch(
    tokens: Vec<String>,
    ranges: Vec<(usize, usize)>,
) -> Result<Vec<f64>, String> {
    validate_ranges(tokens.len(), &ranges)?;

    let token_ids = encode_tokens(&tokens);
    let entropies = ranges
        .par_iter()
        .map(|&(start, end)| skmob_lz77_entropy(&token_ids[start..end]))
        .collect();

    Ok(entropies)
}

pub fn real_entropy_indexed_impl(
    lats: &[f64],
    lngs: &[f64],
    indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
) -> Result<Vec<f64>, String> {
    validate_indexed_coord_ends(lats, lngs, indices, ends)?;

    Ok((0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];

            let mut ids_by_pair: FxHashMap<(u64, u64), usize> = FxHashMap::default();
            let mut token_ids = Vec::with_capacity(end - start);

            for &idx in &indices[start..end] {
                if valid_rows.is_none_or(|v| v[idx])
                    && lats[idx].is_finite()
                    && lngs[idx].is_finite()
                {
                    let pair = (lats[idx].to_bits(), lngs[idx].to_bits());
                    let next_id = ids_by_pair.len();
                    let id = *ids_by_pair.entry(pair).or_insert(next_id);
                    token_ids.push(id);
                }
            }

            skmob_lz77_entropy(&token_ids)
        })
        .collect())
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

fn validate_ranges(n_tokens: usize, ranges: &[(usize, usize)]) -> Result<(), String> {
    for &(start, end) in ranges {
        if start > end || end > n_tokens {
            return Err("ranges must be valid half-open intervals into tokens".to_string());
        }
    }
    Ok(())
}

fn kontoyiannis_entropy(sequence: &[String]) -> f64 {
    let n = sequence.len();
    if n <= 1 {
        return 0.0;
    }

    let mut col_max = vec![1_usize; n];
    let mut prev_row = vec![1_usize; n];

    for i in 1..n {
        let mut curr_row = vec![1_usize; n];
        for j in (i + 1)..n {
            if sequence[i - 1] == sequence[j - 1] {
                curr_row[j] = prev_row[j - 1] + 1;
            }
            if curr_row[j] > col_max[j] {
                col_max[j] = curr_row[j];
            }
        }
        prev_row = curr_row;
    }

    let lambdas: usize = col_max.iter().sum();
    if lambdas == 0 {
        return 0.0;
    }

    (n as f64 / lambdas as f64) * (n as f64).log2()
}

fn fano_equation_term(predictability: f64, real_entropy: f64, n_unique: usize) -> f64 {
    let p = predictability.clamp(1e-12, 1.0 - 1e-12);
    let binary_entropy = -(p * p.log2() + (1.0 - p) * (1.0 - p).log2());
    binary_entropy + (1.0 - p) * ((n_unique - 1) as f64).log2() - real_entropy
}

fn is_close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-8 + 1e-5 * b.abs()
}

fn solve_max_predictability_with_fano(real_entropy: f64, n_unique: usize) -> f64 {
    if n_unique <= 1 {
        return 1.0;
    }

    let entropy_upper_bound = (n_unique as f64).log2();
    let bounded_entropy = real_entropy.clamp(0.0, entropy_upper_bound);

    if is_close(bounded_entropy, 0.0) {
        return 1.0;
    }
    if is_close(bounded_entropy, entropy_upper_bound) {
        return 1.0 / n_unique as f64;
    }

    let mut low = 1.0 / n_unique as f64;
    let mut high = 1.0;

    for _ in 0..100 {
        let mid = 0.5 * (low + high);
        let value = fano_equation_term(mid, bounded_entropy, n_unique);

        if value.abs() < 1e-8 {
            return mid;
        }

        if value > 0.0 {
            low = mid;
        } else {
            high = mid;
        }
    }

    0.5 * (low + high)
}

pub fn trajectory_entropy_batch(
    tokens: Vec<String>,
    ranges: Vec<(usize, usize)>,
    normalized: bool,
) -> Result<Vec<f64>, String> {
    validate_ranges(tokens.len(), &ranges)?;

    let entropies = ranges
        .par_iter()
        .map(|&(start, end)| {
            let sequence = &tokens[start..end];
            let raw = kontoyiannis_entropy(sequence);
            if !normalized {
                return raw;
            }
            if sequence.len() <= 1 {
                return 0.0;
            }
            (raw / (sequence.len() as f64).log2()).clamp(0.0, 1.0)
        })
        .collect();

    Ok(entropies)
}

pub fn trajectory_predictability_batch(
    tokens: Vec<String>,
    ranges: Vec<(usize, usize)>,
) -> Result<PredictabilityBatchResult, String> {
    validate_ranges(tokens.len(), &ranges)?;

    let rows: Vec<(f64, f64, usize, usize)> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let sequence = &tokens[start..end];
            let n_steps = sequence.len();
            let n_unique = sequence.iter().collect::<FxHashSet<_>>().len();
            let real_entropy = kontoyiannis_entropy(sequence);
            let predictability = solve_max_predictability_with_fano(real_entropy, n_unique);
            (real_entropy, predictability, n_unique, n_steps)
        })
        .collect();

    let mut real_entropies = Vec::with_capacity(rows.len());
    let mut predictabilities = Vec::with_capacity(rows.len());
    let mut n_unique_locations = Vec::with_capacity(rows.len());
    let mut n_steps = Vec::with_capacity(rows.len());

    for (real_entropy, predictability, n_unique, steps) in rows {
        real_entropies.push(real_entropy);
        predictabilities.push(predictability);
        n_unique_locations.push(n_unique);
        n_steps.push(steps);
    }

    Ok((
        real_entropies,
        predictabilities,
        n_unique_locations,
        n_steps,
    ))
}
