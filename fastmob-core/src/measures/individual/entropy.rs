use rayon::prelude::*;
use rustc_hash::FxHashSet;

type PredictabilityBatchResult = (Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>);

pub fn real_entropy_users(
    location_ids: Vec<u32>,
    ranges: Vec<(usize, usize)>,
    normalized: bool,
) -> Result<Vec<f64>, String> {
    validate_ranges(location_ids.len(), &ranges)?;

    let entropies = ranges
        .par_iter()
        .map(|&(start, end)| {
            let sequence = &location_ids[start..end];
            let raw = kontoyiannis_entropy(sequence);
            if normalized && sequence.len() > 1 {
                (raw / (sequence.len() as f64).log2()).clamp(0.0, 1.0)
            } else if normalized {
                0.0
            } else {
                raw
            }
        })
        .collect();

    Ok(entropies)
}

fn validate_ranges(n_tokens: usize, ranges: &[(usize, usize)]) -> Result<(), String> {
    for &(start, end) in ranges {
        if start > end || end > n_tokens {
            return Err("ranges must be valid half-open intervals into tokens".to_string());
        }
    }
    Ok(())
}

fn kontoyiannis_entropy<T: PartialEq>(sequence: &[T]) -> f64 {
    let n = sequence.len();
    if n <= 1 {
        return 0.0;
    }

    let mut col_max = vec![1_usize; n];
    let mut prev_row = vec![1_usize; n];
    let mut curr_row = vec![1_usize; n];

    for i in 1..n {
        for j in (i + 1)..n {
            if sequence[i - 1] == sequence[j - 1] {
                curr_row[j] = prev_row[j - 1] + 1;
            } else {
                curr_row[j] = 1;
            }
            if curr_row[j] > col_max[j] {
                col_max[j] = curr_row[j];
            }
        }
        std::mem::swap(&mut prev_row, &mut curr_row);
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

pub fn trajectory_predictability_batch(
    location_ids: Vec<u32>,
    ranges: Vec<(usize, usize)>,
) -> Result<PredictabilityBatchResult, String> {
    validate_ranges(location_ids.len(), &ranges)?;

    let rows: Vec<(f64, f64, usize, usize)> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let sequence = &location_ids[start..end];
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
