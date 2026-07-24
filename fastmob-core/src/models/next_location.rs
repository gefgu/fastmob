//! Order-k Markov chain next-location predictor, with backoff to lower
//! orders on an unseen context.
//!
//! Distinct from `models::markov_diary`'s `MarkovDiaryGenerator`, which is a
//! fixed 48-state hour-of-day x home/away chain that *generates* synthetic
//! mobility diaries. This is an arbitrary-order-k chain fit directly over a
//! user's own dense-coded location-id *sequence*, used to *predict* the next
//! entry in that same sequence -- a different concept despite both being
//! "Markov chains over mobility."
//!
//! Ported from HuMobi's `predictors.markov.MarkovChain` (order-k transition
//! counts with backoff) and `predictors.wrapper.TopLoc` (the order-0,
//! most-frequent-location baseline, which this implementation gets "for
//! free" as the order-0 table).

use rayon::prelude::*;
use rustc_hash::FxHashMap;

#[derive(Clone, Copy)]
pub struct NextLocationConfig {
    pub order: usize,
    pub backoff: bool,
}

impl NextLocationConfig {
    pub fn new(order: usize, backoff: bool) -> Self {
        NextLocationConfig { order, backoff }
    }
}

/// Order-k (and every lower order, for backoff) transition-count tables for
/// one user's location-code sequence.
///
/// `tables[k]` maps a length-`k` context (the `k` most-recent location
/// codes, oldest first) to a map of `next code -> observed count`.
/// `tables[0]`'s only key is the empty context `[]`, mapping to global
/// unigram counts -- i.e. the order-0 table *is* the most-frequent-location
/// baseline (HuMobi's `TopLoc`), obtained for free by always fitting every
/// order `0..=order` simultaneously.
pub struct MarkovLocationModel {
    pub order: usize,
    pub backoff: bool,
    pub tables: Vec<FxHashMap<Vec<u64>, FxHashMap<u64, u32>>>,
}

impl MarkovLocationModel {
    fn empty(order: usize, backoff: bool) -> Self {
        MarkovLocationModel {
            order,
            backoff,
            tables: (0..=order).map(|_| FxHashMap::default()).collect(),
        }
    }
}

fn fit_one_user(seq: &[u64], config: &NextLocationConfig) -> MarkovLocationModel {
    let mut model = MarkovLocationModel::empty(config.order, config.backoff);
    let n = seq.len();
    for (k, table) in model.tables.iter_mut().enumerate() {
        for i in k..n {
            let context = seq[i - k..i].to_vec();
            *table.entry(context).or_default().entry(seq[i]).or_insert(0) += 1;
        }
    }
    model
}

/// Fit one order-k (+ backoff) Markov model per user from each user's
/// chronologically-sorted location-code sequence
/// (`location_codes[sorted_indices[start..end]]`), in parallel via rayon.
pub fn markov_fit_indexed(
    location_codes: &[u64],
    sorted_indices: &[usize],
    ends: &[usize],
    config: &NextLocationConfig,
) -> Result<Vec<MarkovLocationModel>, String> {
    for &idx in sorted_indices {
        if idx >= location_codes.len() {
            return Err("index must be within location_codes bounds".to_string());
        }
    }
    Ok((0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let seq: Vec<u64> = sorted_indices[start..end]
                .iter()
                .map(|&idx| location_codes[idx])
                .collect();
            fit_one_user(&seq, config)
        })
        .collect())
}

struct PredictionResult {
    codes: Vec<u64>,
    probs: Vec<f64>,
}

fn predict_one(model: &MarkovLocationModel, context: &[u64], top_k: usize) -> PredictionResult {
    let max_k = model.order.min(context.len());
    let mut k = max_k;
    loop {
        let ctx_key: Vec<u64> = if k == 0 {
            Vec::new()
        } else {
            context[context.len() - k..].to_vec()
        };
        if let Some(next_counts) = model.tables[k].get(&ctx_key)
            && !next_counts.is_empty()
        {
            let mut items: Vec<(u64, u32)> = next_counts
                .iter()
                .map(|(&code, &count)| (code, count))
                .collect();
            items.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
            items.truncate(top_k);
            let total: u32 = next_counts.values().sum();
            return PredictionResult {
                codes: items.iter().map(|&(c, _)| c).collect(),
                probs: items
                    .iter()
                    .map(|&(_, cnt)| f64::from(cnt) / f64::from(total))
                    .collect(),
            };
        }
        if k == 0 || !model.backoff {
            return PredictionResult {
                codes: Vec::new(),
                probs: Vec::new(),
            };
        }
        k -= 1;
    }
}

/// Batched next-location prediction: one context vector per user (in the
/// same user order as `models`), returning up to `top_k` candidates per
/// user by descending count, backing off from order `model.order` down to
/// order 0 on an unseen context (when `model.backoff`). Flat output +
/// user-boundary convention (matches `location_frequency.rs`/
/// `recency_rank.rs`) since the candidate count varies per user.
pub fn markov_predict_batch(
    models: &[MarkovLocationModel],
    context_codes: &[u64],
    context_starts: &[usize],
    context_ends: &[usize],
    top_k: usize,
) -> Result<(Vec<u64>, Vec<f64>, Vec<usize>, Vec<usize>), String> {
    if models.len() != context_starts.len() || models.len() != context_ends.len() {
        return Err(
            "models, context_starts, and context_ends must have the same length".to_string(),
        );
    }
    let n_users = models.len();
    let mut out_codes = Vec::new();
    let mut out_probs = Vec::new();
    let mut out_starts = Vec::with_capacity(n_users);
    let mut out_ends = Vec::with_capacity(n_users);
    for i in 0..n_users {
        let start = context_starts[i];
        let end = context_ends[i];
        if start > end || end > context_codes.len() {
            return Err(
                "context_starts/context_ends must be within context_codes bounds".to_string(),
            );
        }
        let context = &context_codes[start..end];
        let result = predict_one(&models[i], context, top_k);
        out_starts.push(out_codes.len());
        out_codes.extend(result.codes);
        out_probs.extend(result.probs);
        out_ends.push(out_codes.len());
    }
    Ok((out_codes, out_probs, out_starts, out_ends))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_1_predicts_deterministic_cycle_perfectly() {
        // A->B->C->A->B->C->... : order-1 Markov chain should predict the
        // next element with 100% confidence from any single-element context.
        let seq: Vec<u64> = (0..12).map(|i| i % 3).collect();
        let config = NextLocationConfig::new(1, true);
        let model = fit_one_user(&seq, &config);

        let pred = predict_one(&model, &[0], 1);
        assert_eq!(pred.codes, vec![1]);
        assert_eq!(pred.probs, vec![1.0]);

        let pred = predict_one(&model, &[1], 1);
        assert_eq!(pred.codes, vec![2]);

        let pred = predict_one(&model, &[2], 1);
        assert_eq!(pred.codes, vec![0]);
    }

    #[test]
    fn order_0_is_most_frequent_location_baseline() {
        let seq = vec![5u64, 5, 5, 7, 9];
        let config = NextLocationConfig::new(0, true);
        let model = fit_one_user(&seq, &config);
        let pred = predict_one(&model, &[], 1);
        assert_eq!(pred.codes, vec![5]);
    }

    #[test]
    fn backoff_falls_back_to_lower_order_on_unseen_context() {
        // Order-2 context [9, 9] never appears in the fitted sequence, so
        // with backoff=true it should fall back down to order-0 (most
        // frequent location overall) instead of returning nothing.
        let seq = vec![1u64, 2, 1, 2, 1, 2, 3];
        let config = NextLocationConfig::new(2, true);
        let model = fit_one_user(&seq, &config);
        let pred = predict_one(&model, &[9, 9], 1);
        assert!(!pred.codes.is_empty());
    }

    #[test]
    fn no_backoff_returns_empty_on_unseen_context() {
        let seq = vec![1u64, 2, 1, 2];
        let config = NextLocationConfig::new(2, false);
        let model = fit_one_user(&seq, &config);
        let pred = predict_one(&model, &[9, 9], 1);
        assert!(pred.codes.is_empty());
    }

    #[test]
    fn top_k_returns_multiple_candidates_sorted_by_count_desc() {
        let seq = vec![0u64, 1, 0, 2, 0, 1, 0, 1];
        let config = NextLocationConfig::new(1, true);
        let model = fit_one_user(&seq, &config);
        let pred = predict_one(&model, &[0], 2);
        assert_eq!(pred.codes, vec![1, 2]);
        assert!(pred.probs[0] > pred.probs[1]);
    }
}
