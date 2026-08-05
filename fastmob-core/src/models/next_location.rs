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
//!
//! ## Arena-allocated trie backend
//!
//! Per-user transition counts are stored as an **arena-allocated prefix
//! trie** over **user-local dense `u16` codes**, not the nested
//! `FxHashMap<Vec<u64>, FxHashMap<u64, u32>>` tables an earlier version of
//! this module used. Two things motivated the rewrite: that nested design
//! heap-allocated a fresh `Vec<u64>` context key on *every* (order, position)
//! pair during fitting (`seq[i - k..i].to_vec()`), and hashed a
//! variable-length `Vec<u64>` per lookup at both fit and predict time --
//! both expensive relative to how small the actual per-context transition
//! counts are for one user (human mobility is highly routine: a handful of
//! distinct next-locations per context is typical).
//!
//! The trie encodes context length via tree depth: the root (depth 0) is
//! the order-0 table (global unigram counts, i.e. HuMobi's `TopLoc`
//! baseline "for free"); a node at depth `d` represents the `d` most recent
//! locations before the position being predicted. Edges are traversed
//! **most-recent-first** during both fitting and prediction (i.e. the first
//! edge taken from the root is one step back in time, the second edge two
//! steps back, etc.) so that fitting a position's up-to-`order` contexts is
//! a *single* incremental walk down the trie rather than `order` separate
//! lookups, and backoff during prediction is simply "use the deepest node
//! reached" rather than a series of independent re-lookups at shrinking `k`
//! -- every node the trie ever creates has a non-empty `transitions` list
//! (a transition is always recorded at the moment a node is reached/created
//! during fitting), so the deepest reachable node already *is* the correct
//! backoff target.
//!
//! Per-user location codes are further re-mapped from the (sparse, global)
//! `u64` codes into a dense, per-user `u16` numbering (`fit_one_user`'s
//! local dense-coding pass) so that both the trie's edge labels and
//! transition keys fit in 2 bytes instead of 8 -- trie children/transitions
//! are small `Vec<(u16, _)>`s scanned linearly rather than hashed, which is
//! the same "small-N linear scan beats hashing" bet the rest of this
//! design makes, and keeps nodes compact and cache-friendly. A user with
//! more than `u16::MAX + 1` distinct locations (unseen in practice for the
//! real GPS/check-in datasets this has been benchmarked against, whose most
//! active users see totals, let alone *distinct* locations, in the
//! thousands at most) fails fast with an error rather than silently
//! wrapping codes.

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

/// One node of a user's context trie: `depth` (implicit, via tree position)
/// is the context length that node represents. `children` maps "one more
/// step back in time has this location code" -> child node index;
/// `transitions` maps "the observed next-location from this exact context"
/// -> observed count. Both are small `Vec`s (see module docs), scanned
/// linearly rather than hashed.
#[derive(Default, Clone)]
struct TrieNode {
    children: Vec<(u16, u32)>,
    transitions: Vec<(u16, u32)>,
}

/// Order-k (+ backoff) Markov model for one user, backed by an
/// arena-allocated context trie over that user's own dense-coded (`u16`)
/// location sequence. See module docs for the trie's shape/traversal
/// convention.
pub struct MarkovLocationModel {
    order: usize,
    backoff: bool,
    /// `local_to_global[local_code as usize] == global_code`. Also doubles
    /// as the lookup table for translating an incoming (global-coded)
    /// prediction context into this user's local codes: a linear scan over
    /// this array, not a second hash map, per the same small-N bet the trie
    /// itself makes (at most `order` such lookups per `predict_one` call).
    local_to_global: Vec<u64>,
    /// Flat node storage; index 0 is always the root (order-0 context).
    /// Child/parent links are plain `u32` indices into this `Vec`, not
    /// pointers -- keeps every node contiguous and cheaply relocatable, and
    /// avoids `Rc`/`Box` indirection entirely.
    arena: Vec<TrieNode>,
}

/// Increment `node`'s count for `next`, appending a fresh `(next, 1)` entry
/// if this is the first time `next` has been observed from `node`'s context.
fn record_transition(node: &mut TrieNode, next: u16) {
    for entry in node.transitions.iter_mut() {
        if entry.0 == next {
            entry.1 += 1;
            return;
        }
    }
    node.transitions.push((next, 1));
}

/// Return the child of `arena[node_idx]` reached via edge `loc`, inserting
/// a fresh empty node and edge first if it doesn't exist yet.
fn child_or_insert(arena: &mut Vec<TrieNode>, node_idx: u32, loc: u16) -> u32 {
    if let Some(&(_, child_idx)) = arena[node_idx as usize]
        .children
        .iter()
        .find(|&&(code, _)| code == loc)
    {
        return child_idx;
    }
    let child_idx = arena.len() as u32;
    arena.push(TrieNode::default());
    arena[node_idx as usize].children.push((loc, child_idx));
    child_idx
}

/// Fit one user's order-k (+ backoff) context trie from their
/// chronologically-sorted global location-code sequence.
///
/// Two passes: (1) a one-time dense-coding pass mapping this user's
/// distinct global `u64` codes to sequential `u16`s (a temporary hash map
/// is used *here only* -- a single flat map for a single `O(seq.len())`
/// pass, not the nested per-transition hashing this module used to do; see
/// module docs), and (2) the zero-heap-allocation trie-building loop
/// itself: for each position, walk backward up to `order` steps,
/// descending/creating one trie node per step and recording a transition
/// at each depth reached -- no `Vec` is allocated per (position, order)
/// pair the way the old `seq[i - k..i].to_vec()` context key was.
fn fit_one_user(seq: &[u64], config: &NextLocationConfig) -> Result<MarkovLocationModel, String> {
    let mut global_to_local: FxHashMap<u64, u16> = FxHashMap::default();
    let mut local_to_global: Vec<u64> = Vec::new();
    let mut local_seq: Vec<u16> = Vec::with_capacity(seq.len());
    for &global in seq {
        let local = match global_to_local.get(&global) {
            Some(&code) => code,
            None => {
                let next_code = local_to_global.len();
                if next_code > u16::MAX as usize {
                    return Err(format!(
                        "user has more than {} distinct locations, exceeding the u16 \
                         dense-coding limit used by the context-trie backend",
                        u16::MAX as usize + 1
                    ));
                }
                let code = next_code as u16;
                local_to_global.push(global);
                global_to_local.insert(global, code);
                code
            }
        };
        local_seq.push(local);
    }

    let mut arena: Vec<TrieNode> = vec![TrieNode::default()];
    let n = local_seq.len();
    for i in 0..n {
        let next = local_seq[i];
        record_transition(&mut arena[0], next);

        let max_steps = config.order.min(i);
        let mut node_idx: u32 = 0;
        for step in 1..=max_steps {
            let loc = local_seq[i - step];
            node_idx = child_or_insert(&mut arena, node_idx, loc);
            record_transition(&mut arena[node_idx as usize], next);
        }
    }

    Ok(MarkovLocationModel {
        order: config.order,
        backoff: config.backoff,
        local_to_global,
        arena,
    })
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
    (0..ends.len())
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
        .collect()
}

struct PredictionResult {
    codes: Vec<u64>,
    probs: Vec<f64>,
}

/// Descending-by-count comparator, ties broken by ascending **global** code
/// (matches the old `FxHashMap`-backed implementation's tie-break, which
/// iterated a hash map in arbitrary order and then sorted by the real
/// global `u64` code). Deliberately takes `(u64, u32)` pairs, not the
/// trie's internal `(u16, u32)` local-code pairs: local codes are assigned
/// in first-seen order per user, which has no relationship to global code
/// order, so tie-breaking on the local code would silently reorder ties
/// relative to the original implementation -- caught by
/// `differential_matches_naive_nested_hashmap_reference` before this
/// comment existed.
fn by_count_desc(a: &(u64, u32), b: &(u64, u32)) -> std::cmp::Ordering {
    b.1.cmp(&a.1).then(a.0.cmp(&b.0))
}

/// Reduce `items` (already translated to global codes) to (up to) its
/// top-`top_k` entries by descending count, fully ordered. Uses
/// `select_nth_unstable_by` for an O(N) partition step when `items` is
/// larger than `top_k`, rather than an O(N log N) full sort -- only the
/// resulting (small, <= `top_k`-sized) slice is fully sorted afterward, for
/// a deterministic final order.
fn select_top_k(items: &mut Vec<(u64, u32)>, top_k: usize) {
    if items.len() > top_k {
        if top_k == 0 {
            items.clear();
            return;
        }
        items.select_nth_unstable_by(top_k - 1, by_count_desc);
        items.truncate(top_k);
    }
    items.sort_by(by_count_desc);
}

fn predict_one(model: &MarkovLocationModel, context: &[u64], top_k: usize) -> PredictionResult {
    let max_k = model.order.min(context.len());

    // Walk backward from the most recent context element, descending the
    // trie one edge per step. `reached_depth` stops growing the moment a
    // step's global code is unknown to this user or has no matching child
    // edge -- at that point `node_idx` is already the correct backoff
    // target (see module docs: every node the trie contains has recorded
    // at least one transition), so there is no need for a second,
    // shrinking-k lookup loop the way the old table-per-order design
    // needed.
    let mut node_idx: u32 = 0;
    let mut reached_depth: usize = 0;
    for step in 1..=max_k {
        let global = context[context.len() - step];
        let Some(local) = model
            .local_to_global
            .iter()
            .position(|&code| code == global)
        else {
            break;
        };
        let local = local as u16;
        let Some(&(_, child_idx)) = model.arena[node_idx as usize]
            .children
            .iter()
            .find(|&&(code, _)| code == local)
        else {
            break;
        };
        node_idx = child_idx;
        reached_depth = step;
    }

    // Backoff disabled and the exact order-k context wasn't fully reached:
    // return nothing, matching the old implementation's "only ever try
    // k == max_k when backoff is off" behavior -- do not fall back to
    // whatever shorter context was reached.
    if reached_depth < max_k && !model.backoff {
        return PredictionResult {
            codes: Vec::new(),
            probs: Vec::new(),
        };
    }

    let node = &model.arena[node_idx as usize];
    if node.transitions.is_empty() {
        return PredictionResult {
            codes: Vec::new(),
            probs: Vec::new(),
        };
    }

    let total: u32 = node.transitions.iter().map(|&(_, count)| count).sum();
    let mut items: Vec<(u64, u32)> = node
        .transitions
        .iter()
        .map(|&(local, count)| (model.local_to_global[local as usize], count))
        .collect();
    select_top_k(&mut items, top_k);

    PredictionResult {
        codes: items.iter().map(|&(global, _)| global).collect(),
        probs: items
            .iter()
            .map(|&(_, count)| f64::from(count) / f64::from(total))
            .collect(),
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
    use std::collections::HashMap;

    fn fit(seq: &[u64], config: &NextLocationConfig) -> MarkovLocationModel {
        fit_one_user(seq, config).expect("test sequences never exceed the u16 dense-coding limit")
    }

    #[test]
    fn order_1_predicts_deterministic_cycle_perfectly() {
        // A->B->C->A->B->C->... : order-1 Markov chain should predict the
        // next element with 100% confidence from any single-element context.
        let seq: Vec<u64> = (0..12).map(|i| i % 3).collect();
        let config = NextLocationConfig::new(1, true);
        let model = fit(&seq, &config);

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
        let model = fit(&seq, &config);
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
        let model = fit(&seq, &config);
        let pred = predict_one(&model, &[9, 9], 1);
        assert!(!pred.codes.is_empty());
    }

    #[test]
    fn no_backoff_returns_empty_on_unseen_context() {
        let seq = vec![1u64, 2, 1, 2];
        let config = NextLocationConfig::new(2, false);
        let model = fit(&seq, &config);
        let pred = predict_one(&model, &[9, 9], 1);
        assert!(pred.codes.is_empty());
    }

    #[test]
    fn top_k_returns_multiple_candidates_sorted_by_count_desc() {
        let seq = vec![0u64, 1, 0, 2, 0, 1, 0, 1];
        let config = NextLocationConfig::new(1, true);
        let model = fit(&seq, &config);
        let pred = predict_one(&model, &[0], 2);
        assert_eq!(pred.codes, vec![1, 2]);
        assert!(pred.probs[0] > pred.probs[1]);
    }

    #[test]
    fn no_backoff_still_predicts_when_full_order_context_is_known() {
        // Regression guard for the reached_depth < max_k check: a *known*
        // full-length context with backoff disabled must still predict
        // (only an *unreached* full-length context should return empty).
        let seq = vec![1u64, 2, 3, 1, 2, 3, 1, 2, 3];
        let config = NextLocationConfig::new(2, false);
        let model = fit(&seq, &config);
        let pred = predict_one(&model, &[1, 2], 1);
        assert_eq!(pred.codes, vec![3]);
    }

    #[test]
    fn partially_known_context_backs_off_by_one_level_not_to_zero() {
        // context = [42 (never seen), 1 (seen)] at order 2: the most recent
        // element (1) is known, but the one before it (42) isn't, so the
        // walk should reach depth 1 (context=[1]) rather than falling all
        // the way to depth 0 (root).
        let seq = vec![1u64, 2, 1, 2, 1, 2, 1, 9];
        let config = NextLocationConfig::new(2, true);
        let model = fit(&seq, &config);
        let pred = predict_one(&model, &[42, 1], 1);
        // From context [1] alone (order-1), 1 is always followed by 2
        // except the final occurrence (followed by 9) -- 2 should win.
        assert_eq!(pred.codes, vec![2]);
    }

    #[test]
    fn user_local_dense_coding_handles_non_contiguous_global_codes() {
        // Global codes are sparse/arbitrary (as real location-id codes
        // are), not small contiguous integers like the other tests happen
        // to use -- exercise that explicitly.
        let seq = vec![900_001u64, 42, 900_001, 42, 900_001, 7];
        let config = NextLocationConfig::new(1, true);
        let model = fit(&seq, &config);
        let pred = predict_one(&model, &[900_001], 1);
        assert_eq!(pred.codes, vec![42]);
    }

    // -----------------------------------------------------------------
    // Differential test against a from-scratch naive reference
    // implementation (the pre-trie nested-hashmap design), to gain strong
    // confidence the trie rewrite is behaviorally equivalent, not just
    // passing the hand-picked cases above.
    // -----------------------------------------------------------------

    fn naive_fit(
        seq: &[u64],
        order: usize,
    ) -> Vec<HashMap<Vec<u64>, HashMap<u64, u32>>> {
        let mut tables: Vec<HashMap<Vec<u64>, HashMap<u64, u32>>> =
            (0..=order).map(|_| HashMap::new()).collect();
        let n = seq.len();
        for (k, table) in tables.iter_mut().enumerate() {
            for i in k..n {
                let context = seq[i - k..i].to_vec();
                *table.entry(context).or_default().entry(seq[i]).or_insert(0) += 1;
            }
        }
        tables
    }

    /// Returns the full (code, count) set HuMobi/the old implementation
    /// would predict from, at whatever order backoff settles on -- same
    /// semantics as `predict_one`, but built independently from a plain
    /// `HashMap`, for differential comparison.
    fn naive_predict_set(
        tables: &[HashMap<Vec<u64>, HashMap<u64, u32>>],
        context: &[u64],
        order: usize,
        backoff: bool,
    ) -> Option<HashMap<u64, u32>> {
        let max_k = order.min(context.len());
        let mut k = max_k;
        loop {
            let ctx_key: Vec<u64> = if k == 0 {
                Vec::new()
            } else {
                context[context.len() - k..].to_vec()
            };
            if let Some(next_counts) = tables[k].get(&ctx_key)
                && !next_counts.is_empty()
            {
                return Some(next_counts.clone());
            }
            if k == 0 || !backoff {
                return None;
            }
            k -= 1;
        }
    }

    /// Deterministic xorshift-ish PRNG (no external crate needed for a
    /// unit test) so this test is reproducible across runs.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1);
            self.0 >> 33
        }
        fn range(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    #[test]
    fn differential_matches_naive_nested_hashmap_reference() {
        let mut rng = Lcg(0x5EED_u64);
        for trial in 0..200 {
            let seq_len = 1 + (rng.range(40) as usize);
            let vocab = 1 + (rng.range(6) as usize); // small vocab -> lots of repeated contexts
            let order = (rng.range(4)) as usize; // 0..=3
            let backoff = rng.range(2) == 0;
            let top_k = 1 + (rng.range(3) as usize);

            let seq: Vec<u64> = (0..seq_len)
                .map(|_| 1_000_000 + rng.range(vocab as u64)) // sparse-looking global codes
                .collect();

            let config = NextLocationConfig::new(order, backoff);
            let trie_model = fit(&seq, &config);
            let naive_tables = naive_fit(&seq, order);

            // Try every suffix of the sequence as a context, plus a couple
            // of never-seen contexts, so both known and unknown contexts
            // get exercised.
            let mut contexts: Vec<Vec<u64>> = (0..=seq_len)
                .map(|i| seq[seq_len - i..].to_vec())
                .collect();
            contexts.push(vec![999_999_999, 999_999_998]);

            for context in contexts {
                let trie_pred = predict_one(&trie_model, &context, top_k);
                let naive_set = naive_predict_set(&naive_tables, &context, order, backoff);

                match naive_set {
                    None => {
                        assert!(
                            trie_pred.codes.is_empty(),
                            "trial {trial}: naive found nothing but trie predicted {:?} \
                             (seq={seq:?}, order={order}, backoff={backoff}, context={context:?})",
                            trie_pred.codes
                        );
                    }
                    Some(expected_counts) => {
                        assert!(
                            !trie_pred.codes.is_empty(),
                            "trial {trial}: naive found {expected_counts:?} but trie predicted \
                             nothing (seq={seq:?}, order={order}, backoff={backoff}, \
                             context={context:?})"
                        );
                        let total: u32 = expected_counts.values().sum();
                        // The trie's top-k must be a subset of the naive
                        // reference's full transition set, with matching
                        // probabilities, and must be truly the top-k by
                        // count (descending, ties broken by code).
                        let mut expected_sorted: Vec<(u64, u32)> =
                            expected_counts.iter().map(|(&c, &n)| (c, n)).collect();
                        expected_sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
                        expected_sorted.truncate(top_k);

                        assert_eq!(
                            trie_pred.codes,
                            expected_sorted.iter().map(|&(c, _)| c).collect::<Vec<_>>(),
                            "trial {trial}: top-k code mismatch (seq={seq:?}, order={order}, \
                             backoff={backoff}, context={context:?})"
                        );
                        let expected_probs: Vec<f64> = expected_sorted
                            .iter()
                            .map(|&(_, n)| f64::from(n) / f64::from(total))
                            .collect();
                        for (got, want) in trie_pred.probs.iter().zip(expected_probs.iter()) {
                            assert!(
                                (got - want).abs() < 1e-12,
                                "trial {trial}: prob mismatch {got} vs {want} \
                                 (seq={seq:?}, order={order}, backoff={backoff}, \
                                 context={context:?})"
                            );
                        }
                    }
                }
            }
        }
    }
}
