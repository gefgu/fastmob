//! Arranging rows into `(user, time)` order.
//!
//! This is the hot spot of the whole crate. A profile of the prototype at 4M
//! Brightkite rows put `LazyFrame::sort([uid, datetime])` at 54.8 ms of a
//! 64.1 ms call — the Haversine kernel itself was 4.2 ms. citybehavex's own
//! implementation sorts exactly the same way, so beating a whole-frame Polars
//! sort is the entire point of this module.
//!
//! The strategy is a two-stage sort that never compares two rows against each
//! other:
//!
//! 1. **Counting sort by user code.** Codes are bounded `u32`s (see
//!    [`super::uid_codes`]), so a histogram plus prefix sum lays every user's
//!    rows out contiguously in one linear pass. The prefix sums *are* the group
//!    boundaries, so the separate boundary-detection scan disappears too.
//! 2. **Per-group timestamp sort, in parallel.** Each user's slice is sorted
//!    independently with rayon. Groups are small (Brightkite averages ~69 rows
//!    across 58k users), and the work is embarrassingly parallel.
//!
//! Both stages are stable, so the result is identical to a stable sort by
//! `(uid, datetime)` — which is what fastmob's Python path produces.

use rayon::prelude::*;

/// A row arrangement plus the per-user group boundaries it induces.
pub(crate) struct Arrangement {
    /// `None` when the input was already in `(user, time)` order, so callers
    /// can skip the gather entirely.
    pub permutation: Option<Vec<u32>>,
    /// Exclusive end offset of each user's contiguous run, in arranged order.
    /// This is fastmob-core's `ends` convention.
    pub ends: Vec<usize>,
}

/// True when rows are already grouped by user and non-decreasing in time
/// within each group, i.e. already equal to the sorted arrangement.
///
/// Worth the linear scan: it costs ~1 ms at 4M rows and skips a gather of
/// every column. It does *not* fire on the Brightkite check-in dataset (user
/// `637` reverses direction around row 297458), but simulator output is
/// emitted grouped and ordered, which is the common case in citybehavex.
fn is_arranged(codes: &[u32], timestamps: &[i64]) -> bool {
    codes
        .windows(2)
        .zip(timestamps.windows(2))
        .all(|(code, ts)| code[1] > code[0] || (code[1] == code[0] && ts[1] >= ts[0]))
}

/// Group end offsets for codes already known to be grouped by user.
pub(crate) fn ends_from_arranged_codes(codes: &[u32]) -> Vec<usize> {
    if codes.is_empty() {
        return Vec::new();
    }
    let mut ends = Vec::new();
    for (idx, pair) in codes.windows(2).enumerate() {
        if pair[1] != pair[0] {
            ends.push(idx + 1);
        }
    }
    ends.push(codes.len());
    ends
}

pub(crate) fn arrange(codes: &[u32], timestamps: &[i64], num_codes: usize) -> Arrangement {
    debug_assert_eq!(codes.len(), timestamps.len());
    let len = codes.len();
    if len == 0 {
        return Arrangement {
            permutation: None,
            ends: Vec::new(),
        };
    }

    if is_arranged(codes, timestamps) {
        return Arrangement {
            permutation: None,
            ends: ends_from_arranged_codes(codes),
        };
    }

    // Stage 1: counting sort by user code.
    let mut offsets = vec![0usize; num_codes + 1];
    for &code in codes {
        offsets[code as usize + 1] += 1;
    }
    for idx in 1..offsets.len() {
        offsets[idx] += offsets[idx - 1];
    }

    let ends: Vec<usize> = offsets
        .windows(2)
        .filter(|bounds| bounds[1] > bounds[0])
        .map(|bounds| bounds[1])
        .collect();

    let mut cursors = offsets.clone();
    let mut permutation = vec![0u32; len];
    for (row, &code) in codes.iter().enumerate() {
        let slot = &mut cursors[code as usize];
        permutation[*slot] = row as u32;
        *slot += 1;
    }

    // Stage 2: sort each user's rows by time, in parallel. `sort_by_key` is
    // stable, so rows with equal timestamps keep their original relative order
    // — matching the stable sort fastmob's Python path performs.
    let mut rest = permutation.as_mut_slice();
    let mut groups = Vec::with_capacity(ends.len());
    let mut consumed = 0usize;
    for &end in &ends {
        let (group, tail) = rest.split_at_mut(end - consumed);
        groups.push(group);
        rest = tail;
        consumed = end;
    }
    groups
        .into_par_iter()
        .for_each(|group| group.sort_by_key(|&row| timestamps[row as usize]));

    Arrangement {
        permutation: Some(permutation),
        ends,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference implementation: a plain stable sort by `(code, timestamp)`.
    fn reference_permutation(codes: &[u32], timestamps: &[i64]) -> Vec<u32> {
        let mut rows: Vec<u32> = (0..codes.len() as u32).collect();
        rows.sort_by_key(|&row| (codes[row as usize], timestamps[row as usize]));
        rows
    }

    #[test]
    fn already_arranged_input_skips_the_permutation() {
        let codes = [0u32, 0, 1, 1, 2];
        let timestamps = [0i64, 5, 0, 3, 7];
        let arrangement = arrange(&codes, &timestamps, 3);
        assert!(arrangement.permutation.is_none());
        assert_eq!(arrangement.ends, vec![2, 4, 5]);
    }

    #[test]
    fn interleaved_input_matches_a_stable_sort() {
        let codes = [1u32, 0, 1, 0, 0, 1];
        let timestamps = [2_000i64, 2_000, 0, 0, 1_000, 1_000];
        let arrangement = arrange(&codes, &timestamps, 2);
        assert_eq!(
            arrangement.permutation.unwrap(),
            reference_permutation(&codes, &timestamps)
        );
        assert_eq!(arrangement.ends, vec![3, 6]);
    }

    #[test]
    fn equal_timestamps_keep_original_row_order() {
        let codes = [0u32, 0, 0];
        let timestamps = [5i64, 1, 5];
        let arrangement = arrange(&codes, &timestamps, 1);
        // Rows 0 and 2 tie at t=5 and must stay in ascending row order.
        assert_eq!(arrangement.permutation.unwrap(), vec![1, 0, 2]);
    }

    #[test]
    fn descending_time_within_a_group_is_not_treated_as_arranged() {
        let codes = [0u32, 0];
        let timestamps = [10i64, 0];
        assert!(arrange(&codes, &timestamps, 1).permutation.is_some());
    }

    #[test]
    fn sparse_code_space_produces_no_empty_groups() {
        let codes = [7u32, 0, 7];
        let timestamps = [1i64, 2, 0];
        let arrangement = arrange(&codes, &timestamps, 8);
        assert_eq!(arrangement.ends, vec![1, 3]);
        assert_eq!(
            arrangement.permutation.unwrap(),
            reference_permutation(&codes, &timestamps)
        );
    }

    #[test]
    fn matches_a_stable_sort_on_a_larger_pseudorandom_input() {
        let len = 5_000usize;
        let codes: Vec<u32> = (0..len).map(|i| ((i * 7919) % 97) as u32).collect();
        let timestamps: Vec<i64> = (0..len).map(|i| ((i * 104_729) % 1_000) as i64).collect();
        let arrangement = arrange(&codes, &timestamps, 97);
        assert_eq!(
            arrangement.permutation.unwrap(),
            reference_permutation(&codes, &timestamps)
        );
    }

    #[test]
    fn empty_input_is_handled() {
        let arrangement = arrange(&[], &[], 0);
        assert!(arrangement.permutation.is_none());
        assert!(arrangement.ends.is_empty());
    }
}
