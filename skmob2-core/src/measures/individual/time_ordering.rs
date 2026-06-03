use rayon::prelude::*;

use crate::utils::{split_ranges, validate_uid_len};

pub type IndexRanges = Vec<(usize, usize)>;
pub type OrderedIndexRanges = (Vec<usize>, IndexRanges);

pub fn split_ordered_index_ranges(
    (indices, ranges): OrderedIndexRanges,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let (starts, ends) = split_ranges(ranges);
    (indices, starts, ends)
}

pub fn time_ordered_indices_single_user(timestamps: &[f64]) -> OrderedIndexRanges {
    let mut indices: Vec<usize> = (0..timestamps.len()).collect();
    indices.par_sort_by(|&left, &right| {
        timestamps[left]
            .total_cmp(&timestamps[right])
            .then(left.cmp(&right))
    });
    let n = indices.len();
    (indices, vec![(0, n)])
}

pub fn time_ordered_indices_for_ord_uid_values<T: Ord + Copy + Sync + Send>(
    uids: &[T],
    timestamps: &[f64],
) -> OrderedIndexRanges {
    let n = uids.len();
    let mut grouped: Vec<(T, usize)> = uids
        .par_iter()
        .enumerate()
        .map(|(idx, &uid)| (uid, idx))
        .collect();

    grouped.par_sort_unstable_by(|left, right| left.0.cmp(&right.0));

    let mut ranges = Vec::new();
    if !grouped.is_empty() {
        let mut start = 0;
        for idx in 1..n {
            if grouped[idx].0 != grouped[idx - 1].0 {
                ranges.push((start, idx));
                start = idx;
            }
        }
        ranges.push((start, n));
    }

    let mut indices = vec![0usize; n];
    let indices_ptr = indices.as_mut_ptr() as usize;

    ranges.par_iter().for_each(|&(start, end)| {
        // SAFETY: `ranges` is built from monotonically increasing group boundaries, so each
        // parallel task writes to a unique, non-overlapping slice of `indices`.
        let out_slice = unsafe {
            std::slice::from_raw_parts_mut((indices_ptr as *mut usize).add(start), end - start)
        };

        for offset in 0..(end - start) {
            out_slice[offset] = grouped[start + offset].1;
        }

        out_slice.sort_unstable_by(|&left, &right| {
            timestamps[left]
                .total_cmp(&timestamps[right])
                .then(left.cmp(&right))
        });
    });

    (indices, ranges)
}

pub fn time_ordered_indices_for_f64_uid_values(
    uids: &[f64],
    timestamps: &[f64],
) -> OrderedIndexRanges {
    let n = uids.len();
    let mut grouped: Vec<(f64, usize)> = uids
        .par_iter()
        .enumerate()
        .map(|(idx, &uid)| (uid, idx))
        .collect();

    grouped.par_sort_unstable_by(|left, right| left.0.total_cmp(&right.0));

    let mut ranges = Vec::new();
    if !grouped.is_empty() {
        let mut start = 0;
        for idx in 1..n {
            if grouped[idx].0.total_cmp(&grouped[idx - 1].0) != std::cmp::Ordering::Equal {
                ranges.push((start, idx));
                start = idx;
            }
        }
        ranges.push((start, n));
    }

    let mut indices = vec![0usize; n];
    let indices_ptr = indices.as_mut_ptr() as usize;

    ranges.par_iter().for_each(|&(start, end)| {
        // SAFETY: `ranges` is built from monotonically increasing group boundaries, so each
        // parallel task writes to a unique, non-overlapping slice of `indices`.
        let out_slice = unsafe {
            std::slice::from_raw_parts_mut((indices_ptr as *mut usize).add(start), end - start)
        };

        for offset in 0..(end - start) {
            out_slice[offset] = grouped[start + offset].1;
        }

        out_slice.sort_unstable_by(|&left, &right| {
            timestamps[left]
                .total_cmp(&timestamps[right])
                .then(left.cmp(&right))
        });
    });

    (indices, ranges)
}

pub fn time_ordered_indices_for_option_str_uid_values(
    uids: &[Option<&str>],
    timestamps: &[f64],
) -> Result<OrderedIndexRanges, String> {
    validate_uid_len(timestamps.len(), uids.len())?;
    Ok(time_ordered_indices_for_ord_uid_values(uids, timestamps))
}
