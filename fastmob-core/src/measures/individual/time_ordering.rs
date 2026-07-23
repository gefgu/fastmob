use rayon::prelude::*;

use crate::utils::ends_from_ranges;

pub type IndexRanges = Vec<(usize, usize)>;
pub type OrderedIndexRanges = (Vec<usize>, IndexRanges);

pub fn split_ordered_index_ranges(
    (indices, ranges): OrderedIndexRanges,
) -> (Vec<usize>, Vec<usize>) {
    (indices, ends_from_ranges(&ranges))
}

pub fn presorted_ranges_for_u64_codes(codes: &[u64]) -> (Vec<usize>, Vec<usize>) {
    let n = codes.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let mut boundaries: Vec<usize> = codes
        .par_windows(2)
        .enumerate()
        .filter_map(|(idx, window)| (window[0] != window[1]).then_some(idx + 1))
        .collect();
    boundaries.sort_unstable();

    let mut starts = Vec::with_capacity(boundaries.len() + 1);
    starts.push(0);
    starts.extend_from_slice(&boundaries);

    let mut ends = boundaries;
    ends.push(n);
    (starts, ends)
}

pub fn time_ordered_indices_single_user(timestamps: &[f64]) -> OrderedIndexRanges {
    let mut indices: Vec<usize> = (0..timestamps.len()).collect();
    indices.par_sort_by(|&left, &right| {
        timestamps[left]
            .total_cmp(&timestamps[right])
            .then(left.cmp(&right))
    });
    let n = indices.len();
    if n == 0 {
        return (indices, Vec::new());
    }
    (indices, vec![(0, n)])
}

pub fn time_ordered_indices_for_u64_codes(
    codes: &[u64],
    timestamps: &[f64],
    num_groups: usize,
) -> Result<OrderedIndexRanges, String> {
    let n = codes.len();
    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if num_groups == 0 {
        return Err(
            "num_groups must be greater than zero when uid codes are not empty".to_string(),
        );
    }

    let mut offsets = vec![0usize; num_groups];
    for &code in codes {
        let group_id =
            usize::try_from(code).map_err(|_| "uid code must fit into usize".to_string())?;
        let count = offsets
            .get_mut(group_id)
            .ok_or_else(|| "uid code must be less than num_groups".to_string())?;
        *count += 1;
    }

    let mut ranges = Vec::with_capacity(num_groups);
    let mut current_offset = 0usize;
    for count in offsets.iter_mut() {
        if *count == 0 {
            continue;
        }
        let start = current_offset;
        let end = current_offset + *count;
        ranges.push((start, end));
        *count = start;
        current_offset = end;
    }

    let mut indices = vec![0usize; n];
    for (row_idx, &code) in codes.iter().enumerate() {
        let group_id = code as usize;
        let pos = offsets[group_id];
        indices[pos] = row_idx;
        offsets[group_id] += 1;
    }

    let indices_ptr = indices.as_mut_ptr() as usize;
    ranges.par_iter().for_each(|&(start, end)| {
        // SAFETY: `ranges` is built from monotonically increasing group boundaries, so each
        // parallel task writes to a unique, non-overlapping slice of `indices`.
        let out_slice = unsafe {
            std::slice::from_raw_parts_mut((indices_ptr as *mut usize).add(start), end - start)
        };

        out_slice.sort_unstable_by(|&left, &right| {
            timestamps[left]
                .total_cmp(&timestamps[right])
                .then(left.cmp(&right))
        });
    });

    Ok((indices, ranges))
}
