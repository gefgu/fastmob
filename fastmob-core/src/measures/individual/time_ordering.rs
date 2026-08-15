use rayon::prelude::*;

pub type IndexRanges = Vec<(u64, u64)>;
pub type OrderedIndexRanges = (Vec<u64>, IndexRanges);

pub fn split_ordered_index_ranges((indices, ranges): OrderedIndexRanges) -> (Vec<u64>, Vec<u64>) {
    (indices, ranges.into_iter().map(|(_, end)| end).collect())
}

pub fn presorted_ranges_for_u32_codes(codes: &[u32]) -> (Vec<u64>, Vec<u64>) {
    let n = codes.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }

    let boundaries: Vec<u64> = codes
        .par_windows(2)
        .enumerate()
        .filter_map(|(idx, window)| (window[0] != window[1]).then_some((idx + 1) as u64))
        .collect();

    let mut starts = Vec::with_capacity(boundaries.len() + 1);
    starts.push(0);
    starts.extend_from_slice(&boundaries);

    let mut ends = boundaries;
    ends.push(n as u64);
    (starts, ends)
}

pub fn time_ordered_indices_single_user(timestamps: &[i64]) -> OrderedIndexRanges {
    let mut indices: Vec<u64> = (0..timestamps.len()).map(|value| value as u64).collect();
    indices.par_sort_by(|&left, &right| {
        timestamps[left as usize]
            .cmp(&timestamps[right as usize])
            .then(left.cmp(&right))
    });
    let n = indices.len();
    if n == 0 {
        return (indices, Vec::new());
    }
    (indices, vec![(0, n as u64)])
}

pub fn time_ordered_indices_for_u32_codes(
    codes: &[u32],
    timestamps: &[i64],
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
        ranges.push((start as u64, end as u64));
        *count = start;
        current_offset = end;
    }

    let mut indices = vec![0u64; n];
    for (row_idx, &code) in codes.iter().enumerate() {
        let group_id = code as usize;
        let pos = offsets[group_id];
        indices[pos] = row_idx as u64;
        offsets[group_id] += 1;
    }

    let indices_ptr = indices.as_mut_ptr() as usize;
    ranges.par_iter().for_each(|&(start, end)| {
        // SAFETY: `ranges` is built from monotonically increasing group boundaries, so each
        // parallel task writes to a unique, non-overlapping slice of `indices`.
        let out_slice = unsafe {
            std::slice::from_raw_parts_mut(
                (indices_ptr as *mut u64).add(start as usize),
                (end - start) as usize,
            )
        };

        out_slice.sort_unstable_by(|&left, &right| {
            timestamps[left as usize]
                .cmp(&timestamps[right as usize])
                .then(left.cmp(&right))
        });
    });

    Ok((indices, ranges))
}
