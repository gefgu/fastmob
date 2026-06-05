pub fn validate_coord_ranges(
    latitudes: &[f64],
    longitudes: &[f64],
    ranges: &[(usize, usize)],
) -> Result<(), String> {
    if latitudes.len() != longitudes.len() {
        return Err("latitudes and longitudes must have the same length".to_string());
    }

    let n = latitudes.len();
    for &(start, end) in ranges {
        if start > end {
            return Err("range start must be less than or equal to range end".to_string());
        }
        if end > n {
            return Err("range end must be within coordinate array bounds".to_string());
        }
    }

    Ok(())
}

pub fn validate_ranges(n: usize, ranges: &[(usize, usize)]) -> Result<(), String> {
    for &(start, end) in ranges {
        if start > end {
            return Err("range start must be less than or equal to range end".to_string());
        }
        if end > n {
            return Err("range end must be within array bounds".to_string());
        }
    }
    Ok(())
}

pub fn validate_indexed_ranges(
    value_len: usize,
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> Result<(), String> {
    let n_indices = indices.len();
    for &(start, end) in ranges {
        if start > end {
            return Err("range start must be less than or equal to range end".to_string());
        }
        if end > n_indices {
            return Err("range end must be within index array bounds".to_string());
        }
    }

    for &idx in indices {
        if idx >= value_len {
            return Err("index must be within coordinate array bounds".to_string());
        }
    }

    Ok(())
}

pub fn validate_indexed_ends(
    value_len: usize,
    indices: &[usize],
    ends: &[usize],
) -> Result<(), String> {
    let n_indices = indices.len();
    let mut previous = 0usize;
    for &end in ends {
        if end < previous {
            return Err("range ends must be monotonically non-decreasing".to_string());
        }
        if end > n_indices {
            return Err("range end must be within index array bounds".to_string());
        }
        previous = end;
    }

    for &idx in indices {
        if idx >= value_len {
            return Err("index must be within coordinate array bounds".to_string());
        }
    }

    Ok(())
}

pub fn validate_indexed_coord_ranges(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> Result<(), String> {
    if latitudes.len() != longitudes.len() {
        return Err("latitudes and longitudes must have the same length".to_string());
    }
    validate_indexed_ranges(latitudes.len(), indices, ranges)
}

pub fn validate_indexed_coord_ends(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ends: &[usize],
) -> Result<(), String> {
    if latitudes.len() != longitudes.len() {
        return Err("latitudes and longitudes must have the same length".to_string());
    }
    validate_indexed_ends(latitudes.len(), indices, ends)
}

pub fn ranges_from_starts_ends(
    starts: &[usize],
    ends: &[usize],
) -> Result<Vec<(usize, usize)>, String> {
    if starts.len() != ends.len() {
        return Err("range starts and ends must have the same length".to_string());
    }

    starts
        .iter()
        .zip(ends)
        .map(|(&start, &end)| {
            if start > end {
                Err("range start must be less than or equal to range end".to_string())
            } else {
                Ok((start, end))
            }
        })
        .collect()
}

pub fn ranges_from_ends(ends: &[usize]) -> Result<Vec<(usize, usize)>, String> {
    let mut ranges = Vec::with_capacity(ends.len());
    let mut start = 0usize;
    for &end in ends {
        if end < start {
            return Err("range ends must be monotonically non-decreasing".to_string());
        }
        ranges.push((start, end));
        start = end;
    }
    Ok(ranges)
}

pub fn ends_from_ranges(ranges: &[(usize, usize)]) -> Vec<usize> {
    ranges.iter().map(|&(_, end)| end).collect()
}

/// Splits a `Vec<(usize, usize)>` of ranges into two parallel `Vec<usize>` of starts and ends.
pub fn split_ranges(ranges: Vec<(usize, usize)>) -> (Vec<usize>, Vec<usize>) {
    ranges.into_iter().unzip()
}

/// Extracts each element of a slice as `Option<T>`, mapping NaN f64 values to `None`.
pub fn f64_option_values(values: &[f64]) -> Vec<Option<f64>> {
    values
        .iter()
        .map(|&v| if v.is_nan() { None } else { Some(v) })
        .collect()
}

/// Builds contiguous `(start, end)` ranges from a slice of values that has been sorted by an
/// index permutation, grouping consecutive equal values.
pub fn ranges_from_sorted_values<T: PartialEq>(
    values: &[T],
    indices: &[usize],
) -> Vec<(usize, usize)> {
    if indices.is_empty() {
        return Vec::new();
    }

    let mut ranges = Vec::new();
    let mut start = 0usize;
    for pos in 1..indices.len() {
        if values[indices[pos]] != values[indices[pos - 1]] {
            ranges.push((start, pos));
            start = pos;
        }
    }
    ranges.push((start, indices.len()));
    ranges
}

/// Computes the median of a mutable slice in-place using quickselect (O(n) average).
pub fn median_slice_in_place(v: &mut [f64]) -> f64 {
    let n = v.len();
    if n == 0 {
        return f64::NAN;
    }
    let mid = n / 2;
    v.select_nth_unstable_by(mid, |a, b| a.total_cmp(b));
    if n.is_multiple_of(2) {
        let left_max = v[..mid].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (left_max + v[mid]) / 2.0
    } else {
        v[mid]
    }
}

pub fn validate_uid_len(n: usize, uid_len: usize) -> Result<(), String> {
    if n != uid_len {
        return Err(
            "uids, latitudes, longitudes, and timestamps must have the same length".to_string(),
        );
    }
    Ok(())
}
