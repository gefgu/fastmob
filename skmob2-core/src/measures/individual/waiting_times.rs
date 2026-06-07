use rayon::prelude::*;

use crate::utils::validate_indexed_ends;

type WaitingTimesResult = Result<(Vec<usize>, Vec<usize>, Vec<f64>), String>;

fn validate_ends(value_len: usize, ends: &[usize]) -> Result<(), String> {
    let mut previous = 0usize;
    for &end in ends {
        if end < previous {
            return Err("range ends must be monotonically non-decreasing".to_string());
        }
        if end > value_len {
            return Err("range end must be within array bounds".to_string());
        }
        previous = end;
    }
    Ok(())
}

pub fn waiting_times_for_range(timestamps_s: &[f64], start: usize, end: usize) -> Vec<f64> {
    if end - start < 2 {
        return Vec::new();
    }
    let mut waits = Vec::with_capacity(end - start - 1);
    for idx in start + 1..end {
        waits.push(timestamps_s[idx] - timestamps_s[idx - 1]);
    }
    waits
}

pub fn waiting_times_for_indexed_range(
    timestamps_s: &[f64],
    indices: &[usize],
    start: usize,
    end: usize,
) -> Vec<f64> {
    if end - start < 2 {
        return Vec::new();
    }
    let mut waits = Vec::with_capacity(end - start - 1);
    for pos in start + 1..end {
        waits.push(timestamps_s[indices[pos]] - timestamps_s[indices[pos - 1]]);
    }
    waits
}

fn value_offsets_from_ends(ends: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let mut starts = Vec::with_capacity(ends.len());
    let mut value_ends = Vec::with_capacity(ends.len());
    let mut row_start = 0usize;
    let mut offset = 0usize;
    for &row_end in ends {
        starts.push(offset);
        offset += row_end.saturating_sub(row_start).saturating_sub(1);
        value_ends.push(offset);
        row_start = row_end;
    }
    (starts, value_ends)
}

pub fn waiting_times_impl(timestamps_s: &[f64], ends: &[usize]) -> WaitingTimesResult {
    validate_ends(timestamps_s.len(), ends)?;

    let (value_starts, value_ends) = value_offsets_from_ends(ends);
    let mut values = vec![0.0; value_ends.last().copied().unwrap_or(0)];
    let mut rest = values.as_mut_slice();
    let mut chunks = Vec::with_capacity(ends.len());
    for (&start, &end) in value_starts.iter().zip(&value_ends) {
        let len = end - start;
        let (chunk, next) = rest.split_at_mut(len);
        chunks.push(chunk);
        rest = next;
    }

    chunks.into_par_iter().enumerate().for_each(|(i, chunk)| {
        let row_start = if i == 0 { 0 } else { ends[i - 1] };
        let row_end = ends[i];
        for (out, idx) in chunk.iter_mut().zip(row_start + 1..row_end) {
            *out = timestamps_s[idx] - timestamps_s[idx - 1];
        }
    });

    Ok((value_starts, value_ends, values))
}

pub fn waiting_times_flat_impl(timestamps_s: &[f64], ends: &[usize]) -> Result<Vec<f64>, String> {
    validate_ends(timestamps_s.len(), ends)?;

    let total_len = value_offsets_from_ends(ends).1.last().copied().unwrap_or(0);
    let mut waits = Vec::with_capacity(total_len);
    let mut start = 0usize;
    for &end in ends {
        for idx in start + 1..end {
            waits.push(timestamps_s[idx] - timestamps_s[idx - 1]);
        }
        start = end;
    }
    Ok(waits)
}

pub fn waiting_times_indexed_impl(
    timestamps_s: &[f64],
    indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
) -> WaitingTimesResult {
    validate_indexed_ends(timestamps_s.len(), indices, ends)?;
    if let Some(valid_rows) = valid_rows {
        if valid_rows.len() != timestamps_s.len() {
            return Err("valid_rows and timestamps must have the same length".to_string());
        }
    }

    if valid_rows.is_none() && indices.iter().all(|&idx| timestamps_s[idx].is_finite()) {
        let (value_starts, value_ends) = value_offsets_from_ends(ends);
        let mut values = vec![0.0; value_ends.last().copied().unwrap_or(0)];
        let mut rest = values.as_mut_slice();
        let mut chunks = Vec::with_capacity(ends.len());
        for (&start, &end) in value_starts.iter().zip(&value_ends) {
            let len = end - start;
            let (chunk, next) = rest.split_at_mut(len);
            chunks.push(chunk);
            rest = next;
        }

        chunks.into_par_iter().enumerate().for_each(|(i, chunk)| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            for (out, pos) in chunk.iter_mut().zip(start + 1..end) {
                *out = timestamps_s[indices[pos]] - timestamps_s[indices[pos - 1]];
            }
        });

        return Ok((value_starts, value_ends, values));
    }

    let grouped_values: Vec<Vec<f64>> = (0..ends.len())
        .into_par_iter()
        .map(|i| {
            let start = if i == 0 { 0 } else { ends[i - 1] };
            let end = ends[i];
            let valid: Vec<usize> = indices[start..end]
                .iter()
                .copied()
                .filter(|&idx| valid_rows.is_none_or(|v| v[idx]) && timestamps_s[idx].is_finite())
                .collect();
            if valid.len() < 2 {
                return Vec::new();
            }
            (1..valid.len())
                .map(|pos| timestamps_s[valid[pos]] - timestamps_s[valid[pos - 1]])
                .collect()
        })
        .collect();

    let mut value_starts = Vec::with_capacity(grouped_values.len());
    let mut value_ends = Vec::with_capacity(grouped_values.len());
    let total_len: usize = grouped_values.iter().map(Vec::len).sum();
    let mut values = Vec::with_capacity(total_len);
    for mut group in grouped_values {
        value_starts.push(values.len());
        values.append(&mut group);
        value_ends.push(values.len());
    }

    Ok((value_starts, value_ends, values))
}

pub fn waiting_times_indexed_flat_impl(
    timestamps_s: &[f64],
    indices: &[usize],
    ends: &[usize],
    valid_rows: Option<&[bool]>,
) -> Result<Vec<f64>, String> {
    validate_indexed_ends(timestamps_s.len(), indices, ends)?;
    if let Some(valid_rows) = valid_rows {
        if valid_rows.len() != timestamps_s.len() {
            return Err("valid_rows and timestamps must have the same length".to_string());
        }
    }

    let mut waits = Vec::new();
    for i in 0..ends.len() {
        let start = if i == 0 { 0 } else { ends[i - 1] };
        let end = ends[i];
        let valid: Vec<usize> = indices[start..end]
            .iter()
            .copied()
            .filter(|&idx| valid_rows.is_none_or(|v| v[idx]) && timestamps_s[idx].is_finite())
            .collect();
        for pos in 1..valid.len() {
            waits.push(timestamps_s[valid[pos]] - timestamps_s[valid[pos - 1]]);
        }
    }
    Ok(waits)
}

pub fn waiting_times_seconds(timestamps_s: Vec<f64>, ends: Vec<usize>) -> WaitingTimesResult {
    waiting_times_impl(&timestamps_s, &ends)
}
