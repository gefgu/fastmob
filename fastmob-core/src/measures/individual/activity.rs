use crate::utils::validate_indexed_ends;

pub fn activity_counts(codes: &[u32], n_activities: usize) -> Result<Vec<u64>, String> {
    let mut counts = vec![0u64; n_activities];
    for &code in codes {
        let idx = usize::try_from(code).map_err(|_| "activity code is out of range")?;
        let count = counts
            .get_mut(idx)
            .ok_or_else(|| "activity code is out of range".to_string())?;
        *count += 1;
    }
    Ok(counts)
}

pub fn activity_transition_counts(
    codes: &[u32],
    indices: &[usize],
    ends: &[usize],
    n_activities: usize,
) -> Result<Vec<u64>, String> {
    validate_indexed_ends(codes.len(), indices, ends)?;
    let mut counts = vec![0u64; n_activities.saturating_mul(n_activities)];
    let mut start = 0usize;
    for &end in ends {
        for pair in indices[start..end].windows(2) {
            let left =
                usize::try_from(codes[pair[0]]).map_err(|_| "activity code is out of range")?;
            let right =
                usize::try_from(codes[pair[1]]).map_err(|_| "activity code is out of range")?;
            if left >= n_activities || right >= n_activities {
                return Err("activity code is out of range".to_string());
            }
            counts[left * n_activities + right] += 1;
        }
        start = end;
    }
    Ok(counts)
}

pub fn daily_activity_percentages(
    codes: &[u32],
    start_minutes: &[i64],
    end_minutes: &[i64],
    valid_rows: &[bool],
    n_activities: usize,
    bin_size_minutes: usize,
) -> Result<Vec<f64>, String> {
    if codes.len() != start_minutes.len()
        || codes.len() != end_minutes.len()
        || codes.len() != valid_rows.len()
    {
        return Err("activity, time, and validity arrays must have equal length".to_string());
    }
    if bin_size_minutes == 0 || 1440 % bin_size_minutes != 0 {
        return Err("bin_size_minutes must be a positive divisor of 1440".to_string());
    }

    let n_bins = 1440 / bin_size_minutes;
    let mut counts = vec![0u64; n_activities.saturating_mul(n_bins)];
    for row in 0..codes.len() {
        if !valid_rows[row] {
            continue;
        }
        let activity = usize::try_from(codes[row]).map_err(|_| "activity code is out of range")?;
        if activity >= n_activities {
            return Err("activity code is out of range".to_string());
        }
        let start_minute = start_minutes[row].clamp(0, 1439) as usize;
        let end_minute = end_minutes[row].clamp(0, 1439) as usize;
        let start_bin = start_minute / bin_size_minutes;
        let end_bin = end_minute / bin_size_minutes;
        let offset = activity * n_bins;
        if end_minute < start_minute {
            for bin in start_bin..n_bins {
                counts[offset + bin] += 1;
            }
            for bin in 0..=end_bin {
                counts[offset + bin] += 1;
            }
        } else {
            for bin in start_bin..=end_bin {
                counts[offset + bin] += 1;
            }
        }
    }

    let mut percentages = vec![f64::NAN; counts.len()];
    for bin in 0..n_bins {
        let total: u64 = (0..n_activities)
            .map(|activity| counts[activity * n_bins + bin])
            .sum();
        if total > 0 {
            for activity in 0..n_activities {
                percentages[activity * n_bins + bin] =
                    counts[activity * n_bins + bin] as f64 / total as f64 * 100.0;
            }
        }
    }
    Ok(percentages)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_categories() {
        assert_eq!(activity_counts(&[0, 1, 0, 2], 3).unwrap(), vec![2, 1, 1]);
    }

    #[test]
    fn counts_transitions_within_ranges() {
        let result = activity_transition_counts(&[0, 1, 0, 1], &[0, 1, 2, 3], &[2, 4], 2).unwrap();
        assert_eq!(result, vec![0, 2, 0, 0]);
    }

    #[test]
    fn daily_activity_handles_overnight_ranges() {
        let result = daily_activity_percentages(&[0], &[1430], &[10], &[true], 1, 10).unwrap();
        assert_eq!(result[143], 100.0);
        assert_eq!(result[0], 100.0);
        assert_eq!(result[1], 100.0);
        assert!(result[2].is_nan());
    }

    #[test]
    fn rejects_out_of_range_activity_codes() {
        assert_eq!(
            activity_counts(&[2], 2).unwrap_err(),
            "activity code is out of range"
        );
    }

    #[test]
    fn rejects_mismatched_daily_buffers() {
        assert_eq!(
            daily_activity_percentages(&[0], &[], &[10], &[true], 1, 10).unwrap_err(),
            "activity, time, and validity arrays must have equal length"
        );
    }
}
