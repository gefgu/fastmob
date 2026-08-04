//! Histogram of distinct locations visited by each user on each calendar day.

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

/// Return sorted `(daily_distinct_location_count, frequency)` pairs.
///
/// `valid` excludes rows with a null user, location, or timestamp. `days`
/// contains local calendar-day buckets, computed at the Arrow boundary after
/// timezone metadata has been stripped.
pub fn daily_unique_location_histogram_impl(
    user_codes: &[u64],
    location_codes: &[u64],
    days: &[i64],
    valid: &[bool],
) -> Result<(Vec<u64>, Vec<u64>), String> {
    let len = user_codes.len();
    if location_codes.len() != len || days.len() != len || valid.len() != len {
        return Err("all daily-location columns must have the same length".to_string());
    }

    let groups = (0..len)
        .into_par_iter()
        .fold(FxHashMap::default, |mut groups, index| {
            if valid[index] {
                groups
                    .entry((user_codes[index], days[index]))
                    .or_insert_with(FxHashSet::default)
                    .insert(location_codes[index]);
            }
            groups
        })
        .reduce(FxHashMap::default, |mut left, right| {
            for (key, locations) in right {
                left.entry(key)
                    .or_insert_with(FxHashSet::default)
                    .extend(locations);
            }
            left
        });

    let mut histogram: FxHashMap<u64, u64> = FxHashMap::default();
    for locations in groups.values() {
        *histogram.entry(locations.len() as u64).or_default() += 1;
    }
    let mut pairs: Vec<(u64, u64)> = histogram.into_iter().collect();
    pairs.sort_unstable_by_key(|(count, _)| *count);
    Ok(pairs.into_iter().unzip())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_distinct_locations_per_user_day() {
        let (counts, frequencies) = daily_unique_location_histogram_impl(
            &[0, 0, 0, 1, 1, 2],
            &[0, 1, 1, 0, 2, 4],
            &[10, 10, 11, 10, 10, 12],
            &[true, true, true, true, true, false],
        )
        .unwrap();
        assert_eq!(counts, [1, 2]);
        assert_eq!(frequencies, [1, 2]);
    }
}
