//! Sparse aggregation for spatio-temporal volume distributions.

use chrono::{DateTime, Datelike, Timelike, Utc};
use rustc_hash::{FxHashMap, FxHashSet};

const TEN_MINUTES_US: i64 = 10 * 60 * 1_000_000;
const MINUTES_PER_DAY: u16 = 24 * 60;

/// Aggregate coded stays into `(area_code, ten-minute-bin, mean_volume)` rows.
///
/// Timestamps are Unix microseconds. Null timestamps are represented by `None`;
/// invalid (end-before-start) stays are ignored, matching the Python reference.
pub fn mean_area_volume_impl(
    area_codes: &[Option<u32>],
    user_codes: &[Option<u32>],
    starts_us: &[Option<i64>],
    ends_us: &[Option<i64>],
) -> Result<Vec<(u32, u16, f64)>, String> {
    let n = area_codes.len();
    if user_codes.len() != n || starts_us.len() != n || ends_us.len() != n {
        return Err("area, user, start, and end arrays must have the same length".to_string());
    }

    // (area, calendar-day, weekday, minutes-after-midnight) -> distinct users
    let mut presence: FxHashMap<(u32, i32, u8, u16), FxHashSet<u32>> = FxHashMap::default();
    for index in 0..n {
        let (Some(area), Some(user), Some(start), Some(end)) = (
            area_codes[index],
            user_codes[index],
            starts_us[index],
            ends_us[index],
        ) else {
            continue;
        };
        let mut slot = start.div_euclid(TEN_MINUTES_US) * TEN_MINUTES_US;
        let end_slot = end.div_euclid(TEN_MINUTES_US) * TEN_MINUTES_US;
        while slot <= end_slot {
            let seconds = slot.div_euclid(1_000_000);
            let micros = slot.rem_euclid(1_000_000) as u32;
            let dt = DateTime::<Utc>::from_timestamp(seconds, micros * 1_000)
                .ok_or_else(|| "timestamp is outside the supported date range".to_string())?;
            let date = dt.date_naive();
            let minute = (dt.hour() * 60 + dt.minute()) as u16;
            debug_assert!(minute < MINUTES_PER_DAY);
            presence
                .entry((
                    area,
                    date.num_days_from_ce(),
                    dt.weekday().num_days_from_monday() as u8,
                    minute,
                ))
                .or_default()
                .insert(user);
            slot = slot
                .checked_add(TEN_MINUTES_US)
                .ok_or_else(|| "stay timestamp overflows supported range".to_string())?;
        }
    }

    let mut dates_per_area_dow: FxHashMap<(u32, u8), FxHashSet<i32>> = FxHashMap::default();
    let mut dow_sum: FxHashMap<(u32, u8, u16), f64> = FxHashMap::default();
    for ((area, date, dow, minute), users) in presence {
        dates_per_area_dow
            .entry((area, dow))
            .or_default()
            .insert(date);
        *dow_sum.entry((area, dow, minute)).or_insert(0.0) += users.len() as f64;
    }

    let mut area_bin_sum: FxHashMap<(u32, u16), f64> = FxHashMap::default();
    for ((area, dow, minute), total) in dow_sum {
        let dates = dates_per_area_dow[&(area, dow)].len() as f64;
        *area_bin_sum.entry((area, minute)).or_insert(0.0) += total / dates;
    }

    let mut result: Vec<_> = area_bin_sum
        .into_iter()
        .filter_map(|((area, minute), total)| {
            let mean = total / 7.0;
            (mean > 0.0).then_some((area, minute, mean))
        })
        .collect();
    result.sort_unstable_by_key(|&(area, minute, _)| (area, minute));
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn averages_distinct_users_over_observed_dates_and_weekdays() {
        // Monday 2020-01-06 08:00 and Monday 2020-01-13 08:00.
        let first = 1_578_297_600_000_000_i64;
        let second = first + 7 * 24 * 60 * 60 * 1_000_000;
        let rows = mean_area_volume_impl(
            &[Some(4), Some(4), Some(4)],
            &[Some(8), Some(9), Some(8)],
            &[Some(first), Some(first), Some(second)],
            &[Some(first), Some(first), Some(second)],
        )
        .unwrap();
        assert_eq!(rows, vec![(4, 480, 1.5 / 7.0)]);
    }

    #[test]
    fn expands_across_midnight_and_skips_invalid_stays() {
        let start = 1_578_355_400_000_000_i64; // 2020-01-06 23:50 UTC
        let end = start + 30 * 60 * 1_000_000;
        let rows = mean_area_volume_impl(
            &[Some(1), Some(1)],
            &[Some(2), Some(2)],
            &[Some(start), Some(end)],
            &[Some(end), Some(start)],
        )
        .unwrap();
        assert_eq!(rows.len(), 4);
        assert!(rows
            .iter()
            .all(|&(area, _, mean)| area == 1 && mean == 1.0 / 7.0));
    }
}
