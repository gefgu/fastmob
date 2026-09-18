//! Sparse aggregation for spatio-temporal volume distributions.

use chrono::{DateTime, Datelike, Timelike, Utc};
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

const TEN_MINUTES_US: i64 = 10 * 60 * 1_000_000;
const MINUTES_PER_DAY: u16 = 24 * 60;

/// One expanded 10-minute presence slot: a user seen in an area on a given
/// calendar date (day-of-week is redundant with date but kept alongside it
/// so the sort groups every downstream consumer needs are already adjacent).
type PresenceEntry = (u32 /* area */, i32 /* date */, u8 /* dow */, u16 /* minute */, u32 /* user */);

fn expand_row(
    area: u32,
    user: u32,
    start: i64,
    end: i64,
    out: &mut Vec<PresenceEntry>,
) -> Result<(), String> {
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
        out.push((
            area,
            date.num_days_from_ce(),
            dt.weekday().num_days_from_monday() as u8,
            minute,
            user,
        ));
        slot = slot
            .checked_add(TEN_MINUTES_US)
            .ok_or_else(|| "stay timestamp overflows supported range".to_string())?;
    }
    Ok(())
}

/// Aggregate coded stays into `(area_code, ten-minute-bin, mean_volume)` rows.
///
/// `area_codes`/`user_codes`/`starts_us`/`ends_us` are dense value buffers (Unix
/// microseconds for the timestamps); `valid_rows[i] == false` (or an out-of-range
/// row when the option is `None` is never invalid -- every row is valid) excludes
/// row `i`, matching the Arrow-nullability convention used across the crate. A
/// stay with `start > end` is skipped (produces zero bins), matching the Python
/// reference this replaced.
///
/// Expansion into per-row presence slots and the union-by-sort distinct-user
/// count are both parallelized: expansion is an embarrassingly parallel
/// per-row map, and deduplicating repeated `(area, date, dow, minute, user)`
/// entries only needs adjacent duplicates to collide, which a sort guarantees
/// without ever materializing a per-slot hash set.
pub fn mean_area_volume_impl(
    area_codes: &[u32],
    user_codes: &[u32],
    starts_us: &[i64],
    ends_us: &[i64],
    valid_rows: Option<&[bool]>,
) -> Result<Vec<(u32, u16, f64)>, String> {
    let n = area_codes.len();
    if user_codes.len() != n || starts_us.len() != n || ends_us.len() != n {
        return Err("area, user, start, and end arrays must have the same length".to_string());
    }
    if let Some(valid) = valid_rows
        && valid.len() != n
    {
        return Err("valid_rows must have the same length as the other arrays".to_string());
    }

    let mut entries: Vec<PresenceEntry> = (0..n)
        .into_par_iter()
        .filter(|&index| valid_rows.is_none_or(|v| v[index]))
        .try_fold(Vec::new, |mut local, index| -> Result<Vec<PresenceEntry>, String> {
            expand_row(area_codes[index], user_codes[index], starts_us[index], ends_us[index], &mut local)?;
            Ok(local)
        })
        .try_reduce(Vec::new, |mut a, mut b| {
            a.append(&mut b);
            Ok(a)
        })?;

    entries.par_sort_unstable();

    // Collapse sorted (area, date, dow, minute, user) entries into distinct-user
    // counts per slot. Because the sort's last key is `user`, repeats of the same
    // user within a slot are adjacent, so a single linear scan reproduces exactly
    // what `HashSet::len()` would have counted -- without ever allocating one.
    let mut presence_counts: Vec<(u32, i32, u8, u16, f64)> = Vec::new();
    let mut iter = entries.into_iter();
    if let Some((mut cur_area, mut cur_date, mut cur_dow, mut cur_minute, mut last_user)) = iter.next() {
        let mut count: u32 = 1;
        for (area, date, dow, minute, user) in iter {
            if area == cur_area && date == cur_date && dow == cur_dow && minute == cur_minute {
                if user != last_user {
                    count += 1;
                    last_user = user;
                }
            } else {
                presence_counts.push((cur_area, cur_date, cur_dow, cur_minute, count as f64));
                cur_area = area;
                cur_date = date;
                cur_dow = dow;
                cur_minute = minute;
                last_user = user;
                count = 1;
            }
        }
        presence_counts.push((cur_area, cur_date, cur_dow, cur_minute, count as f64));
    }

    // From here on the working set is bounded by distinct (area, dow[, date],
    // minute) combinations actually observed, not by raw input rows, so plain
    // hash maps are the right tool again.
    let mut dates_per_area_dow: FxHashMap<(u32, u8), FxHashSet<i32>> = FxHashMap::default();
    let mut dow_sum: FxHashMap<(u32, u8, u16), f64> = FxHashMap::default();
    for (area, date, dow, minute, count) in presence_counts {
        dates_per_area_dow.entry((area, dow)).or_default().insert(date);
        *dow_sum.entry((area, dow, minute)).or_insert(0.0) += count;
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

    fn run(
        areas: &[u32],
        users: &[u32],
        starts: &[i64],
        ends: &[i64],
    ) -> Result<Vec<(u32, u16, f64)>, String> {
        mean_area_volume_impl(areas, users, starts, ends, None)
    }

    #[test]
    fn averages_distinct_users_over_observed_dates_and_weekdays() {
        // Monday 2020-01-06 08:00 and Monday 2020-01-13 08:00.
        let first = 1_578_297_600_000_000_i64;
        let second = first + 7 * 24 * 60 * 60 * 1_000_000;
        let rows = run(&[4, 4, 4], &[8, 9, 8], &[first, first, second], &[first, first, second]).unwrap();
        assert_eq!(rows, vec![(4, 480, 1.5 / 7.0)]);
    }

    #[test]
    fn expands_across_midnight_and_skips_invalid_stays() {
        let start = 1_578_355_400_000_000_i64; // 2020-01-06 23:50 UTC
        let end = start + 30 * 60 * 1_000_000;
        let rows = run(&[1, 1], &[2, 2], &[start, end], &[end, start]).unwrap();
        assert_eq!(rows.len(), 4);
        assert!(rows.iter().all(|&(area, _, mean)| area == 1 && mean == 1.0 / 7.0));
    }

    #[test]
    fn invalid_rows_are_excluded() {
        let first = 1_578_297_600_000_000_i64;
        let rows = mean_area_volume_impl(
            &[4, 4],
            &[8, 9],
            &[first, first],
            &[first, first],
            Some(&[true, false]),
        )
        .unwrap();
        assert_eq!(rows, vec![(4, 480, 1.0 / 7.0)]);
    }

    #[test]
    fn duplicate_overlapping_stays_for_the_same_user_do_not_double_count() {
        // Two rows for the same user covering the same slot must still count once.
        let first = 1_578_297_600_000_000_i64;
        let rows = run(&[4, 4], &[8, 8], &[first, first], &[first, first]).unwrap();
        assert_eq!(rows, vec![(4, 480, 1.0 / 7.0)]);
    }

    #[test]
    fn mismatched_lengths_are_rejected() {
        assert!(run(&[1, 2], &[1], &[1, 2], &[1, 2]).is_err());
    }

    #[test]
    fn mismatched_valid_rows_length_is_rejected() {
        assert!(mean_area_volume_impl(&[1], &[1], &[1], &[1], Some(&[true, false])).is_err());
    }

    /// Brute-force reference mirroring the pre-rewrite implementation: a
    /// `HashMap<key, HashSet<user>>` reference structure, kept solely to
    /// differentially test the sort-based rewrite above.
    fn brute_force_reference(
        areas: &[u32],
        users: &[u32],
        starts: &[i64],
        ends: &[i64],
    ) -> Vec<(u32, u16, f64)> {
        use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};

        let mut presence: HashMap<(u32, i32, u8, u16), HashSet<u32>> = HashMap::default();
        for i in 0..areas.len() {
            let mut slot = starts[i].div_euclid(TEN_MINUTES_US) * TEN_MINUTES_US;
            let end_slot = ends[i].div_euclid(TEN_MINUTES_US) * TEN_MINUTES_US;
            while slot <= end_slot {
                let seconds = slot.div_euclid(1_000_000);
                let micros = slot.rem_euclid(1_000_000) as u32;
                let dt = DateTime::<Utc>::from_timestamp(seconds, micros * 1_000).unwrap();
                let date = dt.date_naive();
                let minute = (dt.hour() * 60 + dt.minute()) as u16;
                presence
                    .entry((areas[i], date.num_days_from_ce(), dt.weekday().num_days_from_monday() as u8, minute))
                    .or_default()
                    .insert(users[i]);
                slot += TEN_MINUTES_US;
            }
        }

        let mut dates_per_area_dow: HashMap<(u32, u8), HashSet<i32>> = HashMap::default();
        let mut dow_sum: HashMap<(u32, u8, u16), f64> = HashMap::default();
        for ((area, date, dow, minute), users_set) in presence {
            dates_per_area_dow.entry((area, dow)).or_default().insert(date);
            *dow_sum.entry((area, dow, minute)).or_insert(0.0) += users_set.len() as f64;
        }

        let mut area_bin_sum: HashMap<(u32, u16), f64> = HashMap::default();
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
        result
    }

    /// Deterministic xorshift PRNG -- avoids pulling in a `rand` dev-dependency
    /// just for this one differential test.
    struct XorShift(u64);
    impl XorShift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn range(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    #[test]
    fn matches_brute_force_reference_on_random_inputs() {
        let mut rng = XorShift(0x2545F4914F6CDD1D);
        let base = 1_577_836_800_000_000_i64; // 2020-01-01 00:00 UTC

        for trial in 0..200 {
            let n = 1 + (rng.range(40) as usize);
            let mut areas = Vec::with_capacity(n);
            let mut users = Vec::with_capacity(n);
            let mut starts = Vec::with_capacity(n);
            let mut ends = Vec::with_capacity(n);
            for _ in 0..n {
                areas.push(rng.range(3) as u32);
                users.push(rng.range(4) as u32);
                let start = base + (rng.range(20) as i64) * 24 * 60 * 60 * 1_000_000
                    + (rng.range(144) as i64) * TEN_MINUTES_US;
                // Occasionally invert start/end to exercise the "invalid stay" path too.
                let duration_bins = rng.range(6) as i64;
                let (start, end) = if rng.range(10) == 0 {
                    (start, start - TEN_MINUTES_US)
                } else {
                    (start, start + duration_bins * TEN_MINUTES_US)
                };
                starts.push(start);
                ends.push(end);
            }

            let expected = brute_force_reference(&areas, &users, &starts, &ends);
            let actual = run(&areas, &users, &starts, &ends).unwrap();
            // Both sides sum per-(area,dow) contributions in hash-map iteration
            // order, which differs between the two implementations, so floating
            // point summation order (and therefore rounding) can differ too --
            // compare keys exactly and means within an epsilon instead of bit-exact.
            assert_eq!(
                actual.len(),
                expected.len(),
                "trial {trial} row count diverged for areas={areas:?} users={users:?} starts={starts:?} ends={ends:?}"
            );
            for ((a_area, a_minute, a_mean), (e_area, e_minute, e_mean)) in actual.iter().zip(expected.iter()) {
                assert_eq!((a_area, a_minute), (e_area, e_minute), "trial {trial} key diverged");
                assert!(
                    (a_mean - e_mean).abs() < 1e-9,
                    "trial {trial} mean diverged: {a_mean} vs {e_mean} at area={a_area} minute={a_minute}"
                );
            }
        }
    }
}
