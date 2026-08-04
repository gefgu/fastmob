//! The universal visitation law: aggregating raw staypoints into per-user,
//! per-cell `(r, f)` observations, binning those into `(rf, rho)` pairs, and
//! fitting `rho(r, f) = mu * (rf)^-eta`.

use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

type VisitationCellAggregate = (Vec<u64>, Vec<u64>, Vec<u64>, Vec<f64>, Vec<f64>, Vec<f64>);

struct CellAccumulator {
    count: u64,
    days: FxHashSet<i64>,
    lat: f64,
    lng: f64,
}

/// Aggregate raw per-row staypoints into one row per `(user, h3 cell)` pair.
///
/// Returns `(user_codes, h3_cells, n_staypoints, f, loc_lat, loc_lng)`, sorted
/// by `(user_code, h3_cell)` so the result does not depend on hash iteration
/// order. `f` is the count of distinct values in `days` (typically calendar-day
/// buckets) seen at that `(user, cell)` pair. `loc_lat`/`loc_lng` are the
/// coordinates of the first row observed for that pair, matching the
/// Narwhals `.first()` aggregation this replaces. Rows with a non-finite
/// latitude or longitude are skipped.
pub fn aggregate_visitation_cells_impl(
    user_codes: &[u64],
    h3_cells: &[u64],
    days: &[i64],
    lats: &[f64],
    lngs: &[f64],
) -> Result<VisitationCellAggregate, String> {
    let len = user_codes.len();
    if h3_cells.len() != len || days.len() != len || lats.len() != len || lngs.len() != len {
        return Err("all visitation-cell columns must have the same length".to_string());
    }

    let groups = (0..len)
        .into_par_iter()
        .fold(
            FxHashMap::default,
            |mut groups: FxHashMap<(u64, u64), CellAccumulator>, index| {
                let (lat, lng) = (lats[index], lngs[index]);
                if !lat.is_finite() || !lng.is_finite() {
                    return groups;
                }
                let entry = groups
                    .entry((user_codes[index], h3_cells[index]))
                    .or_insert_with(|| CellAccumulator {
                        count: 0,
                        days: FxHashSet::default(),
                        lat,
                        lng,
                    });
                entry.count += 1;
                entry.days.insert(days[index]);
                groups
            },
        )
        .reduce(FxHashMap::default, |mut left, right| {
            for (key, right_accum) in right {
                match left.entry(key) {
                    std::collections::hash_map::Entry::Occupied(mut existing) => {
                        let left_accum = existing.get_mut();
                        left_accum.count += right_accum.count;
                        left_accum.days.extend(right_accum.days);
                    }
                    std::collections::hash_map::Entry::Vacant(slot) => {
                        slot.insert(right_accum);
                    }
                }
            }
            left
        });

    let mut rows: Vec<((u64, u64), CellAccumulator)> = groups.into_iter().collect();
    rows.sort_unstable_by_key(|(key, _)| *key);

    let mut out_user = Vec::with_capacity(rows.len());
    let mut out_cell = Vec::with_capacity(rows.len());
    let mut out_count = Vec::with_capacity(rows.len());
    let mut out_f = Vec::with_capacity(rows.len());
    let mut out_lat = Vec::with_capacity(rows.len());
    let mut out_lng = Vec::with_capacity(rows.len());
    for ((user_code, cell), accum) in rows {
        out_user.push(user_code);
        out_cell.push(cell);
        out_count.push(accum.count);
        out_f.push(accum.days.len() as f64);
        out_lat.push(accum.lat);
        out_lng.push(accum.lng);
    }
    Ok((out_user, out_cell, out_count, out_f, out_lat, out_lng))
}

/// Aggregate visitation-law rows into logarithmically binned `(rf, rho)` pairs.
///
/// Each `(location, radial bin, frequency)` triple contributes one density
/// observation: the number of *distinct* users seen there, divided by the area
/// of the annulus at that radius. Those observations are then averaged into
/// `n_bins` logarithmic bins of `rf`.
///
/// Rows with non-positive or non-finite `r_km`/`f` are skipped, matching the
/// Python implementation.
pub fn bin_visitation_law_impl(
    user_codes: &[u64],
    location_codes: &[u64],
    r_km: &[f64],
    f: &[f64],
    n_bins: usize,
    distance_bin_width_km: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    if distance_bin_width_km <= 0.0 || distance_bin_width_km.is_nan() {
        return Err("distance_bin_width_km must be positive".to_string());
    }
    if n_bins == 0 {
        return Err("n_bins must be positive".to_string());
    }
    let len = user_codes.len();
    if location_codes.len() != len || r_km.len() != len || f.len() != len {
        return Err("all visitation-law columns must have the same length".to_string());
    }

    // Key on the bit patterns so the radial-bin centre and frequency compare
    // exactly; they are derived deterministically from the inputs.
    let mut groups: FxHashMap<(u64, u64, u64), FxHashSet<u64>> = FxHashMap::default();
    for index in 0..len {
        let (radius, frequency) = (r_km[index], f[index]);
        if !(radius.is_finite() && radius > 0.0 && frequency.is_finite() && frequency > 0.0) {
            continue;
        }
        let r_center = (radius / distance_bin_width_km).floor() * distance_bin_width_km
            + distance_bin_width_km / 2.0;
        groups
            .entry((
                location_codes[index],
                r_center.to_bits(),
                frequency.to_bits(),
            ))
            .or_default()
            .insert(user_codes[index]);
    }

    // Sort before aggregating so the result does not depend on hash iteration
    // order: bin means are floating-point sums, whose last bits otherwise vary.
    let mut observations: Vec<(f64, f64)> = groups
        .iter()
        .filter_map(|((_, r_bits, f_bits), users)| {
            let r_center = f64::from_bits(*r_bits);
            let frequency = f64::from_bits(*f_bits);
            let annulus_area = 2.0 * std::f64::consts::PI * r_center * distance_bin_width_km;
            if annulus_area <= 0.0 {
                return None;
            }
            let rho = users.len() as f64 / annulus_area;
            let rf = r_center * frequency;
            (rf > 0.0 && rho > 0.0).then_some((rf, rho))
        })
        .collect();
    if observations.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }
    observations.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));

    let rf_min = observations
        .iter()
        .map(|(rf, _)| *rf)
        .fold(f64::INFINITY, f64::min);
    let rf_max = observations
        .iter()
        .map(|(rf, _)| *rf)
        .fold(f64::NEG_INFINITY, f64::max);
    if rf_min == rf_max {
        let mean = observations.iter().map(|(_, rho)| rho).sum::<f64>() / observations.len() as f64;
        return Ok((vec![rf_min], vec![mean]));
    }

    let log_min = rf_min.log10();
    let log_max = rf_max.log10();
    let edges: Vec<f64> = (0..=n_bins)
        .map(|index| 10f64.powf(log_min + (log_max - log_min) * index as f64 / n_bins as f64))
        .collect();

    let mut rf_out = Vec::new();
    let mut rho_out = Vec::new();
    for index in 0..n_bins {
        let (low, high) = (edges[index], edges[index + 1]);
        let is_last = index == n_bins - 1;
        let bucket: Vec<f64> = observations
            .iter()
            .filter(|(rf, _)| {
                if is_last {
                    *rf >= low && *rf <= high
                } else {
                    *rf >= low && *rf < high
                }
            })
            .map(|(_, rho)| *rho)
            .collect();
        if bucket.is_empty() {
            continue;
        }
        rf_out.push((low * high).sqrt());
        rho_out.push(bucket.iter().sum::<f64>() / bucket.len() as f64);
    }
    Ok((rf_out, rho_out))
}

/// Fit `rho(r, f) = mu * (rf)^-eta` by ordinary least squares in log-log space.
///
/// Returns `(eta, mu, r2)`, with `r2` measured in log-log space. `min_rf` and
/// `max_rf` optionally restrict the inclusive fitting range.
///
/// This uses the closed-form normal equations; the Python implementation calls
/// `np.polyfit`, which solves the same least-squares problem by a different
/// route. The two agree to roughly 1e-15 relative rather than bit-for-bit -- on
/// an exact power law this returns `mu = 5.0` where `np.polyfit` returns
/// `4.999999999999998`.
pub fn fit_visitation_law_impl(
    rf_values: &[f64],
    rho_values: &[f64],
    min_rf: Option<f64>,
    max_rf: Option<f64>,
) -> Result<(f64, f64, f64), String> {
    if rf_values.len() != rho_values.len() {
        return Err("rf_values and rho_values must have the same length".to_string());
    }
    if let Some(min_rf) = min_rf
        && min_rf <= 0.0
    {
        return Err("min_rf must be positive when provided".to_string());
    }
    if let Some(max_rf) = max_rf
        && max_rf <= 0.0
    {
        return Err("max_rf must be positive when provided".to_string());
    }

    let points: Vec<(f64, f64)> = rf_values
        .iter()
        .zip(rho_values)
        .filter(|(rf, rho)| {
            rf.is_finite()
                && rho.is_finite()
                && **rf > 0.0
                && **rho > 0.0
                && min_rf.is_none_or(|bound| **rf >= bound)
                && max_rf.is_none_or(|bound| **rf <= bound)
        })
        .map(|(rf, rho)| (rf.ln(), rho.ln()))
        .collect();
    if points.len() < 2 {
        return Err("At least two positive finite data points are required to fit.".to_string());
    }

    let n = points.len() as f64;
    let x_mean = points.iter().map(|(x, _)| x).sum::<f64>() / n;
    let y_mean = points.iter().map(|(_, y)| y).sum::<f64>() / n;
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for (x, y) in &points {
        sxy += (x - x_mean) * (y - y_mean);
        sxx += (x - x_mean).powi(2);
    }
    if sxx == 0.0 {
        return Err("rf values must not all be identical to fit a slope.".to_string());
    }

    let slope = sxy / sxx;
    let intercept = y_mean - slope * x_mean;
    let ss_res: f64 = points
        .iter()
        .map(|(x, y)| (y - (intercept + slope * x)).powi(2))
        .sum();
    let ss_tot: f64 = points.iter().map(|(_, y)| (y - y_mean).powi(2)).sum();
    let r2 = if ss_tot == 0.0 {
        1.0
    } else {
        1.0 - ss_res / ss_tot
    };

    Ok((-slope, intercept.exp(), r2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregates_count_distinct_days_and_keep_first_coordinates() {
        // user 1 visits cell 7 twice on day 10 and once on day 11: n_staypoints=3, f=2.
        let (users, cells, counts, f, lat, lng) = aggregate_visitation_cells_impl(
            &[1, 1, 1],
            &[7, 7, 7],
            &[10, 10, 11],
            &[1.0, 2.0, 3.0],
            &[10.0, 20.0, 30.0],
        )
        .unwrap();
        assert_eq!(users, [1]);
        assert_eq!(cells, [7]);
        assert_eq!(counts, [3]);
        assert_eq!(f, [2.0]);
        assert_eq!(lat, [1.0], "first row's coordinates are kept");
        assert_eq!(lng, [10.0]);
    }

    #[test]
    fn distinct_user_cell_pairs_form_separate_groups() {
        let (users, cells, counts, f, ..) = aggregate_visitation_cells_impl(
            &[1, 1, 2],
            &[7, 8, 7],
            &[10, 10, 10],
            &[1.0, 1.0, 1.0],
            &[1.0, 1.0, 1.0],
        )
        .unwrap();
        assert_eq!(users, [1, 1, 2]);
        assert_eq!(cells, [7, 8, 7]);
        assert_eq!(counts, [1, 1, 1]);
        assert_eq!(f, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn output_is_sorted_regardless_of_input_order() {
        let forward = aggregate_visitation_cells_impl(
            &[2, 1, 1],
            &[3, 9, 5],
            &[1, 1, 1],
            &[1.0, 2.0, 3.0],
            &[1.0, 2.0, 3.0],
        )
        .unwrap();
        let backward = aggregate_visitation_cells_impl(
            &[1, 1, 2],
            &[5, 9, 3],
            &[1, 1, 1],
            &[3.0, 2.0, 1.0],
            &[3.0, 2.0, 1.0],
        )
        .unwrap();
        assert_eq!(forward.0, backward.0, "user codes");
        assert_eq!(forward.1, backward.1, "h3 cells");
    }

    #[test]
    fn non_finite_coordinates_are_skipped() {
        let (users, ..) = aggregate_visitation_cells_impl(
            &[1, 2],
            &[7, 8],
            &[10, 10],
            &[f64::NAN, 1.0],
            &[1.0, f64::INFINITY],
        )
        .unwrap();
        assert!(users.is_empty(), "both rows have a non-finite coordinate");
    }

    #[test]
    fn mismatched_lengths_are_rejected() {
        assert!(
            aggregate_visitation_cells_impl(&[1, 2], &[1], &[1, 1], &[1.0, 1.0], &[1.0, 1.0])
                .is_err()
        );
    }

    #[test]
    fn a_perfect_power_law_is_recovered_exactly() {
        // rho = 5 * rf^-2
        let rf: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let rho: Vec<f64> = rf.iter().map(|x| 5.0 * x.powf(-2.0)).collect();
        let (eta, mu, r2) = fit_visitation_law_impl(&rf, &rho, None, None).unwrap();
        assert!((eta - 2.0).abs() < 1e-12, "eta {eta}");
        assert!((mu - 5.0).abs() < 1e-12, "mu {mu}");
        assert!((r2 - 1.0).abs() < 1e-12, "r2 {r2}");
    }

    #[test]
    fn non_positive_and_non_finite_points_are_excluded() {
        let rf = [1.0, 2.0, -1.0, f64::NAN, 4.0];
        let rho = [5.0, 1.25, 1.0, 1.0, 0.3125];
        let (eta, _, _) = fit_visitation_law_impl(&rf, &rho, None, None).unwrap();
        assert!((eta - 2.0).abs() < 1e-12, "eta {eta}");
    }

    #[test]
    fn the_fitting_range_can_be_restricted() {
        let rf = [1.0, 2.0, 4.0, 1e6];
        let rho = [5.0, 1.25, 0.3125, 1.0];
        let (eta, _, _) = fit_visitation_law_impl(&rf, &rho, None, Some(10.0)).unwrap();
        assert!(
            (eta - 2.0).abs() < 1e-12,
            "outlier should be excluded: {eta}"
        );
    }

    #[test]
    fn fewer_than_two_usable_points_is_an_error() {
        assert!(fit_visitation_law_impl(&[1.0], &[1.0], None, None).is_err());
        assert!(fit_visitation_law_impl(&[1.0, -1.0], &[1.0, 1.0], None, None).is_err());
    }

    #[test]
    fn identical_rf_values_cannot_determine_a_slope() {
        assert!(fit_visitation_law_impl(&[2.0, 2.0], &[1.0, 3.0], None, None).is_err());
    }

    #[test]
    fn invalid_bounds_are_rejected() {
        let rf = [1.0, 2.0];
        let rho = [1.0, 2.0];
        assert!(fit_visitation_law_impl(&rf, &rho, Some(0.0), None).is_err());
        assert!(fit_visitation_law_impl(&rf, &rho, None, Some(-1.0)).is_err());
    }

    #[test]
    fn density_counts_distinct_users_per_location_radius_frequency() {
        // Two users at the same location, radius bin and frequency: one group
        // of size 2. r_center = 0.5, annulus area = 2*pi*0.5*1 = pi.
        let (rf, rho) =
            bin_visitation_law_impl(&[1, 2], &[7, 7], &[0.25, 0.75], &[3.0, 3.0], 10, 1.0).unwrap();
        assert_eq!(rf.len(), 1, "all observations collapse into one rf value");
        assert!((rf[0] - 1.5).abs() < 1e-12, "rf {:?}", rf[0]);
        assert!(
            (rho[0] - 2.0 / std::f64::consts::PI).abs() < 1e-12,
            "rho {:?}",
            rho[0]
        );
    }

    #[test]
    fn repeated_visits_by_one_user_count_once() {
        let (_, rho) =
            bin_visitation_law_impl(&[1, 1, 1], &[7, 7, 7], &[0.5, 0.5, 0.5], &[3.0; 3], 10, 1.0)
                .unwrap();
        assert!(
            (rho[0] - 1.0 / std::f64::consts::PI).abs() < 1e-12,
            "{rho:?}"
        );
    }

    #[test]
    fn non_positive_rows_are_skipped_and_can_empty_the_result() {
        let (rf, rho) =
            bin_visitation_law_impl(&[1, 2], &[1, 1], &[0.0, -1.0], &[1.0, 1.0], 10, 1.0).unwrap();
        assert!(rf.is_empty());
        assert!(rho.is_empty());
    }

    #[test]
    fn binning_is_independent_of_row_order() {
        let users = [1u64, 2, 3, 4, 5, 6];
        let locations = [1u64, 2, 3, 1, 2, 3];
        let radii = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5];
        let freqs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let forward = bin_visitation_law_impl(&users, &locations, &radii, &freqs, 4, 1.0).unwrap();

        let rev = |v: &[f64]| v.iter().rev().copied().collect::<Vec<_>>();
        let rev_u = |v: &[u64]| v.iter().rev().copied().collect::<Vec<_>>();
        let backward = bin_visitation_law_impl(
            &rev_u(&users),
            &rev_u(&locations),
            &rev(&radii),
            &rev(&freqs),
            4,
            1.0,
        )
        .unwrap();
        assert_eq!(forward, backward);
    }

    #[test]
    fn invalid_binning_parameters_are_rejected() {
        assert!(bin_visitation_law_impl(&[1], &[1], &[1.0], &[1.0], 0, 1.0).is_err());
        assert!(bin_visitation_law_impl(&[1], &[1], &[1.0], &[1.0], 10, 0.0).is_err());
        assert!(bin_visitation_law_impl(&[1], &[1, 2], &[1.0], &[1.0], 10, 1.0).is_err());
    }
}
