//! The universal visitation law over a `(user, location, r_km, f)` frame.

use fastmob_core::measures::fitting::visitation_law::{
    bin_visitation_law_impl, fit_visitation_law_impl,
};
use polars::prelude::*;

use crate::error::FastmobRsError;
use crate::prepare::categorical::factorize;

/// Aggregate visitation-law rows into logarithmically binned `(rf, rho)` pairs.
///
/// `law_data` needs a user column, a location column, and `r_km`/`f`. Both
/// identifier columns are factorized rather than parsed, so composite string
/// ids work unchanged.
pub fn bin_visitation_law_data(
    law_data: &DataFrame,
    user_col: &str,
    location_col: &str,
    n_bins: usize,
    distance_bin_width_km: f64,
) -> Result<(Vec<f64>, Vec<f64>), FastmobRsError> {
    let users = factorize(law_data.column(user_col)?.as_materialized_series())?;
    let locations = factorize(law_data.column(location_col)?.as_materialized_series())?;
    let r_km = f64_values(law_data, "r_km")?;
    let f = f64_values(law_data, "f")?;

    bin_visitation_law_impl(
        &users.codes,
        &locations.codes,
        &r_km,
        &f,
        n_bins,
        distance_bin_width_km,
    )
    .map_err(FastmobRsError::Core)
}

/// Fit `rho(r, f) = mu * (rf)^-eta`, returning `(eta, mu, r2)`.
pub fn fit_visitation_law(
    rf_values: &[f64],
    rho_values: &[f64],
) -> Result<(f64, f64, f64), FastmobRsError> {
    fit_visitation_law_impl(rf_values, rho_values, None, None).map_err(FastmobRsError::Core)
}

/// Like [`fit_visitation_law`], restricted to an inclusive `rf` range.
pub fn fit_visitation_law_in_range(
    rf_values: &[f64],
    rho_values: &[f64],
    min_rf: Option<f64>,
    max_rf: Option<f64>,
) -> Result<(f64, f64, f64), FastmobRsError> {
    fit_visitation_law_impl(rf_values, rho_values, min_rf, max_rf).map_err(FastmobRsError::Core)
}

fn f64_values(df: &DataFrame, name: &str) -> Result<Vec<f64>, FastmobRsError> {
    let cast = df
        .column(name)?
        .as_materialized_series()
        .cast(&DataType::Float64)?;
    Ok(cast
        .f64()?
        .into_iter()
        .map(|value| value.unwrap_or(f64::NAN))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn law_data() -> DataFrame {
        df![
            "user_id" => ["10_1", "10_2"],
            "location_id" => ["a", "a"],
            "r_km" => [0.25, 0.75],
            "f" => [3.0, 3.0],
        ]
        .unwrap()
    }

    #[test]
    fn composite_string_ids_stay_distinct_users() {
        let (rf, rho) =
            bin_visitation_law_data(&law_data(), "user_id", "location_id", 10, 1.0).unwrap();
        assert_eq!(rf.len(), 1);
        // Two distinct users in one annulus of area pi.
        assert!(
            (rho[0] - 2.0 / std::f64::consts::PI).abs() < 1e-12,
            "rho {rho:?}"
        );
    }

    #[test]
    fn a_missing_column_is_reported_rather_than_panicking() {
        let df = df!["user_id" => ["a"], "location_id" => ["b"], "r_km" => [1.0]].unwrap();
        assert!(bin_visitation_law_data(&df, "user_id", "location_id", 10, 1.0).is_err());
    }

    #[test]
    fn a_perfect_power_law_is_recovered() {
        let rf: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let rho: Vec<f64> = rf.iter().map(|x| 5.0 * x.powf(-2.0)).collect();
        let (eta, mu, r2) = fit_visitation_law(&rf, &rho).unwrap();
        assert!((eta - 2.0).abs() < 1e-12);
        assert!((mu - 5.0).abs() < 1e-12);
        assert!((r2 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_fit_range_can_exclude_an_outlier() {
        let rf = [1.0, 2.0, 4.0, 1e6];
        let rho = [5.0, 1.25, 0.3125, 1.0];
        let (eta, _, _) = fit_visitation_law_in_range(&rf, &rho, None, Some(10.0)).unwrap();
        assert!((eta - 2.0).abs() < 1e-12, "eta {eta}");
    }
}
