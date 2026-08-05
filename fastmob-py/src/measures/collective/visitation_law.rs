use arrow_array::{Array, ArrayRef, UInt64Array};
use arrow_schema::DataType;
use fastmob_core::measures::{
    collective::visitation_law::visitation_distances_km as core_visitation_distances_km,
    fitting::visitation_law::{aggregate_visitation_cells_impl, bin_visitation_law_impl},
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::{
    measures::individual::factorization::factorize_array,
    utils::{
        arrow_i64_values, arrow_u64_values, arrow_values, as_f64_array, as_i64_array, as_u64_array,
        f64_results_into_arrow, u64_results_into_arrow,
    },
};

const MILLISECONDS_PER_DAY: i64 = 86_400_000;

#[pyfunction]
pub fn visitation_distances_km(
    home_latitudes: Vec<f64>,
    home_longitudes: Vec<f64>,
    location_latitudes: Vec<f64>,
    location_longitudes: Vec<f64>,
) -> PyResult<Vec<f64>> {
    core_visitation_distances_km(
        home_latitudes,
        home_longitudes,
        location_latitudes,
        location_longitudes,
    )
    .map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn visitation_distances(
    home_latitudes: ArrowPyArray,
    home_longitudes: ArrowPyArray,
    location_latitudes: ArrowPyArray,
    location_longitudes: ArrowPyArray,
) -> PyResult<Vec<f64>> {
    let home_latitudes = as_f64_array(home_latitudes, "home_latitudes")?;
    let home_longitudes = as_f64_array(home_longitudes, "home_longitudes")?;
    let location_latitudes = as_f64_array(location_latitudes, "location_latitudes")?;
    let location_longitudes = as_f64_array(location_longitudes, "location_longitudes")?;
    core_visitation_distances_km(
        arrow_values(&home_latitudes).to_vec(),
        arrow_values(&home_longitudes).to_vec(),
        arrow_values(&location_latitudes).to_vec(),
        arrow_values(&location_longitudes).to_vec(),
    )
    .map_err(PyValueError::new_err)
}

/// Return grouping codes for `array`, skipping factorization when it is
/// already a non-null UInt64 array.
///
/// `bin_visitation_law_impl` only uses these codes as `HashMap` key
/// components -- it never needs the dense `0..n_unique` property
/// `factorize_array`'s codes have, just *some* hashable u64 per distinct
/// value. A non-null UInt64 array (H3 cell indices, or user codes a caller
/// already factorized upstream) already satisfies that, so re-running
/// `factorize_array` on it is pure duplicate work. This is scoped to this
/// one call site rather than a change to `factorize_array` itself: other
/// callers (e.g. label reconstruction via `pc.take`) do rely on the dense
/// property, which raw non-factorized UInt64 values would not have.
fn codes_for_grouping(array: ArrayRef) -> Result<Vec<u64>, String> {
    if *array.data_type() == DataType::UInt64 {
        if let Some(typed) = array.as_any().downcast_ref::<UInt64Array>() {
            if typed.null_count() == 0 {
                let start = typed.offset();
                let end = start + typed.len();
                return Ok(typed.values()[start..end].to_vec());
            }
        }
    }
    factorize_array(array.as_ref(), false)
        .map(|(codes, _)| codes.into_iter().map(u64::from).collect())
}

/// Bin visitation-law observations directly from Arrow columns.
///
/// Identifier arrays that are not already a non-null UInt64 array are
/// factorized inside the extension so callers do not need dataframe-specific
/// categorical conversions. Numeric arrays must be non-null float64 Arrow
/// arrays.
#[pyfunction]
pub fn bin_visitation_law_arrow(
    py: Python<'_>,
    user_ids: ArrowPyArray,
    location_ids: ArrowPyArray,
    r_km: ArrowPyArray,
    f: ArrowPyArray,
    n_bins: usize,
    distance_bin_width_km: f64,
) -> PyResult<(ArrowPyArray, ArrowPyArray)> {
    let (user_ids, _) = user_ids.into_inner();
    let (location_ids, _) = location_ids.into_inner();
    let r_km = as_f64_array(r_km, "r_km")?;
    let f = as_f64_array(f, "f")?;
    let (rf, rho) = py
        .detach(|| {
            let user_codes = codes_for_grouping(user_ids)?;
            let location_codes = codes_for_grouping(location_ids)?;
            bin_visitation_law_impl(
                &user_codes,
                &location_codes,
                arrow_values(&r_km),
                arrow_values(&f),
                n_bins,
                distance_bin_width_km,
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok((f64_results_into_arrow(rf), f64_results_into_arrow(rho)))
}

/// Aggregate raw staypoints into one row per `(user, h3 cell)` pair directly
/// from Arrow columns.
///
/// `user_codes` are dense UInt64 codes the caller has already factorized (the
/// per-group codes are echoed back so the caller can map them to labels via
/// the same factorization's representatives), `h3_cells` are UInt64 H3 cell
/// indices, `timestamps_ms` are non-null Int64 Unix milliseconds bucketed
/// into calendar days inside the extension, and `lats`/`lngs` are non-null
/// Float64 coordinates. Returns
/// `(user_codes, h3_cells, n_staypoints, f, loc_lat, loc_lng)`.
#[pyfunction]
pub fn aggregate_visitation_cells_arrow(
    py: Python<'_>,
    user_codes: ArrowPyArray,
    h3_cells: ArrowPyArray,
    timestamps_ms: ArrowPyArray,
    lats: ArrowPyArray,
    lngs: ArrowPyArray,
) -> PyResult<(
    ArrowPyArray,
    ArrowPyArray,
    ArrowPyArray,
    ArrowPyArray,
    ArrowPyArray,
    ArrowPyArray,
)> {
    let user_codes = as_u64_array(user_codes, "user_codes")?;
    let h3_cells = as_u64_array(h3_cells, "h3_cells")?;
    let timestamps_ms = as_i64_array(timestamps_ms, "timestamps_ms")?;
    let lats = as_f64_array(lats, "lats")?;
    let lngs = as_f64_array(lngs, "lngs")?;
    let (out_users, out_cells, out_counts, out_f, out_lat, out_lng) = py
        .detach(|| {
            let days: Vec<i64> = arrow_i64_values(&timestamps_ms)
                .iter()
                .map(|ms| ms.div_euclid(MILLISECONDS_PER_DAY))
                .collect();
            aggregate_visitation_cells_impl(
                arrow_u64_values(&user_codes),
                arrow_u64_values(&h3_cells),
                &days,
                arrow_values(&lats),
                arrow_values(&lngs),
            )
        })
        .map_err(PyValueError::new_err)?;
    Ok((
        u64_results_into_arrow(out_users),
        u64_results_into_arrow(out_cells),
        u64_results_into_arrow(out_counts),
        f64_results_into_arrow(out_f),
        f64_results_into_arrow(out_lat),
        f64_results_into_arrow(out_lng),
    ))
}
