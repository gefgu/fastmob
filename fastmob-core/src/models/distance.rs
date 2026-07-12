use rayon::prelude::*;

use crate::models::od::validate_equal_lengths;
use crate::utils::haversine::haversine_km;

pub fn model_distance_matrix_impl(
    latitudes: &[f64],
    longitudes: &[f64],
) -> Result<Vec<f64>, String> {
    let n = validate_equal_lengths(&[
        ("longitudes", longitudes.len()),
        ("latitudes", latitudes.len()),
    ])?;
    let flat: Vec<f64> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| {
            (0..n).map(move |j| {
                if i == j {
                    0.0
                } else {
                    haversine_km(latitudes[i], longitudes[i], latitudes[j], longitudes[j])
                }
            })
        })
        .collect();
    Ok(flat)
}
