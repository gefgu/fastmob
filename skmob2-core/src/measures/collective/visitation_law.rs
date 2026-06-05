use rayon::prelude::*;

use crate::utils::haversine::haversine_km;

fn visitation_distances_impl(
    home_latitudes: &[f64],
    home_longitudes: &[f64],
    location_latitudes: &[f64],
    location_longitudes: &[f64],
) -> Result<Vec<f64>, String> {
    if home_latitudes.len() != home_longitudes.len() {
        return Err("home_latitudes and home_longitudes must have the same length".to_string());
    }
    if location_latitudes.len() != location_longitudes.len() {
        return Err(
            "location_latitudes and location_longitudes must have the same length".to_string(),
        );
    }
    if home_latitudes.len() != location_latitudes.len() {
        return Err("home and location coordinate arrays must have the same length".to_string());
    }

    Ok((0..home_latitudes.len())
        .into_par_iter()
        .map(|idx| {
            haversine_km(
                home_latitudes[idx],
                home_longitudes[idx],
                location_latitudes[idx],
                location_longitudes[idx],
            )
        })
        .collect())
}

pub fn visitation_distances_km(
    home_latitudes: Vec<f64>,
    home_longitudes: Vec<f64>,
    location_latitudes: Vec<f64>,
    location_longitudes: Vec<f64>,
) -> Result<Vec<f64>, String> {
    visitation_distances_impl(
        &home_latitudes,
        &home_longitudes,
        &location_latitudes,
        &location_longitudes,
    )
}
