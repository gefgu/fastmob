use rayon::prelude::*;
use std::sync::Arc;

use crate::utils::haversine::haversine_km;

const EARTH_RADIUS_KM: f64 = 6371.01;
pub const EPR_OD_CACHE_SIZE: u64 = 2_000;

fn haversine_km_radians(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let dlat = lat1 - lat2;
    let dlon = lon1 - lon2;
    let ds = 2.0
        * ((dlat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (dlon / 2.0).sin().powi(2))
            .sqrt()
            .asin();
    EARTH_RADIUS_KM * ds
}

pub fn deterrence(distance: f64, deterrence_type: &str, arg: f64) -> f64 {
    if deterrence_type == "exponential" {
        (-distance * arg).exp()
    } else {
        distance.powf(arg)
    }
}

pub struct CachedGravityOdRows<'a> {
    cache: moka::sync::Cache<usize, Arc<[f64]>>,
    lats: &'a [f64],
    lons: &'a [f64],
    rels_dest: Vec<f64>,
    rels_origin: Vec<f64>,
    deterrence_type: &'a str,
    deterrence_arg: f64,
}

impl<'a> CachedGravityOdRows<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        lats: &'a [f64],
        lons: &'a [f64],
        rels: &[f64],
        deterrence_type: &'a str,
        deterrence_arg: f64,
        origin_exp: f64,
        dest_exp: f64,
    ) -> Self {
        let rels_dest: Vec<f64> = rels.iter().map(|&r| r.powf(dest_exp)).collect();
        let rels_origin: Vec<f64> = rels.iter().map(|&r| r.powf(origin_exp)).collect();
        Self {
            cache: moka::sync::Cache::new(EPR_OD_CACHE_SIZE),
            lats,
            lons,
            rels_dest,
            rels_origin,
            deterrence_type,
            deterrence_arg,
        }
    }

    pub fn get(&self, origin: usize) -> Arc<[f64]> {
        self.cache.get_with(origin, || {
            Arc::from(gravity_od_row_seq(
                origin,
                self.lats,
                self.lons,
                &self.rels_dest,
                &self.rels_origin,
                self.deterrence_type,
                self.deterrence_arg,
            ))
        })
    }
}

pub fn gravity_od_row_seq(
    origin: usize,
    lats: &[f64],
    lons: &[f64],
    rels_dest: &[f64],
    rels_origin: &[f64],
    deterrence_type: &str,
    deterrence_arg: f64,
) -> Vec<f64> {
    let n = lats.len();
    let origin_power = rels_origin[origin];
    let mut row: Vec<f64> = (0..n)
        .map(|j| {
            if j == origin {
                return 0.0;
            }
            let d = haversine_km(lats[origin], lons[origin], lats[j], lons[j]);
            let s = deterrence(d, deterrence_type, deterrence_arg) * rels_dest[j] * origin_power;
            if s.is_finite() { s } else { 0.0 }
        })
        .collect();
    let total: f64 = row.iter().sum();
    if total != 0.0 {
        row.iter_mut().for_each(|v| *v /= total);
    } else if n > 0 {
        row.fill(1.0 / n as f64);
    }
    let mut cumsum = 0.0;
    for value in &mut row {
        cumsum += *value;
        *value = cumsum;
    }
    row
}

pub fn validate_equal_lengths(arrays: &[(&str, usize)]) -> Result<usize, String> {
    let Some((_, n)) = arrays.first() else {
        return Ok(0);
    };
    for (name, len) in arrays {
        if len != n {
            return Err(format!(
                "{name} must have the same length as the coordinate arrays"
            ));
        }
    }
    Ok(*n)
}

#[allow(clippy::too_many_arguments)]
pub fn gravity_matrix_impl(
    lats: &[f64],
    lons: &[f64],
    relevances: &[f64],
    tot_outflows: &[f64],
    deterrence_type: &str,
    deterrence_arg: f64,
    origin_exp: f64,
    destination_exp: f64,
    gravity_type: &str,
    out_format: &str,
) -> Result<Vec<f64>, String> {
    let n = validate_equal_lengths(&[
        ("longitudes", lons.len()),
        ("relevances", relevances.len()),
        ("tot_outflows", tot_outflows.len()),
        ("latitudes", lats.len()),
    ])?;

    let lats_rad: Vec<f64> = lats.par_iter().map(|x| x.to_radians()).collect();
    let lons_rad: Vec<f64> = lons.par_iter().map(|x| x.to_radians()).collect();
    let rels_origin: Vec<f64> = relevances.par_iter().map(|x| x.powf(origin_exp)).collect();
    let rels_dest: Vec<f64> = relevances
        .par_iter()
        .map(|x| x.powf(destination_exp))
        .collect();

    let mut matrix = vec![0.0; n * n];

    matrix.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for j in 0..n {
            if i == j {
                row[j] = 0.0;
            } else {
                let distance =
                    haversine_km_radians(lats_rad[i], lons_rad[i], lats_rad[j], lons_rad[j]);
                let score = deterrence(distance, deterrence_type, deterrence_arg)
                    * rels_dest[j]
                    * rels_origin[i];
                row[j] = if score.is_finite() { score } else { 0.0 };
            }
        }
    });

    if gravity_type == "globally constrained" {
        let total: f64 = matrix.par_iter().sum();
        if total != 0.0 {
            for value in &mut matrix {
                *value /= total;
            }
        }
        if out_format == "flows" {
            let total_outflow: f64 = tot_outflows.par_iter().sum();
            for value in &mut matrix {
                *value *= total_outflow;
            }
        }
    } else {
        matrix.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            let row_sum: f64 = row.par_iter().sum();
            if row_sum != 0.0 {
                for value in row.iter_mut() {
                    *value /= row_sum;
                    if out_format == "flows" {
                        *value *= tot_outflows[i];
                    }
                }
            } else {
                for value in row.iter_mut() {
                    *value = 0.0;
                }
            }
        });
    }

    Ok(matrix)
}

#[allow(clippy::too_many_arguments)]
pub fn gravity_od_row_impl(
    origin: usize,
    latitudes: &[f64],
    longitudes: &[f64],
    relevances: &[f64],
    deterrence_type: &str,
    deterrence_arg: f64,
    origin_exp: f64,
    destination_exp: f64,
) -> Result<Vec<f64>, String> {
    let n = validate_equal_lengths(&[
        ("longitudes", longitudes.len()),
        ("relevances", relevances.len()),
        ("latitudes", latitudes.len()),
    ])?;
    if origin >= n {
        return Err("origin must be within coordinate bounds".to_string());
    }

    let mut row: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|j| {
            if j == origin {
                return 0.0;
            }
            let distance = haversine_km(
                latitudes[origin],
                longitudes[origin],
                latitudes[j],
                longitudes[j],
            );
            let score = deterrence(distance, deterrence_type, deterrence_arg)
                * relevances[j].powf(destination_exp)
                * relevances[origin].powf(origin_exp);
            if score.is_finite() { score } else { 0.0 }
        })
        .collect();
    let total: f64 = row.iter().sum();
    if total != 0.0 {
        for value in &mut row {
            *value /= total;
        }
    } else if n > 0 {
        let uniform = 1.0 / n as f64;
        row.fill(uniform);
    }
    Ok(row)
}
