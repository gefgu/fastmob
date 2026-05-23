use geo::{Distance, Haversine, Point};
use numkong::Haversine as NumKongHaversine;
use pyo3::prelude::*;

const GEO_HAVERSINE_RADIUS_M: f64 = 6_371_008.8;
const NUMKONG_HAVERSINE_RADIUS_M: f64 = 6_335_439.0;
const NUMKONG_TO_GEO_KM: f64 = GEO_HAVERSINE_RADIUS_M / NUMKONG_HAVERSINE_RADIUS_M / 1000.0;

#[pyfunction]
pub(crate) fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let p1 = Point::new(lon1, lat1);
    let p2 = Point::new(lon2, lat2);
    Haversine.distance(p1, p2) / 1000.0
}

pub(crate) fn adjacent_haversine_distances_km(
    latitudes: &[f64],
    longitudes: &[f64],
    start: usize,
    end: usize,
) -> Vec<f64> {
    if end - start < 2 {
        return Vec::new();
    }

    let mut distances = vec![0.0; end - start - 1];
    adjacent_haversine_distances_into_km(latitudes, longitudes, start, end, &mut distances);
    distances
}

pub(crate) fn adjacent_haversine_distances_into_km(
    latitudes: &[f64],
    longitudes: &[f64],
    start: usize,
    end: usize,
    distances: &mut [f64],
) {
    if end - start < 2 {
        return;
    }

    let latitudes_rad: Vec<f64> = latitudes[start..end]
        .iter()
        .map(|lat| lat.to_radians())
        .collect();
    let longitudes_rad: Vec<f64> = longitudes[start..end]
        .iter()
        .map(|lon| lon.to_radians())
        .collect();

    f64::haversine(
        &latitudes_rad[..latitudes_rad.len() - 1],
        &longitudes_rad[..longitudes_rad.len() - 1],
        &latitudes_rad[1..],
        &longitudes_rad[1..],
        distances,
    )
    .expect("adjacent coordinate slices have matching lengths");

    distances
        .iter_mut()
        .for_each(|distance| *distance *= NUMKONG_TO_GEO_KM);
}

pub(crate) fn adjacent_haversine_sum_km(
    latitudes: &[f64],
    longitudes: &[f64],
    start: usize,
    end: usize,
) -> f64 {
    adjacent_haversine_distances_km(latitudes, longitudes, start, end)
        .into_iter()
        .sum()
}

pub(crate) fn adjacent_haversine_max_km(
    latitudes: &[f64],
    longitudes: &[f64],
    start: usize,
    end: usize,
) -> f64 {
    adjacent_haversine_distances_km(latitudes, longitudes, start, end)
        .into_iter()
        .fold(0.0f64, f64::max)
}
