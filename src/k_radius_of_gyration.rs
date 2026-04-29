use geo::{Distance, Haversine, Point};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction]
pub(crate) fn k_radius_of_gyration_km(
    coords: Vec<(f64, f64)>,
    visit_counts: Vec<u64>,
    k: usize,
) -> PyResult<f64> {
    if coords.len() != visit_counts.len() {
        return Err(PyValueError::new_err(
            "coords and visit_counts must have the same length",
        ));
    }

    if coords.is_empty() || k == 0 {
        return Ok(0.0);
    }

    let mut pairs: Vec<((f64, f64), u64)> = coords.into_iter().zip(visit_counts).collect();
    pairs.sort_unstable_by(|a, b| b.1.cmp(&a.1));
    let top_k = &pairs[..k.min(pairs.len())];

    let total_weight: f64 = top_k.iter().map(|(_, w)| *w as f64).sum();
    if total_weight == 0.0 {
        return Ok(0.0);
    }

    let (lat_sum, lng_sum) = top_k
        .iter()
        .fold((0.0f64, 0.0f64), |(ls, ns), &((lat, lng), w)| {
            (ls + lat * w as f64, ns + lng * w as f64)
        });
    let cm_lat = lat_sum / total_weight;
    let cm_lng = lng_sum / total_weight;
    let cm = Point::new(cm_lng, cm_lat);

    let sum_sq: f64 = top_k
        .iter()
        .map(|&((lat, lng), w)| {
            let p = Point::new(lng, lat);
            let d = Haversine.distance(p, cm) / 1000.0;
            w as f64 * d * d
        })
        .sum();

    Ok((sum_sq / total_weight).sqrt())
}
