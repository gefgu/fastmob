use ndarray::{Array1, Array2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use wass::sinkhorn_log;

use crate::haversine::haversine_km;

/// Build an (m × n) pairwise Haversine cost matrix (in km) as f32.
///
/// # Parameters
/// - `lats_a`: latitudes of the first point cloud (length m).
/// - `lons_a`: longitudes of the first point cloud (length m).
/// - `lats_b`: latitudes of the second point cloud (length n).
/// - `lons_b`: longitudes of the second point cloud (length n).
///
/// # Returns
/// An `ndarray::Array2<f32>` of shape (m, n).
fn build_cost_matrix(
    lats_a: &[f64],
    lons_a: &[f64],
    lats_b: &[f64],
    lons_b: &[f64],
) -> Array2<f32> {
    let m = lats_a.len();
    let n = lats_b.len();
    let mut cost = Array2::<f32>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            cost[[i, j]] = haversine_km(lats_a[i], lons_a[i], lats_b[j], lons_b[j]) as f32;
        }
    }
    cost
}

/// Compute the Earth Mover's Distance (Wasserstein-1) between two GPS point clouds.
///
/// Uses the Sinkhorn log-domain algorithm (via the `wass` crate) with a pairwise
/// Haversine cost matrix.  Weights are uniform by default (uniform marginals).
///
/// # Parameters
/// - `lats_a`: latitudes of the first trajectory (f64 degrees).
/// - `lons_a`: longitudes of the first trajectory (f64 degrees).
/// - `lats_b`: latitudes of the second trajectory (f64 degrees).
/// - `lons_b`: longitudes of the second trajectory (f64 degrees).
/// - `reg`:    Sinkhorn regularisation parameter (smaller → more exact, slower).
/// - `max_iter`: maximum Sinkhorn iterations.
///
/// # Returns
/// The Wasserstein distance in km (f64).
///
/// # Called from
/// `skmob2/measures/wasserstein_distance.py` → `wasserstein_distance`.
///
/// # Side effects
/// None.
#[pyfunction]
#[pyo3(signature = (lats_a, lons_a, lats_b, lons_b, reg=0.1, max_iter=100))]
pub(crate) fn wasserstein_emd(
    lats_a: Vec<f64>,
    lons_a: Vec<f64>,
    lats_b: Vec<f64>,
    lons_b: Vec<f64>,
    reg: f64,
    max_iter: usize,
) -> PyResult<f64> {
    if lats_a.len() != lons_a.len() {
        return Err(PyValueError::new_err(
            "lats_a and lons_a must have the same length",
        ));
    }
    if lats_b.len() != lons_b.len() {
        return Err(PyValueError::new_err(
            "lats_b and lons_b must have the same length",
        ));
    }
    if lats_a.is_empty() || lats_b.is_empty() {
        return Err(PyValueError::new_err(
            "both point clouds must be non-empty",
        ));
    }

    let m = lats_a.len();
    let n = lats_b.len();

    // Uniform marginals (each point carries equal weight).
    let a: Array1<f32> = Array1::from_elem(m, 1.0_f32 / m as f32);
    let b: Array1<f32> = Array1::from_elem(n, 1.0_f32 / n as f32);

    let cost = build_cost_matrix(&lats_a, &lons_a, &lats_b, &lons_b);

    // sinkhorn_log returns (transport_plan: Array2<f32>, distance: f32)
    let (_plan, distance) = sinkhorn_log(&a, &b, &cost, reg as f32, max_iter);

    Ok(distance as f64)
}
