use fastmob_core::trajectory::shape_cluster::cluster_shape_signatures;
use fastmob_core::trajectory::shape_signature::batch_distance_geometry_signatures;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Compute the distance-geometry shape signature for each independent
/// trajectory in `trajectories` (a Python list of `(lats, lngs)` NumPy-array
/// pairs -- a list of independent whole sequences, not a single grouped
/// dataframe, so this takes the same "no shared grouping" shape as
/// `trajectory_distance`'s two-sequence binding rather than one of
/// `adapters::trajectory`'s coordinate-shaped generics).
///
/// Returns a flat `n_trajectories * depth*(depth+1)/2` array (row-major,
/// one trajectory's signature per contiguous chunk); the caller already
/// knows `depth`, so reshaping to `(n_trajectories, depth*(depth+1)/2)` is
/// left to the Python side rather than binding a 2D NumPy array here.
#[pyfunction]
pub fn trajectory_shape_signatures<'py>(
    py: Python<'py>,
    trajectories: Vec<(PyReadonlyArray1<'py, f64>, PyReadonlyArray1<'py, f64>)>,
    depth: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let owned: Vec<(Vec<f64>, Vec<f64>)> = trajectories
        .iter()
        .map(|(lats, lngs)| Ok((lats.as_slice()?.to_vec(), lngs.as_slice()?.to_vec())))
        .collect::<PyResult<_>>()?;

    let signatures = py
        .detach(|| batch_distance_geometry_signatures(&owned, depth))
        .map_err(PyValueError::new_err)?;

    let flat: Vec<f64> = signatures.into_iter().flatten().collect();
    Ok(flat.into_pyarray(py))
}

/// Euclidean-eps DBSCAN over already-computed shape signatures, given as a
/// flat `n_trajectories * signature_len` array (row-major) plus the
/// explicit `signature_len` (mirroring `trajectory_shape_signatures`'s flat
/// output convention). Returns one label per trajectory: a non-negative
/// cluster id, or `-1` for noise (same convention as
/// `fastmob.preprocessing.cluster`).
#[pyfunction]
pub fn cluster_trajectory_shape_signatures<'py>(
    py: Python<'py>,
    flat_signatures: PyReadonlyArray1<'py, f64>,
    signature_len: usize,
    epsilon: f64,
    min_cluster_size: usize,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let flat = flat_signatures.as_slice()?;
    if signature_len == 0 || !flat.len().is_multiple_of(signature_len) {
        return Err(PyValueError::new_err(
            "flat_signatures length must be a multiple of signature_len",
        ));
    }
    let rows: Vec<Vec<f64>> = flat
        .chunks_exact(signature_len)
        .map(<[f64]>::to_vec)
        .collect();

    let labels = py
        .detach(|| cluster_shape_signatures(&rows, epsilon, min_cluster_size))
        .map_err(PyValueError::new_err)?;

    Ok(labels.into_pyarray(py))
}
