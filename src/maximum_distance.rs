use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rayon::prelude::*;

use crate::haversine::haversine_km;

#[pyfunction]
pub(crate) fn maximum_distance_batch_km(
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }

    let coords: Vec<(f64, f64)> = latitudes.into_iter().zip(longitudes).collect();

    let results: Vec<f64> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let slice = &coords[start..end];
            if slice.len() < 2 {
                return 0.0;
            }
            slice
                .windows(2)
                .map(|w| haversine_km(w[0].0, w[0].1, w[1].0, w[1].1))
                .fold(0.0f64, f64::max)
        })
        .collect();

    Ok(results)
}
