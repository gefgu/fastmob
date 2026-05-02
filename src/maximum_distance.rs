use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use rayon::prelude::*;

use crate::haversine::haversine_km;
use crate::utils::{as_f64_array, arrow_values, validate_coord_ranges};

fn maximum_distance_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ranges: &[(usize, usize)],
) -> PyResult<Vec<f64>> {
    validate_coord_ranges(latitudes, longitudes, ranges)?;

    let results: Vec<f64> = ranges
        .par_iter()
        .map(|&(start, end)| {
            if end - start < 2 {
                return 0.0;
            }
            (start + 1..end)
                .map(|idx| {
                    haversine_km(
                        latitudes[idx - 1],
                        longitudes[idx - 1],
                        latitudes[idx],
                        longitudes[idx],
                    )
                })
                .fold(0.0f64, f64::max)
        })
        .collect();

    Ok(results)
}

#[pyfunction]
pub(crate) fn maximum_distance_batch_km(
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    maximum_distance_impl(&latitudes, &longitudes, &ranges)
}

#[pyfunction]
pub(crate) fn maximum_distance_numpy(
    latitudes: PyReadonlyArray1<f64>,
    longitudes: PyReadonlyArray1<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    maximum_distance_impl(latitudes.as_slice()?, longitudes.as_slice()?, &ranges)
}

#[pyfunction]
pub(crate) fn maximum_distance_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    maximum_distance_impl(arrow_values(&latitudes), arrow_values(&longitudes), &ranges)
}
