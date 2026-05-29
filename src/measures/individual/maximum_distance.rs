use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use rayon::prelude::*;

use crate::haversine::{adjacent_haversine_max_km, haversine_km};
use crate::utils::{
    arrow_values, as_f64_array, f64_results_into_arrow, ranges_from_starts_ends,
    validate_coord_ranges, validate_indexed_coord_ranges,
};

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
            adjacent_haversine_max_km(latitudes, longitudes, start, end)
        })
        .collect();

    Ok(results)
}

fn maximum_distance_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> PyResult<Vec<f64>> {
    validate_indexed_coord_ranges(latitudes, longitudes, indices, ranges)?;

    let results: Vec<f64> = ranges
        .par_iter()
        .map(|&(start, end)| {
            if end - start < 2 {
                return 0.0;
            }
            (start + 1..end)
                .map(|pos| {
                    let previous_idx = indices[pos - 1];
                    let current_idx = indices[pos];
                    haversine_km(
                        latitudes[previous_idx],
                        longitudes[previous_idx],
                        latitudes[current_idx],
                        longitudes[current_idx],
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
pub(crate) fn maximum_distance_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    starts: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    Ok(maximum_distance_indexed_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        indices.as_slice()?,
        &ranges,
    )?
    .into_pyarray(py))
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

#[pyfunction]
pub(crate) fn maximum_distance_indexed_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    indices: PyReadonlyArray1<usize>,
    starts: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<PyArray> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    Ok(f64_results_into_arrow(maximum_distance_indexed_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        indices.as_slice()?,
        &ranges,
    )?))
}
