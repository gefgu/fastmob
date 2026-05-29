use std::collections::HashSet;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use rayon::prelude::*;

use crate::utils::{
    arrow_values, as_f64_array, ranges_from_starts_ends, u64_results_into_arrow,
    validate_coord_ranges, validate_indexed_coord_ranges, validate_indexed_ranges, validate_ranges,
};

fn number_of_visits_impl(n_values: usize, ranges: &[(usize, usize)]) -> PyResult<Vec<u64>> {
    validate_ranges(n_values, ranges)?;
    Ok(ranges
        .iter()
        .map(|&(start, end)| (end - start) as u64)
        .collect())
}

fn number_of_visits_indexed_impl(
    n_values: usize,
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> PyResult<Vec<u64>> {
    validate_indexed_ranges(n_values, indices, ranges)?;
    Ok(ranges
        .iter()
        .map(|&(start, end)| (end - start) as u64)
        .collect())
}

fn number_of_locations_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ranges: &[(usize, usize)],
) -> PyResult<Vec<u64>> {
    validate_coord_ranges(latitudes, longitudes, ranges)?;
    Ok(ranges
        .par_iter()
        .map(|&(start, end)| {
            let mut seen = HashSet::with_capacity(end.saturating_sub(start));
            for idx in start..end {
                seen.insert((latitudes[idx].to_bits(), longitudes[idx].to_bits()));
            }
            seen.len() as u64
        })
        .collect())
}

fn number_of_locations_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> PyResult<Vec<u64>> {
    validate_indexed_coord_ranges(latitudes, longitudes, indices, ranges)?;
    Ok(ranges
        .par_iter()
        .map(|&(start, end)| {
            let mut seen = HashSet::with_capacity(end.saturating_sub(start));
            for &idx in &indices[start..end] {
                seen.insert((latitudes[idx].to_bits(), longitudes[idx].to_bits()));
            }
            seen.len() as u64
        })
        .collect())
}

#[pyfunction]
pub(crate) fn number_of_visits_numpy<'py>(
    py: Python<'py>,
    n_values: usize,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    Ok(number_of_visits_impl(n_values, &ranges)?.into_pyarray(py))
}

#[pyfunction]
pub(crate) fn number_of_visits_arrow(
    n_values: usize,
    ranges: Vec<(usize, usize)>,
) -> PyResult<PyArray> {
    Ok(u64_results_into_arrow(number_of_visits_impl(
        n_values, &ranges,
    )?))
}

#[pyfunction]
pub(crate) fn number_of_visits_indexed_numpy<'py>(
    py: Python<'py>,
    n_values: usize,
    indices: PyReadonlyArray1<'py, usize>,
    starts: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    Ok(number_of_visits_indexed_impl(n_values, indices.as_slice()?, &ranges)?.into_pyarray(py))
}

#[pyfunction]
pub(crate) fn number_of_visits_indexed_arrow(
    n_values: usize,
    indices: PyReadonlyArray1<usize>,
    starts: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<PyArray> {
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    Ok(u64_results_into_arrow(number_of_visits_indexed_impl(
        n_values,
        indices.as_slice()?,
        &ranges,
    )?))
}

#[pyfunction]
pub(crate) fn number_of_locations_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    Ok(
        number_of_locations_impl(latitudes.as_slice()?, longitudes.as_slice()?, &ranges)?
            .into_pyarray(py),
    )
}

#[pyfunction]
pub(crate) fn number_of_locations_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    ranges: Vec<(usize, usize)>,
) -> PyResult<PyArray> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    Ok(u64_results_into_arrow(number_of_locations_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        &ranges,
    )?))
}

#[pyfunction]
pub(crate) fn number_of_locations_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    starts: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    Ok(number_of_locations_indexed_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        indices.as_slice()?,
        &ranges,
    )?
    .into_pyarray(py))
}

#[pyfunction]
pub(crate) fn number_of_locations_indexed_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    indices: PyReadonlyArray1<usize>,
    starts: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<PyArray> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    Ok(u64_results_into_arrow(number_of_locations_indexed_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        indices.as_slice()?,
        &ranges,
    )?))
}
