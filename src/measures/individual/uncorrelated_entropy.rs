use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use rayon::prelude::*;
use rustc_hash::FxHashMap;

use crate::utils::{
    arrow_values, as_f64_array, f64_results_into_arrow, ranges_from_starts_ends,
    validate_indexed_coord_ranges,
};

fn uncorrelated_entropy_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
    normalize: bool,
) -> PyResult<Vec<f64>> {
    validate_indexed_coord_ranges(latitudes, longitudes, indices, ranges)?;

    Ok(ranges
        .par_iter()
        .map(|&(start, end)| {
            let n = end - start;
            if n == 0 {
                return 0.0;
            }
            let mut counts: FxHashMap<(u64, u64), u64> =
                FxHashMap::with_capacity_and_hasher(n, Default::default());
            for &idx in &indices[start..end] {
                let key = (latitudes[idx].to_bits(), longitudes[idx].to_bits());
                *counts.entry(key).or_insert(0) += 1;
            }
            let total = n as f64;
            let entropy = counts.values().fold(0.0f64, |acc, &c| {
                let p = c as f64 / total;
                acc - p * p.log2()
            });
            if normalize {
                let n_unique = counts.len();
                if n_unique > 1 {
                    entropy / (n_unique as f64).log2()
                } else {
                    0.0
                }
            } else {
                entropy
            }
        })
        .collect())
}

#[pyfunction]
pub(crate) fn uncorrelated_entropy_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    starts: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    normalize: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    Ok(uncorrelated_entropy_indexed_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        indices.as_slice()?,
        &ranges,
        normalize,
    )?
    .into_pyarray(py))
}

#[pyfunction]
pub(crate) fn uncorrelated_entropy_indexed_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    indices: PyReadonlyArray1<usize>,
    starts: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
    normalize: bool,
) -> PyResult<PyArray> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    Ok(f64_results_into_arrow(uncorrelated_entropy_indexed_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        indices.as_slice()?,
        &ranges,
        normalize,
    )?))
}
