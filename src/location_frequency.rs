use std::collections::HashMap;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use rayon::prelude::*;

use crate::utils::{
    arrow_values, as_f64_array, f64_results_into_arrow, ranges_from_starts_ends,
    u64_results_into_arrow, validate_indexed_coord_ranges,
};

type LocFreqData = (Vec<f64>, Vec<f64>, Vec<u64>, Vec<usize>, Vec<usize>);

fn location_frequency_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> PyResult<LocFreqData> {
    validate_indexed_coord_ranges(latitudes, longitudes, indices, ranges)?;

    let per_user: Vec<Vec<(f64, f64, u64)>> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let mut counts: HashMap<(u64, u64), (f64, f64, u64)> =
                HashMap::with_capacity(end.saturating_sub(start));
            for &idx in &indices[start..end] {
                let lat = latitudes[idx];
                let lng = longitudes[idx];
                let key = (lat.to_bits(), lng.to_bits());
                let entry = counts.entry(key).or_insert((lat, lng, 0));
                entry.2 += 1;
            }
            let mut locs: Vec<(f64, f64, u64)> = counts.into_values().collect();
            locs.sort_unstable_by(|a, b| {
                b.2.cmp(&a.2)
                    .then(a.0.total_cmp(&b.0))
                    .then(a.1.total_cmp(&b.1))
            });
            locs
        })
        .collect();

    let total_locs: usize = per_user.iter().map(|v| v.len()).sum();
    let mut out_lats = Vec::with_capacity(total_locs);
    let mut out_lngs = Vec::with_capacity(total_locs);
    let mut out_counts = Vec::with_capacity(total_locs);
    let mut out_user_starts = Vec::with_capacity(ranges.len());
    let mut out_user_ends = Vec::with_capacity(ranges.len());

    let mut offset = 0usize;
    for locs in per_user {
        out_user_starts.push(offset);
        for (lat, lng, count) in locs {
            out_lats.push(lat);
            out_lngs.push(lng);
            out_counts.push(count);
        }
        offset = out_lats.len();
        out_user_ends.push(offset);
    }

    Ok((out_lats, out_lngs, out_counts, out_user_starts, out_user_ends))
}

type LocFreqNumpy<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<u64>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
);

type LocFreqArrow<'py> = (
    PyArray,
    PyArray,
    PyArray,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
);

#[pyfunction]
pub(crate) fn location_frequency_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    starts: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<LocFreqNumpy<'py>> {
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    let (out_lats, out_lngs, out_counts, out_starts, out_ends) = location_frequency_indexed_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        indices.as_slice()?,
        &ranges,
    )?;
    Ok((
        out_lats.into_pyarray(py),
        out_lngs.into_pyarray(py),
        out_counts.into_pyarray(py),
        out_starts.into_pyarray(py),
        out_ends.into_pyarray(py),
    ))
}

#[pyfunction]
pub(crate) fn location_frequency_indexed_arrow<'py>(
    py: Python<'py>,
    latitudes: PyArray,
    longitudes: PyArray,
    indices: PyReadonlyArray1<'py, usize>,
    starts: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<LocFreqArrow<'py>> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    let (out_lats, out_lngs, out_counts, out_starts, out_ends) = location_frequency_indexed_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        indices.as_slice()?,
        &ranges,
    )?;
    Ok((
        f64_results_into_arrow(out_lats),
        f64_results_into_arrow(out_lngs),
        u64_results_into_arrow(out_counts),
        out_starts.into_pyarray(py),
        out_ends.into_pyarray(py),
    ))
}
