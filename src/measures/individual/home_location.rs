use std::collections::HashMap;

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use rayon::prelude::*;

use crate::utils::{
    arrow_values, as_f64_array, f64_results_into_arrow, ranges_from_starts_ends,
    validate_coord_ranges, validate_indexed_coord_ranges,
};

type HomeResults = (Vec<f64>, Vec<f64>);
type PyHomeResults<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

struct HomeInputs<'a> {
    latitudes: &'a [f64],
    longitudes: &'a [f64],
    hours: &'a [f64],
}

struct NightWindow {
    start: f64,
    end: f64,
}

fn best_location_for_indices(
    inputs: &HomeInputs<'_>,
    indices: &[usize],
    start: usize,
    end: usize,
    night: &NightWindow,
) -> (f64, f64) {
    let has_night = indices[start..end]
        .iter()
        .any(|&idx| inputs.hours[idx] >= night.start || inputs.hours[idx] < night.end);
    let mut counts: HashMap<(u64, u64), (f64, f64, u64)> = HashMap::new();

    for &idx in &indices[start..end] {
        let is_night = inputs.hours[idx] >= night.start || inputs.hours[idx] < night.end;
        if has_night && !is_night {
            continue;
        }
        let lat = inputs.latitudes[idx];
        let lng = inputs.longitudes[idx];
        let entry = counts
            .entry((lat.to_bits(), lng.to_bits()))
            .or_insert((lat, lng, 0));
        entry.2 += 1;
    }

    counts
        .into_values()
        .max_by(|left, right| {
            left.2
                .cmp(&right.2)
                .then_with(|| right.0.total_cmp(&left.0))
                .then_with(|| right.1.total_cmp(&left.1))
        })
        .map(|(lat, lng, _)| (lat, lng))
        .unwrap_or((f64::NAN, f64::NAN))
}

fn best_location_for_range(
    inputs: &HomeInputs<'_>,
    start: usize,
    end: usize,
    night: &NightWindow,
) -> (f64, f64) {
    let has_night =
        (start..end).any(|idx| inputs.hours[idx] >= night.start || inputs.hours[idx] < night.end);
    let mut counts: HashMap<(u64, u64), (f64, f64, u64)> = HashMap::new();

    for idx in start..end {
        let is_night = inputs.hours[idx] >= night.start || inputs.hours[idx] < night.end;
        if has_night && !is_night {
            continue;
        }
        let lat = inputs.latitudes[idx];
        let lng = inputs.longitudes[idx];
        let entry = counts
            .entry((lat.to_bits(), lng.to_bits()))
            .or_insert((lat, lng, 0));
        entry.2 += 1;
    }

    counts
        .into_values()
        .max_by(|left, right| {
            left.2
                .cmp(&right.2)
                .then_with(|| right.0.total_cmp(&left.0))
                .then_with(|| right.1.total_cmp(&left.1))
        })
        .map(|(lat, lng, _)| (lat, lng))
        .unwrap_or((f64::NAN, f64::NAN))
}

fn home_location_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    hours: &[f64],
    ranges: &[(usize, usize)],
    start_night: f64,
    end_night: f64,
) -> PyResult<HomeResults> {
    validate_coord_ranges(latitudes, longitudes, ranges)?;
    if hours.len() != latitudes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "hours, latitudes, and longitudes must have the same length",
        ));
    }

    let inputs = HomeInputs {
        latitudes,
        longitudes,
        hours,
    };
    let night = NightWindow {
        start: start_night,
        end: end_night,
    };
    let homes: Vec<(f64, f64)> = ranges
        .par_iter()
        .map(|&(start, end)| best_location_for_range(&inputs, start, end, &night))
        .collect();
    Ok(homes.into_iter().unzip())
}

fn home_location_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    hours: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
    start_night: f64,
    end_night: f64,
) -> PyResult<HomeResults> {
    validate_indexed_coord_ranges(latitudes, longitudes, indices, ranges)?;
    if hours.len() != latitudes.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "hours, latitudes, and longitudes must have the same length",
        ));
    }

    let inputs = HomeInputs {
        latitudes,
        longitudes,
        hours,
    };
    let night = NightWindow {
        start: start_night,
        end: end_night,
    };
    let homes: Vec<(f64, f64)> = ranges
        .par_iter()
        .map(|&(start, end)| best_location_for_indices(&inputs, indices, start, end, &night))
        .collect();
    Ok(homes.into_iter().unzip())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn home_location_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    hours: PyReadonlyArray1<'py, f64>,
    ranges: Vec<(usize, usize)>,
    start_night: f64,
    end_night: f64,
) -> PyResult<PyHomeResults<'py>> {
    let (home_lats, home_lngs) = home_location_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        hours.as_slice()?,
        &ranges,
        start_night,
        end_night,
    )?;
    Ok((home_lats.into_pyarray(py), home_lngs.into_pyarray(py)))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn home_location_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    hours: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    starts: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    start_night: f64,
    end_night: f64,
) -> PyResult<PyHomeResults<'py>> {
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    let (home_lats, home_lngs) = home_location_indexed_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        hours.as_slice()?,
        indices.as_slice()?,
        &ranges,
        start_night,
        end_night,
    )?;
    Ok((home_lats.into_pyarray(py), home_lngs.into_pyarray(py)))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn home_location_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    hours: PyArray,
    ranges: Vec<(usize, usize)>,
    start_night: f64,
    end_night: f64,
) -> PyResult<(PyArray, PyArray)> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let hours = as_f64_array(hours, "hours")?;
    let (home_lats, home_lngs) = home_location_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        arrow_values(&hours),
        &ranges,
        start_night,
        end_night,
    )?;
    Ok((
        f64_results_into_arrow(home_lats),
        f64_results_into_arrow(home_lngs),
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub(crate) fn home_location_indexed_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    hours: PyArray,
    indices: PyReadonlyArray1<usize>,
    starts: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
    start_night: f64,
    end_night: f64,
) -> PyResult<(PyArray, PyArray)> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let hours = as_f64_array(hours, "hours")?;
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    let (home_lats, home_lngs) = home_location_indexed_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        arrow_values(&hours),
        indices.as_slice()?,
        &ranges,
        start_night,
        end_night,
    )?;
    Ok((
        f64_results_into_arrow(home_lats),
        f64_results_into_arrow(home_lngs),
    ))
}
