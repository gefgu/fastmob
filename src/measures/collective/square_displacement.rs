use numpy::PyReadonlyArray1;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::haversine::haversine_km;
use crate::utils::{
    arrow_values, as_f64_array, ranges_from_starts_ends, validate_indexed_coord_ranges,
};

#[pyfunction]
pub(crate) fn square_displacement_km2(lat0: f64, lng0: f64, lat_t: f64, lng_t: f64) -> f64 {
    let d = haversine_km(lat0, lng0, lat_t, lng_t);
    d * d
}

fn mean_square_displacement_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps_s: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
    delta_s: f64,
) -> PyResult<f64> {
    validate_indexed_coord_ranges(latitudes, longitudes, indices, ranges)?;
    if latitudes.len() != timestamps_s.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "timestamps_s must have the same length as latitudes and longitudes",
        ));
    }

    let mut total = 0.0f64;
    let mut count = 0usize;

    for &(start, end) in ranges {
        if start >= end {
            continue;
        }
        let first_idx = indices[start];
        let t_limit = timestamps_s[first_idx] + delta_s;

        // indices[start..end] are in chronological order (time-ordered ranges)
        let mut rt_idx = first_idx;
        for &idx in &indices[start..end] {
            if timestamps_s[idx] <= t_limit {
                rt_idx = idx;
            } else {
                break;
            }
        }

        let d = haversine_km(
            latitudes[first_idx],
            longitudes[first_idx],
            latitudes[rt_idx],
            longitudes[rt_idx],
        );
        total += d * d;
        count += 1;
    }

    if count == 0 {
        Ok(0.0)
    } else {
        Ok(total / count as f64)
    }
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub(crate) fn mean_square_displacement_indexed_numpy<'py>(
    _py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    timestamps_s: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    starts: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    delta_s: f64,
) -> PyResult<f64> {
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    mean_square_displacement_indexed_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        timestamps_s.as_slice()?,
        indices.as_slice()?,
        &ranges,
        delta_s,
    )
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub(crate) fn mean_square_displacement_indexed_arrow<'py>(
    _py: Python<'py>,
    latitudes: PyArray,
    longitudes: PyArray,
    timestamps_s: PyArray,
    indices: PyReadonlyArray1<'py, usize>,
    starts: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    delta_s: f64,
) -> PyResult<f64> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let timestamps_s = as_f64_array(timestamps_s, "timestamps_s")?;
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    mean_square_displacement_indexed_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        arrow_values(&timestamps_s),
        indices.as_slice()?,
        &ranges,
        delta_s,
    )
}
