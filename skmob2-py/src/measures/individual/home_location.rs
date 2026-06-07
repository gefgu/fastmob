use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use skmob2_core::measures::individual::home_location::{
    home_location_from_ends_impl, home_location_indexed_impl,
};

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, f64_results_into_arrow,
};

type PyHomeResults<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>);

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn home_location_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    hours: PyReadonlyArray1<'py, f64>,
    ends: PyReadonlyArray1<'py, usize>,
    start_night: f64,
    end_night: f64,
) -> PyResult<PyHomeResults<'py>> {
    let (home_lats, home_lngs) = home_location_from_ends_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        hours.as_slice()?,
        ends.as_slice()?,
        start_night,
        end_night,
    )
    .map_err(PyValueError::new_err)?;
    Ok((home_lats.into_pyarray(py), home_lngs.into_pyarray(py)))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn home_location_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    hours: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    start_night: f64,
    end_night: f64,
) -> PyResult<PyHomeResults<'py>> {
    let (home_lats, home_lngs) = home_location_indexed_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        hours.as_slice()?,
        indices.as_slice()?,
        ends.as_slice()?,
        start_night,
        end_night,
        None,
    )
    .map_err(PyValueError::new_err)?;
    Ok((home_lats.into_pyarray(py), home_lngs.into_pyarray(py)))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn home_location_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    hours: PyArray,
    ends: PyReadonlyArray1<usize>,
    start_night: f64,
    end_night: f64,
) -> PyResult<(PyArray, PyArray)> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let hours = as_f64_array(hours, "hours")?;
    let (home_lats, home_lngs) = home_location_from_ends_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        arrow_values(&hours),
        ends.as_slice()?,
        start_night,
        end_night,
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        f64_results_into_arrow(home_lats),
        f64_results_into_arrow(home_lngs),
    ))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn home_location_indexed_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    hours: PyArray,
    indices: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
    start_night: f64,
    end_night: f64,
) -> PyResult<(PyArray, PyArray)> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let hours = as_nullable_f64_array(hours, "hours")?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes, &hours]);
    let (home_lats, home_lngs) = home_location_indexed_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        arrow_values(&hours),
        indices.as_slice()?,
        ends.as_slice()?,
        start_night,
        end_night,
        valid_rows.as_deref(),
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        f64_results_into_arrow(home_lats),
        f64_results_into_arrow(home_lngs),
    ))
}
