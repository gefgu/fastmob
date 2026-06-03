use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use skmob2_core::measures::individual::location_frequency::location_frequency_indexed_impl;

use crate::utils::{
    arrow_values, as_f64_array, f64_results_into_arrow, ranges_from_starts_ends,
    u64_results_into_arrow,
};

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
pub fn location_frequency_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    starts: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<LocFreqNumpy<'py>> {
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    let (out_lats, out_lngs, out_counts, out_starts, out_ends) =
        location_frequency_indexed_impl(
            latitudes.as_slice()?,
            longitudes.as_slice()?,
            indices.as_slice()?,
            &ranges,
        )
        .map_err(PyValueError::new_err)?;
    Ok((
        out_lats.into_pyarray(py),
        out_lngs.into_pyarray(py),
        out_counts.into_pyarray(py),
        out_starts.into_pyarray(py),
        out_ends.into_pyarray(py),
    ))
}

#[pyfunction]
pub fn location_frequency_indexed_arrow<'py>(
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
    let (out_lats, out_lngs, out_counts, out_starts, out_ends) =
        location_frequency_indexed_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            indices.as_slice()?,
            &ranges,
        )
        .map_err(PyValueError::new_err)?;
    Ok((
        f64_results_into_arrow(out_lats),
        f64_results_into_arrow(out_lngs),
        u64_results_into_arrow(out_counts),
        out_starts.into_pyarray(py),
        out_ends.into_pyarray(py),
    ))
}
