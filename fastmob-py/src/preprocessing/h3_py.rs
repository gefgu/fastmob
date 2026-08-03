use arrow_array::Array;
use fastmob_core::preprocessing::h3::{batch_cells_to_latlng, batch_latlng_to_cells, INVALID_CELL};
use h3o::Resolution;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_u64_values, arrow_valid_rows, arrow_values, as_nullable_f64_array, as_nullable_u64_array,
    f64_results_into_arrow_nullable, u64_results_into_arrow_nullable,
};

fn resolve_resolution(resolution: u8) -> PyResult<Resolution> {
    Resolution::try_from(resolution).map_err(|_| {
        PyValueError::new_err(format!(
            "H3 resolution must be between 0 and 15, got {resolution}"
        ))
    })
}

#[pyfunction]
pub fn h3_to_latlng_arrow(
    py: Python<'_>,
    cells: ArrowPyArray,
) -> PyResult<(ArrowPyArray, ArrowPyArray)> {
    let cells = as_nullable_u64_array(cells, "cells")?;
    let values = arrow_u64_values(&cells);
    let valid_rows: Option<Vec<bool>> =
        (cells.null_count() > 0).then(|| (0..cells.len()).map(|idx| cells.is_valid(idx)).collect());
    let (lats, lngs) = py.detach(|| batch_cells_to_latlng(values, valid_rows.as_deref()));
    Ok((
        f64_results_into_arrow_nullable(lats),
        f64_results_into_arrow_nullable(lngs),
    ))
}

#[pyfunction]
pub fn latlng_to_h3_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    resolution: u8,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let resolution = resolve_resolution(resolution)?;
    let lats = latitudes.as_slice()?;
    let lngs = longitudes.as_slice()?;

    let cells = py.detach(|| batch_latlng_to_cells(lats, lngs, resolution, None));

    Ok(PyArray1::from_vec(py, cells))
}

#[pyfunction]
pub fn latlng_to_h3_arrow(
    py: Python<'_>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    resolution: u8,
) -> PyResult<ArrowPyArray> {
    let resolution = resolve_resolution(resolution)?;
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;

    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);

    let cells = py.detach(|| batch_latlng_to_cells(lats, lngs, resolution, valid_rows.as_deref()));

    Ok(u64_results_into_arrow_nullable(cells, INVALID_CELL))
}
