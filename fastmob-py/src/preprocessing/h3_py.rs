use arrow_array::Array;
use fastmob_core::preprocessing::h3::{
    batch_cells_to_latlng, batch_latlng_to_cells, batch_latlng_to_h3_centered, INVALID_CELL,
};
use geo::{LineString, Polygon};
use h3o::geom::TilerBuilder;
use h3o::{CellIndex, Resolution};
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

fn polygon_from_rings(rings: Vec<Vec<(f64, f64)>>) -> Result<Polygon<f64>, String> {
    let mut rings = rings.into_iter();
    let exterior = rings
        .next()
        .ok_or_else(|| "polygon must have an exterior ring".to_owned())?;
    if exterior.len() < 4 {
        return Err("polygon exterior ring must have at least four coordinates".to_owned());
    }
    let interiors = rings
        .map(|ring| {
            if ring.len() < 4 {
                Err("polygon interior ring must have at least four coordinates".to_owned())
            } else {
                Ok(LineString::from(ring))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Polygon::new(LineString::from(exterior), interiors))
}

/// Cover GeoJSON-style polygons with H3 cells using Rust ``h3o``.
///
/// ``polygons`` is a multipolygon-shaped list: polygons, rings, then
/// ``(longitude, latitude)`` coordinate pairs. Keeping the interchange shape
/// free of GeoPandas/Shapely types lets the GIL be released for the H3 work.
#[pyfunction]
pub fn h3_polygons_to_cells(
    py: Python<'_>,
    polygons: Vec<Vec<Vec<(f64, f64)>>>,
    resolution: u8,
) -> PyResult<Vec<u64>> {
    let resolution = resolve_resolution(resolution)?;
    let result: Result<Vec<u64>, String> = py.detach(|| {
        let mut tiler = TilerBuilder::new(resolution).build();
        for polygon in polygons {
            tiler
                .add(polygon_from_rings(polygon)?)
                .map_err(|err| err.to_string())?;
        }
        let mut cells: Vec<u64> = tiler.into_coverage().map(u64::from).collect();
        cells.sort_unstable();
        cells.dedup();
        Ok(cells)
    });
    result.map_err(PyValueError::new_err)
}

/// Return one ``(longitude, latitude)`` ring per H3 cell using Rust ``h3o``.
#[pyfunction]
pub fn h3_cells_to_boundaries(py: Python<'_>, cells: Vec<u64>) -> PyResult<Vec<Vec<(f64, f64)>>> {
    let result: Result<Vec<Vec<(f64, f64)>>, String> = py.detach(|| {
        cells
            .into_iter()
            .map(|cell| {
                let cell = CellIndex::try_from(cell).map_err(|err| err.to_string())?;
                let mut ring: Vec<(f64, f64)> = cell
                    .boundary()
                    .iter()
                    .map(|point| (point.lng(), point.lat()))
                    .collect();
                if let Some(first) = ring.first().copied() {
                    ring.push(first);
                }
                Ok(ring)
            })
            .collect()
    });
    result.map_err(PyValueError::new_err)
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

/// Convert `(lat, lng)` pairs directly to `(cell, center_lat, center_lng)` in
/// one pass, for callers that need both instead of chaining
/// [`latlng_to_h3_arrow`] and [`h3_to_latlng_arrow`].
#[pyfunction]
pub fn latlng_to_h3_centered_arrow(
    py: Python<'_>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    resolution: u8,
) -> PyResult<(ArrowPyArray, ArrowPyArray, ArrowPyArray)> {
    let resolution = resolve_resolution(resolution)?;
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;

    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);

    let (cells, center_lats, center_lngs) =
        py.detach(|| batch_latlng_to_h3_centered(lats, lngs, resolution, valid_rows.as_deref()));

    Ok((
        u64_results_into_arrow_nullable(cells, INVALID_CELL),
        f64_results_into_arrow_nullable(center_lats),
        f64_results_into_arrow_nullable(center_lngs),
    ))
}
