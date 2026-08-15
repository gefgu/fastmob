use crate::utils::ArrowUsizeArrayExt;
use fastmob_core::measures::individual::spatial_counts::{
    number_of_locations_from_ends_impl, number_of_locations_indexed_impl,
    number_of_visits_from_ends_impl, number_of_visits_indexed_impl,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::adapters::trajectory::{run_indexed_coordinate_arrow, run_presorted_coordinate_arrow};
use crate::utils::u64_results_into_arrow;

fn arrow_u64_output(py: Python<'_>, values: Vec<u64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, u64_results_into_arrow(values))?.into_any())
}

#[pyfunction]
pub fn number_of_visits_presorted<'py>(
    py: Python<'py>,
    n_values: usize,
    ends: pyo3_arrow::PyArray,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    Ok(number_of_visits_from_ends_impl(n_values, &ends.as_slice()?)
        .map_err(PyValueError::new_err)?
        .into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (n_values, indices, ends, valid_rows = None))]
pub fn number_of_visits_indexed<'py>(
    py: Python<'py>,
    n_values: usize,
    indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
    valid_rows: Option<PyReadonlyArray1<'py, bool>>,
) -> PyResult<Bound<'py, PyArray1<u64>>> {
    let valid_slice: Option<&[bool]> = if let Some(ref v) = valid_rows {
        Some(v.as_slice()?)
    } else {
        None
    };
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;
    Ok(
        number_of_visits_indexed_impl(n_values, &indices, &ends, valid_slice)
            .map_err(PyValueError::new_err)?
            .into_pyarray(py),
    )
}

#[pyfunction]
pub fn number_of_locations_presorted<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
) -> PyResult<Py<PyAny>> {
    let values =
        run_presorted_coordinate_arrow(py, latitudes, longitudes, ends, |coords, ends| {
            number_of_locations_from_ends_impl(coords.latitudes, coords.longitudes, ends)
        })?
        .map_err(PyValueError::new_err)?;
    arrow_u64_output(py, values)
}

#[pyfunction]
pub fn number_of_locations_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
) -> PyResult<Py<PyAny>> {
    let values = run_indexed_coordinate_arrow(py, latitudes, longitudes, indices, ends, |view| {
        number_of_locations_indexed_impl(
            view.coordinates.latitudes,
            view.coordinates.longitudes,
            view.indices,
            view.ends,
            view.valid_rows,
        )
    })?
    .map_err(PyValueError::new_err)?;
    arrow_u64_output(py, values)
}
