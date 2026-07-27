use fastmob_core::measures::individual::radius_of_gyration::{
    radius_of_gyration_indexed_impl, radius_of_gyration_presorted_impl,
};
use fastmob_core::utils::validate_ends;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, f64_results_into_arrow,
    validate_indexed_ends,
};

type PyRogResult<'py> = (Py<PyAny>, Bound<'py, PyArray1<bool>>);

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

fn validate_coordinate_lengths(latitudes_len: usize, longitudes_len: usize) -> PyResult<()> {
    if latitudes_len != longitudes_len {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }
    Ok(())
}

#[pyfunction]
pub fn radius_of_gyration_presorted<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyRogResult<'py>> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    validate_coordinate_lengths(latitudes.len(), longitudes.len())?;
    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    let ends = ends.as_slice()?;
    validate_ends(lats.len(), ends).map_err(PyValueError::new_err)?;

    let (values, validity) = py.detach(|| radius_of_gyration_presorted_impl(lats, lngs, ends));
    Ok((arrow_f64_output(py, values)?, validity.into_pyarray(py)))
}

#[pyfunction]
pub fn radius_of_gyration_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyRogResult<'py>> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    validate_coordinate_lengths(latitudes.len(), longitudes.len())?;
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;
    validate_indexed_ends(latitudes.len(), indices, ends)?;

    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    let (values, validity) = py.detach(|| {
        radius_of_gyration_indexed_impl(lats, lngs, indices, ends, valid_rows.as_deref())
    });
    Ok((arrow_f64_output(py, values)?, validity.into_pyarray(py)))
}
