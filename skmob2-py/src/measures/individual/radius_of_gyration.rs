use arrow_array::{Array, UInt64Array};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;
use skmob2_core::measures::individual::radius_of_gyration::{
    UserIndexRanges, radius_of_gyration_batch_impl, radius_of_gyration_batch_with_counts_impl,
    radius_of_gyration_indexed_impl, radius_of_gyration_indexed_with_valid_rows_impl,
    split_user_index_ranges, user_indices_for_u64_codes,
};

use crate::utils::{
    arrow_valid_rows, arrow_values, as_f64_array, as_nullable_f64_array, f64_results_into_arrow,
};

type PyUserIndexRanges<'py> = (Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>);
type PyRadiusOfGyrationWithCounts<'py> = (Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<usize>>);

fn user_index_ranges_into_numpy<'py>(
    py: Python<'py>,
    (indices, ranges): UserIndexRanges,
) -> PyUserIndexRanges<'py> {
    let (indices, ends) = split_user_index_ranges((indices, ranges));
    (indices.into_pyarray(py), ends.into_pyarray(py))
}

#[pyfunction]
pub fn radius_of_gyration_km(coords: Vec<(f64, f64)>) -> PyResult<f64> {
    use skmob2_core::measures::individual::radius_of_gyration::rog_for_slice;
    Ok(rog_for_slice(&coords))
}

#[pyfunction]
pub fn radius_of_gyration_batch_km(
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    radius_of_gyration_batch_impl(&latitudes, &longitudes, &ranges).map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn radius_of_gyration_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(
        radius_of_gyration_batch_impl(latitudes.as_slice()?, longitudes.as_slice()?, &ranges)
            .map_err(PyValueError::new_err)?
            .into_pyarray(py),
    )
}

#[pyfunction]
pub fn radius_of_gyration_numpy_with_counts<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<PyRadiusOfGyrationWithCounts<'py>> {
    let (values, counts) = radius_of_gyration_batch_with_counts_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        &ranges,
    )
    .map_err(PyValueError::new_err)?;
    Ok((values.into_pyarray(py), counts.into_pyarray(py)))
}

#[pyfunction]
pub fn radius_of_gyration_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<PyRadiusOfGyrationWithCounts<'py>> {
    let (values, counts) = radius_of_gyration_indexed_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        indices.as_slice()?,
        ends.as_slice()?,
    )
    .map_err(PyValueError::new_err)?;
    Ok((values.into_pyarray(py), counts.into_pyarray(py)))
}

#[pyfunction]
pub fn radius_of_gyration_user_indices_numpy<'py>(
    py: Python<'py>,
    uids: PyReadonlyArray1<'py, u64>,
    num_groups: usize,
) -> PyResult<PyUserIndexRanges<'py>> {
    Ok(user_index_ranges_into_numpy(
        py,
        user_indices_for_u64_codes(uids.as_slice()?, num_groups).map_err(PyValueError::new_err)?,
    ))
}

#[pyfunction]
pub fn radius_of_gyration_arrow(
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    ranges: Vec<(usize, usize)>,
) -> PyResult<ArrowPyArray> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;

    Ok(f64_results_into_arrow(
        radius_of_gyration_batch_impl(arrow_values(&latitudes), arrow_values(&longitudes), &ranges)
            .map_err(PyValueError::new_err)?,
    ))
}

#[pyfunction]
pub fn radius_of_gyration_arrow_with_counts<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    ranges: Vec<(usize, usize)>,
) -> PyResult<(ArrowPyArray, Bound<'py, PyArray1<usize>>)> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let (values, counts) = radius_of_gyration_batch_with_counts_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        &ranges,
    )
    .map_err(PyValueError::new_err)?;

    Ok((f64_results_into_arrow(values), counts.into_pyarray(py)))
}

#[pyfunction]
pub fn radius_of_gyration_user_indices_arrow<'py>(
    py: Python<'py>,
    uids: ArrowPyArray,
    num_groups: usize,
) -> PyResult<PyUserIndexRanges<'py>> {
    let (array_ref, _field) = uids.into_inner();
    let array = array_ref.as_any();

    if let Some(array) = array.downcast_ref::<UInt64Array>() {
        if array.null_count() > 0 {
            return Err(PyValueError::new_err(
                "uint64 uid codes for indexed radius_of_gyration must not contain nulls",
            ));
        }
        let start = array.offset();
        let end = start + array.len();
        let values = &array.values()[start..end];
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_u64_codes(values, num_groups).map_err(PyValueError::new_err)?,
        ));
    }

    Err(PyValueError::new_err(
        "expected uint64 Arrow uid codes for indexed radius_of_gyration",
    ))
}

#[pyfunction]
pub fn radius_of_gyration_indexed_arrow<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    indices: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<(ArrowPyArray, Bound<'py, PyArray1<usize>>)> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }

    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    let (values, counts) = radius_of_gyration_indexed_with_valid_rows_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        indices.as_slice()?,
        ends.as_slice()?,
        valid_rows.as_deref(),
    )
    .map_err(PyValueError::new_err)?;

    Ok((f64_results_into_arrow(values), counts.into_pyarray(py)))
}
