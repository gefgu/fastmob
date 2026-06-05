use crate::utils::primitive_option_values;
use arrow_array::{
    Array, Int32Array, Int64Array, LargeStringArray, StringArray, UInt32Array, UInt64Array,
    types::{Int32Type, Int64Type, UInt32Type, UInt64Type},
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;
use skmob2_core::measures::individual::radius_of_gyration::{
    UserIndexRanges, radius_of_gyration_batch_impl, radius_of_gyration_batch_with_counts_impl,
    radius_of_gyration_indexed_impl, split_user_index_ranges, user_indices_for_f64_values,
    user_indices_for_f64_values_at_indices, user_indices_for_ord_values,
    user_indices_for_ord_values_at_indices, valid_coord_indices,
};

use crate::utils::{as_nullable_f64_array, f64_results_into_arrow};

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
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(radius_of_gyration_indexed_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        indices.as_slice()?,
        ends.as_slice()?,
    )
    .map_err(PyValueError::new_err)?
    .into_pyarray(py))
}

#[pyfunction]
pub fn radius_of_gyration_user_indices_numpy<'py>(
    py: Python<'py>,
    uids: &Bound<'py, PyAny>,
) -> PyResult<PyUserIndexRanges<'py>> {
    if let Ok(array) = uids.extract::<PyReadonlyArray1<i64>>() {
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_ord_values(array.as_slice()?),
        ));
    }
    if let Ok(array) = uids.extract::<PyReadonlyArray1<i32>>() {
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_ord_values(array.as_slice()?),
        ));
    }
    if let Ok(array) = uids.extract::<PyReadonlyArray1<u64>>() {
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_ord_values(array.as_slice()?),
        ));
    }
    if let Ok(array) = uids.extract::<PyReadonlyArray1<u32>>() {
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_ord_values(array.as_slice()?),
        ));
    }
    if let Ok(array) = uids.extract::<PyReadonlyArray1<f64>>() {
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_f64_values(array.as_slice()?),
        ));
    }

    Err(PyValueError::new_err(
        "unsupported numpy uid dtype for indexed radius_of_gyration",
    ))
}

#[pyfunction]
pub fn radius_of_gyration_valid_user_indices_numpy<'py>(
    py: Python<'py>,
    uids: &Bound<'py, PyAny>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
) -> PyResult<PyUserIndexRanges<'py>> {
    let latitudes = latitudes.as_slice()?;
    let longitudes = longitudes.as_slice()?;
    let valid_indices =
        valid_coord_indices(latitudes, longitudes).map_err(PyValueError::new_err)?;

    if let Ok(array) = uids.extract::<PyReadonlyArray1<i64>>() {
        if array.len()? != latitudes.len() {
            return Err(PyValueError::new_err(
                "uids and coordinates must have the same length",
            ));
        }
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_ord_values_at_indices(array.as_slice()?, valid_indices),
        ));
    }
    if let Ok(array) = uids.extract::<PyReadonlyArray1<i32>>() {
        if array.len()? != latitudes.len() {
            return Err(PyValueError::new_err(
                "uids and coordinates must have the same length",
            ));
        }
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_ord_values_at_indices(array.as_slice()?, valid_indices),
        ));
    }
    if let Ok(array) = uids.extract::<PyReadonlyArray1<u64>>() {
        if array.len()? != latitudes.len() {
            return Err(PyValueError::new_err(
                "uids and coordinates must have the same length",
            ));
        }
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_ord_values_at_indices(array.as_slice()?, valid_indices),
        ));
    }
    if let Ok(array) = uids.extract::<PyReadonlyArray1<u32>>() {
        if array.len()? != latitudes.len() {
            return Err(PyValueError::new_err(
                "uids and coordinates must have the same length",
            ));
        }
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_ord_values_at_indices(array.as_slice()?, valid_indices),
        ));
    }
    if let Ok(array) = uids.extract::<PyReadonlyArray1<f64>>() {
        if array.len()? != latitudes.len() {
            return Err(PyValueError::new_err(
                "uids and coordinates must have the same length",
            ));
        }
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_f64_values_at_indices(array.as_slice()?, valid_indices),
        ));
    }

    Err(PyValueError::new_err(
        "unsupported numpy uid dtype for indexed radius_of_gyration",
    ))
}

#[pyfunction]
pub fn radius_of_gyration_arrow(
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    ranges: Vec<(usize, usize)>,
) -> PyResult<ArrowPyArray> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }

    let n = latitudes.len();
    for &(start, end) in &ranges {
        if start > end {
            return Err(PyValueError::new_err(
                "range start must be less than or equal to range end",
            ));
        }
        if end > n {
            return Err(PyValueError::new_err(
                "range end must be within coordinate array bounds",
            ));
        }
    }

    let mut valid_latitudes = Vec::new();
    let mut valid_longitudes = Vec::new();
    let mut valid_ranges = Vec::with_capacity(ranges.len());
    for &(start, end) in &ranges {
        let valid_start = valid_latitudes.len();
        for idx in start..end {
            if !latitudes.is_null(idx) && !longitudes.is_null(idx) {
                let lat = latitudes.value(idx);
                let lng = longitudes.value(idx);
                if !lat.is_nan() && !lng.is_nan() {
                    valid_latitudes.push(lat);
                    valid_longitudes.push(lng);
                }
            }
        }
        valid_ranges.push((valid_start, valid_latitudes.len()));
    }

    Ok(f64_results_into_arrow(
        radius_of_gyration_batch_impl(&valid_latitudes, &valid_longitudes, &valid_ranges)
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
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }

    let n = latitudes.len();
    for &(start, end) in &ranges {
        if start > end {
            return Err(PyValueError::new_err(
                "range start must be less than or equal to range end",
            ));
        }
        if end > n {
            return Err(PyValueError::new_err(
                "range end must be within coordinate array bounds",
            ));
        }
    }

    let mut valid_latitudes = Vec::new();
    let mut valid_longitudes = Vec::new();
    let mut valid_ranges = Vec::with_capacity(ranges.len());
    let mut valid_counts = Vec::with_capacity(ranges.len());
    for &(start, end) in &ranges {
        let valid_start = valid_latitudes.len();
        for idx in start..end {
            if !latitudes.is_null(idx) && !longitudes.is_null(idx) {
                let lat = latitudes.value(idx);
                let lng = longitudes.value(idx);
                if !lat.is_nan() && !lng.is_nan() {
                    valid_latitudes.push(lat);
                    valid_longitudes.push(lng);
                }
            }
        }
        let valid_end = valid_latitudes.len();
        valid_ranges.push((valid_start, valid_end));
        valid_counts.push(valid_end - valid_start);
    }

    Ok((
        f64_results_into_arrow(
            radius_of_gyration_batch_impl(&valid_latitudes, &valid_longitudes, &valid_ranges)
                .map_err(PyValueError::new_err)?,
        ),
        valid_counts.into_pyarray(py),
    ))
}

#[pyfunction]
pub fn radius_of_gyration_user_indices_arrow<'py>(
    py: Python<'py>,
    uids: ArrowPyArray,
) -> PyResult<PyUserIndexRanges<'py>> {
    let (array_ref, _field) = uids.into_inner();
    let array = array_ref.as_any();

    if let Some(array) = array.downcast_ref::<Int64Array>() {
        let values = primitive_option_values::<Int64Type>(array);
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_ord_values(&values),
        ));
    }
    if let Some(array) = array.downcast_ref::<Int32Array>() {
        let values = primitive_option_values::<Int32Type>(array);
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_ord_values(&values),
        ));
    }
    if let Some(array) = array.downcast_ref::<UInt64Array>() {
        let values = primitive_option_values::<UInt64Type>(array);
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_ord_values(&values),
        ));
    }
    if let Some(array) = array.downcast_ref::<UInt32Array>() {
        let values = primitive_option_values::<UInt32Type>(array);
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_ord_values(&values),
        ));
    }
    if let Some(array) = array.downcast_ref::<StringArray>() {
        let values: Vec<Option<&str>> = (0..array.len())
            .map(|idx| {
                if array.is_null(idx) {
                    None
                } else {
                    Some(array.value(idx))
                }
            })
            .collect();
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_ord_values(&values),
        ));
    }
    if let Some(array) = array.downcast_ref::<LargeStringArray>() {
        let values: Vec<Option<&str>> = (0..array.len())
            .map(|idx| {
                if array.is_null(idx) {
                    None
                } else {
                    Some(array.value(idx))
                }
            })
            .collect();
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_ord_values(&values),
        ));
    }

    Err(PyValueError::new_err(
        "unsupported Arrow uid type for indexed radius_of_gyration",
    ))
}

#[pyfunction]
pub fn radius_of_gyration_valid_user_indices_arrow<'py>(
    py: Python<'py>,
    uids: ArrowPyArray,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
) -> PyResult<PyUserIndexRanges<'py>> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }

    let valid_indices: Vec<usize> = (0..latitudes.len())
        .filter(|&idx| {
            !latitudes.is_null(idx)
                && !longitudes.is_null(idx)
                && !latitudes.value(idx).is_nan()
                && !longitudes.value(idx).is_nan()
        })
        .collect();

    let (array_ref, _field) = uids.into_inner();
    if array_ref.len() != latitudes.len() {
        return Err(PyValueError::new_err(
            "uids and coordinates must have the same length",
        ));
    }
    let array = array_ref.as_any();

    if let Some(array) = array.downcast_ref::<Int64Array>() {
        let values = primitive_option_values::<Int64Type>(array);
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_ord_values_at_indices(&values, valid_indices),
        ));
    }
    if let Some(array) = array.downcast_ref::<Int32Array>() {
        let values = primitive_option_values::<Int32Type>(array);
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_ord_values_at_indices(&values, valid_indices),
        ));
    }
    if let Some(array) = array.downcast_ref::<UInt64Array>() {
        let values = primitive_option_values::<UInt64Type>(array);
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_ord_values_at_indices(&values, valid_indices),
        ));
    }
    if let Some(array) = array.downcast_ref::<UInt32Array>() {
        let values = primitive_option_values::<UInt32Type>(array);
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_ord_values_at_indices(&values, valid_indices),
        ));
    }
    if let Some(array) = array.downcast_ref::<StringArray>() {
        let values: Vec<Option<&str>> = (0..array.len())
            .map(|idx| {
                if array.is_null(idx) {
                    None
                } else {
                    Some(array.value(idx))
                }
            })
            .collect();
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_ord_values_at_indices(&values, valid_indices),
        ));
    }
    if let Some(array) = array.downcast_ref::<LargeStringArray>() {
        let values: Vec<Option<&str>> = (0..array.len())
            .map(|idx| {
                if array.is_null(idx) {
                    None
                } else {
                    Some(array.value(idx))
                }
            })
            .collect();
        return Ok(user_index_ranges_into_numpy(
            py,
            user_indices_for_ord_values_at_indices(&values, valid_indices),
        ));
    }

    Err(PyValueError::new_err(
        "unsupported Arrow uid type for indexed radius_of_gyration",
    ))
}

#[pyfunction]
pub fn radius_of_gyration_indexed_arrow(
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    indices: PyReadonlyArray1<usize>,
    ends: PyReadonlyArray1<usize>,
) -> PyResult<ArrowPyArray> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }

    let lat_values: Vec<f64> = (0..latitudes.len())
        .map(|idx| {
            if latitudes.is_null(idx) {
                f64::NAN
            } else {
                latitudes.value(idx)
            }
        })
        .collect();
    let lng_values: Vec<f64> = (0..longitudes.len())
        .map(|idx| {
            if longitudes.is_null(idx) {
                f64::NAN
            } else {
                longitudes.value(idx)
            }
        })
        .collect();

    Ok(f64_results_into_arrow(
        radius_of_gyration_indexed_impl(
            &lat_values,
            &lng_values,
            indices.as_slice()?,
            ends.as_slice()?,
        )
        .map_err(PyValueError::new_err)?,
    ))
}
