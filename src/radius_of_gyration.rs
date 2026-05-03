use arrow_array::{
    Array, ArrayRef, Float64Array, Int32Array, Int64Array, LargeStringArray, PrimitiveArray,
    StringArray, UInt32Array, UInt64Array,
    types::{Int32Type, Int64Type, UInt32Type, UInt64Type},
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;
use rayon::prelude::*;
use std::sync::Arc;

use crate::utils::validate_coord_ranges;

type UserIndexRanges = (Vec<usize>, Vec<(usize, usize)>);
type PyUserIndexRanges<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
);

#[pyfunction]
pub(crate) fn radius_of_gyration_km(coords: Vec<(f64, f64)>) -> PyResult<f64> {
    Ok(rog_for_slice(&coords))
}

pub(crate) fn rog_for_slice(coords: &[(f64, f64)]) -> f64 {
    let n = coords.len();
    if n == 0 {
        return 0.0;
    }

    let (lat_sum, lng_sum) = coords
        .iter()
        .fold((0.0f64, 0.0f64), |(ls, ns), &(lat, lng)| {
            (ls + lat, ns + lng)
        });
    let cm_lat = lat_sum / n as f64;
    let cm_lng = lng_sum / n as f64;

    let cm_lat_rad = cm_lat.to_radians();
    let cm_lng_rad = cm_lng.to_radians();
    let cos_cm_lat = cm_lat_rad.cos();

    let sum_sq: f64 = coords
        .iter()
        .map(|&(lat, lng)| {
            let lat_rad = lat.to_radians();
            let dlat = lat_rad - cm_lat_rad;
            let dlng = lng.to_radians() - cm_lng_rad;
            let a = (dlat / 2.0).sin().powi(2)
                + cos_cm_lat * lat_rad.cos() * (dlng / 2.0).sin().powi(2);
            let d = 2.0 * a.sqrt().asin() * 6371.0088;
            d * d
        })
        .sum();

    (sum_sq / n as f64).sqrt()
}

fn rog_for_parallel_slices(latitudes: &[f64], longitudes: &[f64], start: usize, end: usize) -> f64 {
    let n = end - start;
    if n == 0 {
        return 0.0;
    }

    let (lat_sum, lng_sum) = (start..end).fold((0.0f64, 0.0f64), |(ls, ns), idx| {
        (ls + latitudes[idx], ns + longitudes[idx])
    });
    let cm_lat = lat_sum / n as f64;
    let cm_lng = lng_sum / n as f64;

    let cm_lat_rad = cm_lat.to_radians();
    let cm_lng_rad = cm_lng.to_radians();
    let cos_cm_lat = cm_lat_rad.cos();

    let sum_sq: f64 = (start..end)
        .map(|idx| {
            let lat_rad = latitudes[idx].to_radians();
            let dlat = lat_rad - cm_lat_rad;
            let dlng = longitudes[idx].to_radians() - cm_lng_rad;
            let a = (dlat / 2.0).sin().powi(2)
                + cos_cm_lat * lat_rad.cos() * (dlng / 2.0).sin().powi(2);
            let d = 2.0 * a.sqrt().asin() * 6371.0088;
            d * d
        })
        .sum();

    (sum_sq / n as f64).sqrt()
}

fn rog_for_indexed_slice(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    start: usize,
    end: usize,
) -> f64 {
    let n = end - start;
    if n == 0 {
        return 0.0;
    }

    let (lat_sum, lng_sum) = indices[start..end]
        .iter()
        .fold((0.0f64, 0.0f64), |(ls, ns), &idx| {
            (ls + latitudes[idx], ns + longitudes[idx])
        });
    let cm_lat = lat_sum / n as f64;
    let cm_lng = lng_sum / n as f64;

    let cm_lat_rad = cm_lat.to_radians();
    let cm_lng_rad = cm_lng.to_radians();
    let cos_cm_lat = cm_lat_rad.cos();

    let sum_sq: f64 = indices[start..end]
        .iter()
        .map(|&idx| {
            let lat_rad = latitudes[idx].to_radians();
            let dlat = lat_rad - cm_lat_rad;
            let dlng = longitudes[idx].to_radians() - cm_lng_rad;
            let a = (dlat / 2.0).sin().powi(2)
                + cos_cm_lat * lat_rad.cos() * (dlng / 2.0).sin().powi(2);
            let d = 2.0 * a.sqrt().asin() * 6371.0088;
            d * d
        })
        .sum();

    (sum_sq / n as f64).sqrt()
}

fn validate_indexed_inputs(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> PyResult<()> {
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }

    let n_indices = indices.len();
    for &(start, end) in ranges {
        if start > end {
            return Err(PyValueError::new_err(
                "range start must be less than or equal to range end",
            ));
        }
        if end > n_indices {
            return Err(PyValueError::new_err(
                "range end must be within index array bounds",
            ));
        }
    }

    let n_coords = latitudes.len();
    for &idx in indices {
        if idx >= n_coords {
            return Err(PyValueError::new_err(
                "index must be within coordinate array bounds",
            ));
        }
    }

    Ok(())
}

fn radius_of_gyration_batch_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ranges: &[(usize, usize)],
) -> PyResult<Vec<f64>> {
    validate_coord_ranges(latitudes, longitudes, ranges)?;

    let mut valid_latitudes = Vec::new();
    let mut valid_longitudes = Vec::new();
    let mut valid_ranges = Vec::with_capacity(ranges.len());
    for &(start, end) in ranges {
        let valid_start = valid_latitudes.len();
        for idx in start..end {
            let lat = latitudes[idx];
            let lng = longitudes[idx];
            if !lat.is_nan() && !lng.is_nan() {
                valid_latitudes.push(lat);
                valid_longitudes.push(lng);
            }
        }
        valid_ranges.push((valid_start, valid_latitudes.len()));
    }

    let results: Vec<f64> = valid_ranges
        .par_iter()
        .map(|&(start, end)| {
            rog_for_parallel_slices(&valid_latitudes, &valid_longitudes, start, end)
        })
        .collect();

    Ok(results)
}

fn radius_of_gyration_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    indices: &[usize],
    ranges: &[(usize, usize)],
) -> PyResult<Vec<f64>> {
    validate_indexed_inputs(latitudes, longitudes, indices, ranges)?;

    let mut valid_indices = Vec::new();
    let mut valid_ranges = Vec::with_capacity(ranges.len());
    for &(start, end) in ranges {
        let valid_start = valid_indices.len();
        for &idx in &indices[start..end] {
            if !latitudes[idx].is_nan() && !longitudes[idx].is_nan() {
                valid_indices.push(idx);
            }
        }
        valid_ranges.push((valid_start, valid_indices.len()));
    }

    let results: Vec<f64> = valid_ranges
        .par_iter()
        .map(|&(start, end)| {
            rog_for_indexed_slice(latitudes, longitudes, &valid_indices, start, end)
        })
        .collect();

    Ok(results)
}

fn ranges_from_sorted_indices<T: PartialEq>(
    values: &[T],
    indices: &[usize],
) -> Vec<(usize, usize)> {
    if indices.is_empty() {
        return Vec::new();
    }

    let mut ranges = Vec::new();
    let mut start = 0usize;
    for pos in 1..indices.len() {
        if values[indices[pos]] != values[indices[pos - 1]] {
            ranges.push((start, pos));
            start = pos;
        }
    }
    ranges.push((start, indices.len()));
    ranges
}

fn split_ranges(ranges: Vec<(usize, usize)>) -> (Vec<usize>, Vec<usize>) {
    ranges.into_iter().unzip()
}

fn ranges_from_starts_ends(starts: &[usize], ends: &[usize]) -> PyResult<Vec<(usize, usize)>> {
    if starts.len() != ends.len() {
        return Err(PyValueError::new_err(
            "range starts and ends must have the same length",
        ));
    }

    starts
        .iter()
        .zip(ends)
        .map(|(&start, &end)| {
            if start > end {
                Err(PyValueError::new_err(
                    "range start must be less than or equal to range end",
                ))
            } else {
                Ok((start, end))
            }
        })
        .collect()
}

fn user_index_ranges_into_numpy<'py>(
    py: Python<'py>,
    (indices, ranges): UserIndexRanges,
) -> PyUserIndexRanges<'py> {
    let (starts, ends) = split_ranges(ranges);
    (
        indices.into_pyarray(py),
        starts.into_pyarray(py),
        ends.into_pyarray(py),
    )
}

fn f64_results_into_arrow(results: Vec<f64>) -> ArrowPyArray {
    let array: ArrayRef = Arc::new(Float64Array::from(results));
    ArrowPyArray::from_array_ref(array)
}

fn user_indices_for_ord_values<T: Ord>(values: &[T]) -> UserIndexRanges {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&left, &right| values[left].cmp(&values[right]).then(left.cmp(&right)));
    let ranges = ranges_from_sorted_indices(values, &indices);
    (indices, ranges)
}

fn user_indices_for_ord_values_at_indices<T: Ord>(
    values: &[T],
    mut indices: Vec<usize>,
) -> UserIndexRanges {
    indices.sort_by(|&left, &right| values[left].cmp(&values[right]).then(left.cmp(&right)));
    let ranges = ranges_from_sorted_indices(values, &indices);
    (indices, ranges)
}

fn compare_f64_values(left: f64, right: f64) -> std::cmp::Ordering {
    left.total_cmp(&right)
}

fn user_indices_for_f64_values(values: &[f64]) -> UserIndexRanges {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&left, &right| {
        compare_f64_values(values[left], values[right]).then(left.cmp(&right))
    });
    let ranges = ranges_from_sorted_indices(values, &indices);
    (indices, ranges)
}

fn user_indices_for_f64_values_at_indices(
    values: &[f64],
    mut indices: Vec<usize>,
) -> UserIndexRanges {
    indices.sort_by(|&left, &right| {
        compare_f64_values(values[left], values[right]).then(left.cmp(&right))
    });
    let ranges = ranges_from_sorted_indices(values, &indices);
    (indices, ranges)
}

fn valid_coord_indices(latitudes: &[f64], longitudes: &[f64]) -> PyResult<Vec<usize>> {
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }

    Ok((0..latitudes.len())
        .filter(|&idx| !latitudes[idx].is_nan() && !longitudes[idx].is_nan())
        .collect())
}

fn primitive_option_values<T>(array: &PrimitiveArray<T>) -> Vec<Option<T::Native>>
where
    T: arrow_array::types::ArrowPrimitiveType,
    T::Native: Copy,
{
    (0..array.len())
        .map(|idx| {
            if array.is_null(idx) {
                None
            } else {
                Some(array.value(idx))
            }
        })
        .collect()
}

#[pyfunction]
pub(crate) fn radius_of_gyration_batch_km(
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    radius_of_gyration_batch_impl(&latitudes, &longitudes, &ranges)
}

#[pyfunction]
pub(crate) fn radius_of_gyration_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(
        radius_of_gyration_batch_impl(latitudes.as_slice()?, longitudes.as_slice()?, &ranges)?
            .into_pyarray(py),
    )
}

#[pyfunction]
pub(crate) fn radius_of_gyration_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    indices: PyReadonlyArray1<'py, usize>,
    starts: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;
    Ok(radius_of_gyration_indexed_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        indices.as_slice()?,
        &ranges,
    )?
    .into_pyarray(py))
}

#[pyfunction]
pub(crate) fn radius_of_gyration_user_indices_numpy<'py>(
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
pub(crate) fn radius_of_gyration_valid_user_indices_numpy<'py>(
    py: Python<'py>,
    uids: &Bound<'py, PyAny>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
) -> PyResult<PyUserIndexRanges<'py>> {
    let latitudes = latitudes.as_slice()?;
    let longitudes = longitudes.as_slice()?;
    let valid_indices = valid_coord_indices(latitudes, longitudes)?;

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

fn as_nullable_f64_array(arr: ArrowPyArray, name: &str) -> PyResult<Float64Array> {
    let (array_ref, _field) = arr.into_inner();
    array_ref
        .as_any()
        .downcast_ref::<Float64Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected float64 Arrow array for {name}")))
}

#[pyfunction]
pub(crate) fn radius_of_gyration_arrow(
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

    Ok(f64_results_into_arrow(radius_of_gyration_batch_impl(
        &valid_latitudes,
        &valid_longitudes,
        &valid_ranges,
    )?))
}

#[pyfunction]
pub(crate) fn radius_of_gyration_user_indices_arrow<'py>(
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
pub(crate) fn radius_of_gyration_valid_user_indices_arrow<'py>(
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
pub(crate) fn radius_of_gyration_indexed_arrow(
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    indices: PyReadonlyArray1<usize>,
    starts: PyReadonlyArray1<usize>,
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

    let ranges = ranges_from_starts_ends(starts.as_slice()?, ends.as_slice()?)?;

    Ok(f64_results_into_arrow(radius_of_gyration_indexed_impl(
        &lat_values,
        &lng_values,
        indices.as_slice()?,
        &ranges,
    )?))
}
