use arrow_array::{Array, Float64Array, PrimitiveArray, types::Float64Type};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use rayon::prelude::*;

use crate::haversine::haversine_km;

fn validate_inputs(
    latitudes: &[f64],
    longitudes: &[f64],
    ranges: &[(usize, usize)],
) -> PyResult<()> {
    if latitudes.len() != longitudes.len() {
        return Err(PyValueError::new_err(
            "latitudes and longitudes must have the same length",
        ));
    }

    let n = latitudes.len();
    for &(start, end) in ranges {
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

    Ok(())
}

fn maximum_distance_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    ranges: &[(usize, usize)],
) -> PyResult<Vec<f64>> {
    validate_inputs(latitudes, longitudes, ranges)?;

    let results: Vec<f64> = ranges
        .par_iter()
        .map(|&(start, end)| {
            if end - start < 2 {
                return 0.0;
            }
            (start + 1..end)
                .map(|idx| {
                    haversine_km(
                        latitudes[idx - 1],
                        longitudes[idx - 1],
                        latitudes[idx],
                        longitudes[idx],
                    )
                })
                .fold(0.0f64, f64::max)
        })
        .collect();

    Ok(results)
}

fn as_f64_array(arr: PyArray, name: &str) -> PyResult<PrimitiveArray<Float64Type>> {
    let (array_ref, _field) = arr.into_inner();
    let array = array_ref
        .as_any()
        .downcast_ref::<Float64Array>()
        .cloned()
        .ok_or_else(|| PyValueError::new_err(format!("expected float64 Arrow array for {name}")))?;

    if array.null_count() > 0 {
        return Err(PyValueError::new_err(format!(
            "Arrow array for {name} must not contain nulls"
        )));
    }

    Ok(array)
}

fn arrow_values(array: &PrimitiveArray<Float64Type>) -> &[f64] {
    let start = array.offset();
    let end = start + array.len();
    &array.values()[start..end]
}

#[pyfunction]
pub(crate) fn maximum_distance_batch_km(
    latitudes: Vec<f64>,
    longitudes: Vec<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    maximum_distance_impl(&latitudes, &longitudes, &ranges)
}

#[pyfunction]
pub(crate) fn maximum_distance_numpy(
    latitudes: PyReadonlyArray1<f64>,
    longitudes: PyReadonlyArray1<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    maximum_distance_impl(latitudes.as_slice()?, longitudes.as_slice()?, &ranges)
}

#[pyfunction]
pub(crate) fn maximum_distance_arrow(
    latitudes: PyArray,
    longitudes: PyArray,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<f64>> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    maximum_distance_impl(arrow_values(&latitudes), arrow_values(&longitudes), &ranges)
}
