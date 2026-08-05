use arrow_array::{Array, Float64Array, UInt64Array};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::as_nullable_u64_array;

enum Weights {
    Values(Float64Array),
    Ones,
}

impl Weights {
    fn len(&self) -> Option<usize> {
        match self {
            Self::Values(values) => Some(values.len()),
            Self::Ones => None,
        }
    }

    fn value(&self, index: usize) -> f64 {
        match self {
            Self::Values(values) => {
                if values.is_valid(index) {
                    values.value(index)
                } else {
                    0.0
                }
            }
            Self::Ones => 1.0,
        }
    }
}

fn weights(array: Option<PyArray>, name: &str) -> PyResult<Weights> {
    let Some(array) = array else {
        return Ok(Weights::Ones);
    };
    let (array, _) = array.into_inner();
    array
        .as_any()
        .downcast_ref::<Float64Array>()
        .cloned()
        .map(Weights::Values)
        .ok_or_else(|| PyValueError::new_err(format!("{name} must be an Arrow float64 array")))
}

fn encoded_rows<'a>(
    origin: &'a UInt64Array,
    destination: &'a UInt64Array,
    weights: &'a Weights,
) -> impl Iterator<Item = (Option<u64>, Option<u64>, f64)> + 'a {
    (0..origin.len()).map(move |index| {
        (
            origin.is_valid(index).then(|| origin.value(index)),
            destination
                .is_valid(index)
                .then(|| destination.value(index)),
            weights.value(index),
        )
    })
}

/// Compute CPC from pre-factorized sparse Arrow OD columns.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn common_part_of_commuters(
    origin_a: PyArray,
    destination_a: PyArray,
    weight_a: Option<PyArray>,
    origin_b: PyArray,
    destination_b: PyArray,
    weight_b: Option<PyArray>,
) -> PyResult<f64> {
    let origin_a = as_nullable_u64_array(origin_a, "origin_a")?;
    let destination_a = as_nullable_u64_array(destination_a, "destination_a")?;
    let origin_b = as_nullable_u64_array(origin_b, "origin_b")?;
    let destination_b = as_nullable_u64_array(destination_b, "destination_b")?;
    let weight_a = weights(weight_a, "weight_a")?;
    let weight_b = weights(weight_b, "weight_b")?;
    for (origin, destination, weight, name) in [
        (&origin_a, &destination_a, &weight_a, "a"),
        (&origin_b, &destination_b, &weight_b, "b"),
    ] {
        if origin.len() != destination.len()
            || weight.len().is_some_and(|length| origin.len() != length)
        {
            return Err(PyValueError::new_err(format!(
                "origin_{name}, destination_{name}, and weight_{name} must have equal lengths"
            )));
        }
    }
    let (counts_a, total_a) = fastmob_core::measures::evaluation::cpc::sparse_edge_counts(
        encoded_rows(&origin_a, &destination_a, &weight_a),
    )
    .map_err(PyValueError::new_err)?;
    let (counts_b, total_b) = fastmob_core::measures::evaluation::cpc::sparse_edge_counts(
        encoded_rows(&origin_b, &destination_b, &weight_b),
    )
    .map_err(PyValueError::new_err)?;
    Ok(
        fastmob_core::measures::evaluation::cpc::common_part_of_commuters_from_counts(
            counts_a, total_a, counts_b, total_b,
        ),
    )
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn common_part_of_links(
    origin_a: PyArray,
    destination_a: PyArray,
    weight_a: Option<PyArray>,
    origin_b: PyArray,
    destination_b: PyArray,
    weight_b: Option<PyArray>,
) -> PyResult<f64> {
    let origin_a = as_nullable_u64_array(origin_a, "origin_a")?;
    let destination_a = as_nullable_u64_array(destination_a, "destination_a")?;
    let origin_b = as_nullable_u64_array(origin_b, "origin_b")?;
    let destination_b = as_nullable_u64_array(destination_b, "destination_b")?;
    let weight_a = weights(weight_a, "weight_a")?;
    let weight_b = weights(weight_b, "weight_b")?;
    let (counts_a, _) = fastmob_core::measures::evaluation::cpc::sparse_edge_counts(encoded_rows(
        &origin_a,
        &destination_a,
        &weight_a,
    ))
    .map_err(PyValueError::new_err)?;
    let (counts_b, _) = fastmob_core::measures::evaluation::cpc::sparse_edge_counts(encoded_rows(
        &origin_b,
        &destination_b,
        &weight_b,
    ))
    .map_err(PyValueError::new_err)?;
    Ok(
        fastmob_core::measures::evaluation::cpc::common_part_of_links_from_counts(
            &counts_a, &counts_b,
        ),
    )
}

#[pyfunction]
pub fn common_part_of_commuters_distance(values_a: PyArray, values_b: PyArray) -> PyResult<f64> {
    let (a, _) = values_a.into_inner();
    let (b, _) = values_b.into_inner();
    let a = a
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err("values_a must be float64"))?;
    let b = b
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| PyValueError::new_err("values_b must be float64"))?;
    Ok(
        fastmob_core::measures::evaluation::cpc::common_part_of_commuters_distance(
            a.values(),
            b.values(),
        ),
    )
}
