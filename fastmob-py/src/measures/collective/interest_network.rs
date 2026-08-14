use arrow_array::Array;
use fastmob_core::measures::collective::interest_network::interest_network_impl;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{arrow_u32_values, as_nullable_u32_array, u32_results_into_arrow, u64_results_into_arrow};

/// Count distinct users shared by each unordered pair of catalogue locations.
#[pyfunction]
pub fn interest_network(
    py: Python<'_>,
    user_codes: PyArray,
    location_ranks: PyArray,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let user_codes = as_nullable_u32_array(user_codes, "user_codes")?;
    let location_ranks = as_nullable_u32_array(location_ranks, "location_ranks")?;
    if user_codes.null_count() > 0 || location_ranks.null_count() > 0 {
        return Err(PyValueError::new_err(
            "user_codes and location_ranks must not contain nulls",
        ));
    }
    let edges = py
        .detach(|| {
            interest_network_impl(
                arrow_u32_values(&user_codes),
                arrow_u32_values(&location_ranks),
            )
        })
        .map_err(PyValueError::new_err)?;
    let mut location_a = Vec::with_capacity(edges.len());
    let mut location_b = Vec::with_capacity(edges.len());
    let mut n_people = Vec::with_capacity(edges.len());
    for (left, right, count) in edges {
        location_a.push(left);
        location_b.push(right);
        n_people.push(count);
    }
    Ok((
        Py::new(py, u32_results_into_arrow(location_a))?.into_any(),
        Py::new(py, u32_results_into_arrow(location_b))?.into_any(),
        Py::new(py, u64_results_into_arrow(n_people))?.into_any(),
    ))
}
