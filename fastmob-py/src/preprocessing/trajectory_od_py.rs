use fastmob_core::preprocessing::trajectory_od::od_edge_counts_impl;
use fastmob_core::utils::validate_ends;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{arrow_u64_values, as_u64_array, u64_results_into_arrow, ArrowUsizeArrayExt};

/// Pair each user's consecutive H3 cells and count unique `(origin, destination)`
/// edges in one pass, fusing what `trajectory_to_od` used to do as a separate
/// NumPy pairing step followed by a Narwhals `group_by`.
///
/// `cells` must already be sorted/grouped by user (nulls already dropped by the
/// caller); `ends` are cumulative per-user end offsets, matching
/// `_build_presorted_user_ends`'s contract.
#[pyfunction]
pub fn od_edge_counts_presorted(
    py: Python<'_>,
    cells: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
    drop_self_loops: bool,
) -> PyResult<(ArrowPyArray, ArrowPyArray, ArrowPyArray)> {
    let cells = as_u64_array(cells, "cells")?;
    let ends = ends.as_slice()?;
    validate_ends(cells.len(), &ends).map_err(PyValueError::new_err)?;

    let (origins, destinations, counts) =
        py.detach(|| od_edge_counts_impl(arrow_u64_values(&cells), &ends, drop_self_loops));

    Ok((
        u64_results_into_arrow(origins),
        u64_results_into_arrow(destinations),
        u64_results_into_arrow(counts),
    ))
}
