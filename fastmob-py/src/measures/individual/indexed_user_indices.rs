use arrow_array::{Array, UInt64Array};
use fastmob_core::utils::{UserIndexRanges, split_user_index_ranges, user_indices_for_u64_codes};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

type PyUserIndexRanges<'py> = (Bound<'py, PyArray1<usize>>, Bound<'py, PyArray1<usize>>);

fn user_index_ranges_into_numpy<'py>(
    py: Python<'py>,
    (indices, ranges): UserIndexRanges,
) -> PyUserIndexRanges<'py> {
    let (indices, ends) = split_user_index_ranges((indices, ranges));
    (indices.into_pyarray(py), ends.into_pyarray(py))
}

#[pyfunction]
pub fn indexed_user_indices<'py>(
    py: Python<'py>,
    uids: &Bound<'py, PyAny>,
    num_groups: usize,
) -> PyResult<PyUserIndexRanges<'py>> {
    if let Ok(array) = uids.extract::<PyReadonlyArray1<u64>>() {
        let values = array.as_slice()?;
        return Ok(user_index_ranges_into_numpy(
            py,
            py.detach(|| user_indices_for_u64_codes(values, num_groups))
                .map_err(PyValueError::new_err)?,
        ));
    }

    if uids.hasattr("__arrow_c_array__")? {
        let uids = uids.extract::<ArrowPyArray>()?;
        let (array_ref, _field) = uids.into_inner();
        let array = array_ref.as_any();

        if let Some(array) = array.downcast_ref::<UInt64Array>() {
            if array.null_count() > 0 {
                return Err(PyValueError::new_err(
                    "uint64 uid codes for indexed user grouping must not contain nulls",
                ));
            }
            let start = array.offset();
            let end = start + array.len();
            let values = &array.values()[start..end];
            return Ok(user_index_ranges_into_numpy(
                py,
                py.detach(|| user_indices_for_u64_codes(values, num_groups))
                    .map_err(PyValueError::new_err)?,
            ));
        }
    }

    Err(PyValueError::new_err(
        "expected uint64 NumPy or Arrow uid codes for indexed user grouping",
    ))
}
