use crate::utils::ArrowUsizeArrayExt;
use fastmob_core::privacy::privacy_assess_risk_impl;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_u64_values, arrow_valid_rows, arrow_values, as_nullable_f64_array, as_u64_array,
};

type PyPrivacyResult<'py> = (
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<usize>>,
    Bound<'py, PyArray1<f64>>,
);

fn into_py_result<'py>(
    py: Python<'py>,
    result: fastmob_core::privacy::PrivacyRiskResult,
) -> PyPrivacyResult<'py> {
    let (user_indices, risks, lats, lngs, row_indices, force_user_indices, instances, elems, probs) =
        result;
    (
        user_indices.into_pyarray(py),
        risks.into_pyarray(py),
        lats.into_pyarray(py),
        lngs.into_pyarray(py),
        row_indices.into_pyarray(py),
        force_user_indices.into_pyarray(py),
        instances.into_pyarray(py),
        elems.into_pyarray(py),
        probs.into_pyarray(py),
    )
}

#[pyfunction]
#[pyo3(signature = (
    latitudes,
    longitudes,
    time_keys,
    indices,
    ends,
    target_user_indices,
    attack_kind,
    knowledge_length,
    tolerance,
    force_instances
))]
#[allow(clippy::too_many_arguments)]
pub fn privacy_assess_risk_indexed<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    time_keys: Option<ArrowPyArray>,
    indices: pyo3_arrow::PyArray,
    ends: pyo3_arrow::PyArray,
    target_user_indices: pyo3_arrow::PyArray,
    attack_kind: u8,
    knowledge_length: usize,
    tolerance: f64,
    force_instances: bool,
) -> PyResult<PyPrivacyResult<'py>> {
    let time_keys = match time_keys {
        Some(values) => Some(as_u64_array(values, "time_keys")?),
        None => None,
    };
    let time_keys_slice = time_keys.as_ref().map(arrow_u64_values);
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;
    let target_user_indices = target_user_indices.as_slice()?;

    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    let result = privacy_assess_risk_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        time_keys_slice,
        Some(indices),
        ends,
        target_user_indices,
        attack_kind,
        knowledge_length,
        tolerance,
        force_instances,
        valid_rows.as_deref(),
    )
    .map_err(PyValueError::new_err)?;
    Ok(into_py_result(py, result))
}

#[pyfunction]
#[pyo3(signature = (
    latitudes,
    longitudes,
    time_keys,
    ends,
    target_user_indices,
    attack_kind,
    knowledge_length,
    tolerance,
    force_instances
))]
#[allow(clippy::too_many_arguments)]
pub fn privacy_assess_risk_presorted<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    time_keys: Option<ArrowPyArray>,
    ends: pyo3_arrow::PyArray,
    target_user_indices: pyo3_arrow::PyArray,
    attack_kind: u8,
    knowledge_length: usize,
    tolerance: f64,
    force_instances: bool,
) -> PyResult<PyPrivacyResult<'py>> {
    let time_keys = match time_keys {
        Some(values) => Some(as_u64_array(values, "time_keys")?),
        None => None,
    };
    let time_keys_slice = time_keys.as_ref().map(arrow_u64_values);
    let ends = ends.as_slice()?;
    let target_user_indices = target_user_indices.as_slice()?;

    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    let result = privacy_assess_risk_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        time_keys_slice,
        None,
        ends,
        target_user_indices,
        attack_kind,
        knowledge_length,
        tolerance,
        force_instances,
        valid_rows.as_deref(),
    )
    .map_err(PyValueError::new_err)?;
    Ok(into_py_result(py, result))
}
