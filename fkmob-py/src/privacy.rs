use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use fkmob_core::privacy::privacy_assess_risk_impl;

use crate::utils::{arrow_valid_rows, arrow_values, as_nullable_f64_array};

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
    result: fkmob_core::privacy::PrivacyRiskResult,
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
pub fn privacy_assess_risk_indexed_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    time_keys: Option<PyReadonlyArray1<'py, u64>>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    target_user_indices: PyReadonlyArray1<'py, usize>,
    attack_kind: u8,
    knowledge_length: usize,
    tolerance: f64,
    force_instances: bool,
) -> PyResult<PyPrivacyResult<'py>> {
    let time_keys_slice = match time_keys.as_ref() {
        Some(values) => Some(values.as_slice()?),
        None => None,
    };
    let result = privacy_assess_risk_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        time_keys_slice,
        Some(indices.as_slice()?),
        ends.as_slice()?,
        target_user_indices.as_slice()?,
        attack_kind,
        knowledge_length,
        tolerance,
        force_instances,
        None,
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
pub fn privacy_assess_risk_presorted_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    time_keys: Option<PyReadonlyArray1<'py, u64>>,
    ends: PyReadonlyArray1<'py, usize>,
    target_user_indices: PyReadonlyArray1<'py, usize>,
    attack_kind: u8,
    knowledge_length: usize,
    tolerance: f64,
    force_instances: bool,
) -> PyResult<PyPrivacyResult<'py>> {
    let time_keys_slice = match time_keys.as_ref() {
        Some(values) => Some(values.as_slice()?),
        None => None,
    };
    let result = privacy_assess_risk_impl(
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        time_keys_slice,
        None,
        ends.as_slice()?,
        target_user_indices.as_slice()?,
        attack_kind,
        knowledge_length,
        tolerance,
        force_instances,
        None,
    )
    .map_err(PyValueError::new_err)?;
    Ok(into_py_result(py, result))
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
pub fn privacy_assess_risk_indexed_arrow<'py>(
    py: Python<'py>,
    latitudes: PyArray,
    longitudes: PyArray,
    time_keys: Option<PyReadonlyArray1<'py, u64>>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    target_user_indices: PyReadonlyArray1<'py, usize>,
    attack_kind: u8,
    knowledge_length: usize,
    tolerance: f64,
    force_instances: bool,
) -> PyResult<PyPrivacyResult<'py>> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let time_keys_slice = match time_keys.as_ref() {
        Some(values) => Some(values.as_slice()?),
        None => None,
    };
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    let result = privacy_assess_risk_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        time_keys_slice,
        Some(indices.as_slice()?),
        ends.as_slice()?,
        target_user_indices.as_slice()?,
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
pub fn privacy_assess_risk_presorted_arrow<'py>(
    py: Python<'py>,
    latitudes: PyArray,
    longitudes: PyArray,
    time_keys: Option<PyReadonlyArray1<'py, u64>>,
    ends: PyReadonlyArray1<'py, usize>,
    target_user_indices: PyReadonlyArray1<'py, usize>,
    attack_kind: u8,
    knowledge_length: usize,
    tolerance: f64,
    force_instances: bool,
) -> PyResult<PyPrivacyResult<'py>> {
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let time_keys_slice = match time_keys.as_ref() {
        Some(values) => Some(values.as_slice()?),
        None => None,
    };
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    let result = privacy_assess_risk_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        time_keys_slice,
        None,
        ends.as_slice()?,
        target_user_indices.as_slice()?,
        attack_kind,
        knowledge_length,
        tolerance,
        force_instances,
        valid_rows.as_deref(),
    )
    .map_err(PyValueError::new_err)?;
    Ok(into_py_result(py, result))
}
