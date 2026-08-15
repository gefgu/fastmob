use crate::utils::ArrowUsizeArrayExt;
use fastmob_core::preprocessing::h3::{INVALID_CELL, batch_latlng_to_cells};
use fastmob_core::privacy::{AttackKind, privacy_assess_risk_impl};
use h3o::Resolution;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_u32_values, arrow_u64_values, arrow_valid_rows, arrow_values, as_nullable_f64_array,
    as_u32_array, as_u64_array,
};

#[pyclass(name = "PrivacyRiskResult")]
pub struct PyPrivacyRiskResult {
    user_indices: Vec<usize>,
    risks: Vec<f64>,
    force_lats: Vec<f64>,
    force_lngs: Vec<f64>,
    row_indices: Vec<usize>,
    force_user_indices: Vec<usize>,
    instances: Vec<usize>,
    elems: Vec<usize>,
    probs: Vec<f64>,
}

impl From<fastmob_core::privacy::PrivacyRiskResult> for PyPrivacyRiskResult {
    fn from(result: fastmob_core::privacy::PrivacyRiskResult) -> Self {
        Self {
            user_indices: result.user_indices,
            risks: result.risks,
            force_lats: result.force_lats,
            force_lngs: result.force_lngs,
            row_indices: result.row_indices,
            force_user_indices: result.force_user_indices,
            instances: result.instances,
            elems: result.elems,
            probs: result.probs,
        }
    }
}

#[pymethods]
impl PyPrivacyRiskResult {
    #[getter]
    fn user_indices(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        arrow_usize_array(py, &self.user_indices)
    }
    #[getter]
    fn risks(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        arrow_f64_array(py, &self.risks)
    }
    #[getter]
    fn force_lats(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        arrow_f64_array(py, &self.force_lats)
    }
    #[getter]
    fn force_lngs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        arrow_f64_array(py, &self.force_lngs)
    }
    #[getter]
    fn row_indices(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        arrow_usize_array(py, &self.row_indices)
    }
    #[getter]
    fn force_user_indices(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        arrow_usize_array(py, &self.force_user_indices)
    }
    #[getter]
    fn instances(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        arrow_usize_array(py, &self.instances)
    }
    #[getter]
    fn elems(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        arrow_usize_array(py, &self.elems)
    }
    #[getter]
    fn probs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        arrow_f64_array(py, &self.probs)
    }
}

fn arrow_usize_array(py: Python<'_>, values: &[usize]) -> PyResult<Py<PyAny>> {
    PyModule::import(py, "pyarrow")?
        .getattr("array")?
        .call1((values.to_vec(),))
        .map(Bound::unbind)
}

fn arrow_f64_array(py: Python<'_>, values: &[f64]) -> PyResult<Py<PyAny>> {
    PyModule::import(py, "pyarrow")?
        .getattr("array")?
        .call1((values.to_vec(),))
        .map(Bound::unbind)
}

fn attack_kind(name: &str) -> PyResult<AttackKind> {
    AttackKind::try_from(name).map_err(PyValueError::new_err)
}

fn resolve_resolution(resolution: u8) -> PyResult<Resolution> {
    Resolution::try_from(resolution).map_err(|_| {
        PyValueError::new_err(format!(
            "h3_resolution must be between 0 and 15, got {resolution}"
        ))
    })
}

#[pyfunction]
#[pyo3(signature = (latitudes, longitudes, ends, target_user_indices, attack, knowledge_length, time_keys=None, indices=None, tolerance=0.0, force_instances=false, h3_resolution=12, location_ids=None))]
#[allow(clippy::too_many_arguments)]
pub fn privacy_assess_risk<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    ends: pyo3_arrow::PyArray,
    target_user_indices: pyo3_arrow::PyArray,
    attack: &str,
    knowledge_length: usize,
    time_keys: Option<ArrowPyArray>,
    indices: Option<pyo3_arrow::PyArray>,
    tolerance: f64,
    force_instances: bool,
    h3_resolution: u8,
    location_ids: Option<ArrowPyArray>,
) -> PyResult<Py<PyPrivacyRiskResult>> {
    let time_keys = time_keys
        .map(|values| as_u32_array(values, "time_keys"))
        .transpose()?;
    let time_keys_slice = time_keys.as_ref().map(arrow_u32_values);
    let indices = match indices.as_ref() {
        Some(values) => Some(values.as_slice()?),
        None => None,
    };
    let ends = ends.as_slice()?;
    let target_user_indices = target_user_indices.as_slice()?;
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let provided_location_ids = location_ids
        .map(|values| as_u64_array(values, "location_ids"))
        .transpose()?;
    if force_instances && provided_location_ids.is_some() {
        return Err(PyValueError::new_err(
            "force_instances is not supported with externally assigned location_ids",
        ));
    }
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    let lats = arrow_values(&latitudes);
    let lngs = arrow_values(&longitudes);
    let (location_ids, h3_valid_rows) = if let Some(ids) = provided_location_ids {
        let cells = arrow_u64_values(&ids).to_vec();
        let valid = cells
            .iter()
            .enumerate()
            .map(|(idx, _)| valid_rows.as_ref().is_none_or(|rows| rows[idx]))
            .collect();
        (cells, valid)
    } else {
        let resolution = resolve_resolution(h3_resolution)?;
        py.detach(|| {
            let cells = batch_latlng_to_cells(lats, lngs, resolution, valid_rows.as_deref());
            let valid: Vec<bool> = cells
                .iter()
                .enumerate()
                .map(|(idx, &cell)| {
                    cell != INVALID_CELL && valid_rows.as_ref().is_none_or(|rows| rows[idx])
                })
                .collect();
            (cells, valid)
        })
    };
    let result = privacy_assess_risk_impl(
        lats,
        lngs,
        time_keys_slice,
        indices.as_deref(),
        &ends,
        &target_user_indices,
        attack_kind(attack)?,
        knowledge_length,
        tolerance,
        force_instances,
        Some(&h3_valid_rows),
        &location_ids,
    )
    .map_err(PyValueError::new_err)?;
    Py::new(py, PyPrivacyRiskResult::from(result))
}
