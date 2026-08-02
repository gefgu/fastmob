use crate::utils::ArrowUsizeArrayExt;
use fastmob_core::privacy::{AttackKind, privacy_assess_risk_impl};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{
    arrow_u64_values, arrow_valid_rows, arrow_values, as_nullable_f64_array, as_u64_array,
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

#[pyfunction]
#[pyo3(signature = (latitudes, longitudes, ends, target_user_indices, attack, knowledge_length, time_keys=None, indices=None, tolerance=0.0, force_instances=false))]
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
) -> PyResult<Py<PyPrivacyRiskResult>> {
    let time_keys = time_keys
        .map(|values| as_u64_array(values, "time_keys"))
        .transpose()?;
    let time_keys_slice = time_keys.as_ref().map(arrow_u64_values);
    let indices = match indices.as_ref() {
        Some(values) => Some(values.as_slice()?),
        None => None,
    };
    let ends = ends.as_slice()?;
    let target_user_indices = target_user_indices.as_slice()?;
    let latitudes = as_nullable_f64_array(latitudes, "latitudes")?;
    let longitudes = as_nullable_f64_array(longitudes, "longitudes")?;
    let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes]);
    let result = privacy_assess_risk_impl(
        arrow_values(&latitudes),
        arrow_values(&longitudes),
        time_keys_slice,
        indices,
        ends,
        target_user_indices,
        attack_kind(attack)?,
        knowledge_length,
        tolerance,
        force_instances,
        valid_rows.as_deref(),
    )
    .map_err(PyValueError::new_err)?;
    Py::new(py, PyPrivacyRiskResult::from(result))
}
