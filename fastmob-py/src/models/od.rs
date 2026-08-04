use fastmob_core::models::od::{gravity_matrix_impl, gravity_od_row_impl};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray as ArrowPyArray;

use crate::utils::{arrow_values, as_f64_array, f64_results_into_arrow, u64_results_into_arrow};

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (latitudes, longitudes, relevances, tot_outflows, deterrence_type, deterrence_arg, origin_exp, destination_exp, gravity_type, out_format))]
pub fn model_gravity_matrix_numpy<'py>(
    py: Python<'py>,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    relevances: PyReadonlyArray1<'py, f64>,
    tot_outflows: PyReadonlyArray1<'py, f64>,
    deterrence_type: &str,
    deterrence_arg: f64,
    origin_exp: f64,
    destination_exp: f64,
    gravity_type: &str,
    out_format: &str,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let lats = latitudes.as_slice()?;
    let lons = longitudes.as_slice()?;
    let relevances = relevances.as_slice()?;
    let tot_outflows = tot_outflows.as_slice()?;
    let matrix = gravity_matrix_impl(
        lats,
        lons,
        relevances,
        tot_outflows,
        deterrence_type,
        deterrence_arg,
        origin_exp,
        destination_exp,
        gravity_type,
        out_format,
    )
    .map_err(PyValueError::new_err)?;
    Ok(matrix.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn model_gravity_od_row_numpy<'py>(
    py: Python<'py>,
    origin: usize,
    latitudes: PyReadonlyArray1<'py, f64>,
    longitudes: PyReadonlyArray1<'py, f64>,
    relevances: PyReadonlyArray1<'py, f64>,
    deterrence_type: &str,
    deterrence_arg: f64,
    origin_exp: f64,
    destination_exp: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let row = gravity_od_row_impl(
        origin,
        latitudes.as_slice()?,
        longitudes.as_slice()?,
        relevances.as_slice()?,
        deterrence_type,
        deterrence_arg,
        origin_exp,
        destination_exp,
    )
    .map_err(PyValueError::new_err)?;
    Ok(row.into_pyarray(py))
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn model_gravity_flows_arrow<'py>(
    py: Python<'py>,
    latitudes: ArrowPyArray,
    longitudes: ArrowPyArray,
    relevances: ArrowPyArray,
    tot_outflows: ArrowPyArray,
    deterrence_type: &str,
    deterrence_arg: f64,
    origin_exp: f64,
    destination_exp: f64,
    gravity_type: &str,
    out_format: &str,
) -> PyResult<(Py<PyAny>, Py<PyAny>, Py<PyAny>)> {
    let latitudes = as_f64_array(latitudes, "latitudes")?;
    let longitudes = as_f64_array(longitudes, "longitudes")?;
    let relevances = as_f64_array(relevances, "relevances")?;
    let tot_outflows = as_f64_array(tot_outflows, "tot_outflows")?;
    let n = latitudes.len();
    let matrix = py
        .detach(|| {
            gravity_matrix_impl(
                arrow_values(&latitudes),
                arrow_values(&longitudes),
                arrow_values(&relevances),
                arrow_values(&tot_outflows),
                deterrence_type,
                deterrence_arg,
                origin_exp,
                destination_exp,
                gravity_type,
                out_format,
            )
        })
        .map_err(PyValueError::new_err)?;
    let mut origins = Vec::new();
    let mut destinations = Vec::new();
    let mut values = Vec::new();
    for (index, value) in matrix.into_iter().enumerate() {
        if value > 0.0 {
            origins.push((index / n) as u64);
            destinations.push((index % n) as u64);
            values.push(value);
        }
    }
    Ok((
        Py::new(py, u64_results_into_arrow(origins))?.into_any(),
        Py::new(py, u64_results_into_arrow(destinations))?.into_any(),
        Py::new(py, f64_results_into_arrow(values))?.into_any(),
    ))
}
