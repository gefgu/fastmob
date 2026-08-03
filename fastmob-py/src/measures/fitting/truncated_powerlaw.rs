use fastmob_core::measures::fitting::truncated_powerlaw::fit_truncated_powerlaw_grid as core_fit_truncated_powerlaw_grid;
use numpy::PyReadonlyArray1;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Fit a truncated power-law to positive values via a parallel Rust grid
/// search. Returns `((c, r0, beta, kappa), x_bin_centers, y_bin_densities)`.
#[pyfunction]
pub fn fit_truncated_powerlaw_grid(
    py: Python<'_>,
    values: PyReadonlyArray1<'_, f64>,
    n_bins: usize,
) -> PyResult<((f64, f64, f64, f64), Vec<f64>, Vec<f64>)> {
    let values = values.as_slice()?.to_vec();
    let fit = py
        .detach(|| core_fit_truncated_powerlaw_grid(&values, n_bins))
        .map_err(PyValueError::new_err)?;
    let [c, r0, beta, kappa] = fit.parameters;
    Ok(((c, r0, beta, kappa), fit.x, fit.y))
}
