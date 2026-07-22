use fastmob_core::measures::collective::square_displacement::{
    mean_square_displacement_indexed_impl, square_displacement_km2 as core_square_displacement_km2,
};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::PyArray;

use crate::utils::{arrow_valid_rows, arrow_values, as_nullable_f64_array, validate_indexed_ends};

fn is_arrow_array(obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    obj.hasattr("__arrow_c_array__")
}

#[pyfunction]
pub fn square_displacement_km2(lat0: f64, lng0: f64, lat_t: f64, lng_t: f64) -> f64 {
    core_square_displacement_km2(lat0, lng0, lat_t, lng_t)
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn mean_square_displacement_indexed<'py>(
    latitudes: &Bound<'py, PyAny>,
    longitudes: &Bound<'py, PyAny>,
    timestamps_s: &Bound<'py, PyAny>,
    indices: PyReadonlyArray1<'py, usize>,
    ends: PyReadonlyArray1<'py, usize>,
    delta_s: f64,
) -> PyResult<f64> {
    let indices = indices.as_slice()?;
    let ends = ends.as_slice()?;

    if let (Ok(latitudes), Ok(longitudes), Ok(timestamps_s)) = (
        latitudes.extract::<PyReadonlyArray1<f64>>(),
        longitudes.extract::<PyReadonlyArray1<f64>>(),
        timestamps_s.extract::<PyReadonlyArray1<f64>>(),
    ) {
        let lats = latitudes.as_slice()?;
        let lngs = longitudes.as_slice()?;
        let times = timestamps_s.as_slice()?;
        validate_indexed_ends(lats.len(), indices, ends)?;
        return mean_square_displacement_indexed_impl(
            lats, lngs, times, indices, ends, delta_s, None,
        )
        .map_err(PyValueError::new_err);
    }

    if is_arrow_array(latitudes)? && is_arrow_array(longitudes)? && is_arrow_array(timestamps_s)? {
        let latitudes = as_nullable_f64_array(latitudes.extract::<PyArray>()?, "latitudes")?;
        let longitudes = as_nullable_f64_array(longitudes.extract::<PyArray>()?, "longitudes")?;
        let timestamps_s =
            as_nullable_f64_array(timestamps_s.extract::<PyArray>()?, "timestamps_s")?;
        validate_indexed_ends(latitudes.len(), indices, ends)?;
        let valid_rows = arrow_valid_rows(&[&latitudes, &longitudes, &timestamps_s]);
        return mean_square_displacement_indexed_impl(
            arrow_values(&latitudes),
            arrow_values(&longitudes),
            arrow_values(&timestamps_s),
            indices,
            ends,
            delta_s,
            valid_rows.as_deref(),
        )
        .map_err(PyValueError::new_err);
    }

    Err(PyTypeError::new_err(
        "latitudes, longitudes, and timestamps_s must all be NumPy arrays or all be Arrow arrays",
    ))
}
