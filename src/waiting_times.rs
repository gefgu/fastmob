use pyo3::prelude::*;
use rayon::prelude::*;

#[pyfunction]
pub(crate) fn waiting_times_seconds(
    timestamps_s: Vec<f64>,
    ranges: Vec<(usize, usize)>,
) -> PyResult<Vec<Vec<f64>>> {
    let results: Vec<Vec<f64>> = ranges
        .par_iter()
        .map(|&(start, end)| {
            let slice = &timestamps_s[start..end];
            if slice.len() < 2 {
                return Vec::new();
            }
            slice.windows(2).map(|w| w[1] - w[0]).collect()
        })
        .collect();

    Ok(results)
}
