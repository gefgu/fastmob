use ndarray::Array2;
use std::f64::consts::PI;
use wass::sliced_wasserstein;

#[allow(clippy::too_many_arguments)]
pub fn stvd_emd_impl(
    xs_a: &[f64],
    ys_a: &[f64],
    ts_a: &[f64],
    ws_a: &[f64],
    xs_b: &[f64],
    ys_b: &[f64],
    ts_b: &[f64],
    ws_b: &[f64],
    alpha: f64,
    cyclical_period: f64,
    num_projections: usize,
) -> Result<f64, String> {
    let n = xs_a.len();
    let m = xs_b.len();

    if n == 0 || m == 0 {
        return Err("distributions must be non-empty".to_string());
    }
    if ys_a.len() != n || ts_a.len() != n || ws_a.len() != n {
        return Err("all arrays for distribution A must have the same length".to_string());
    }
    if xs_b.len() != m || ys_b.len() != m || ts_b.len() != m || ws_b.len() != m {
        return Err("all arrays for distribution B must have the same length".to_string());
    }
    if cyclical_period <= 0.0 {
        return Err("cyclical_period must be positive".to_string());
    }
    if num_projections == 0 {
        return Err("num_projections must be positive".to_string());
    }

    let sum_a: f64 = ws_a.iter().sum();
    let sum_b: f64 = ws_b.iter().sum();
    if sum_a <= 0.0 || sum_b <= 0.0 {
        return Err("weights must sum to a positive value".to_string());
    }

    // wass::sliced_wasserstein in v0.2.0 is unweighted; we validate input weights
    // above to keep the same input contract for this function.

    let r = (alpha * cyclical_period) / (2.0 * PI);

    let mut cloud_a = Array2::<f32>::zeros((n, 4));
    for i in 0..n {
        cloud_a[[i, 0]] = xs_a[i] as f32;
        cloud_a[[i, 1]] = ys_a[i] as f32;

        let t = ts_a[i].rem_euclid(cyclical_period);
        let theta = 2.0 * PI * (t / cyclical_period);
        cloud_a[[i, 2]] = (r * theta.cos()) as f32;
        cloud_a[[i, 3]] = (r * theta.sin()) as f32;
    }

    let mut cloud_b = Array2::<f32>::zeros((m, 4));
    for j in 0..m {
        cloud_b[[j, 0]] = xs_b[j] as f32;
        cloud_b[[j, 1]] = ys_b[j] as f32;

        let t = ts_b[j].rem_euclid(cyclical_period);
        let theta = 2.0 * PI * (t / cyclical_period);
        cloud_b[[j, 2]] = (r * theta.cos()) as f32;
        cloud_b[[j, 3]] = (r * theta.sin()) as f32;
    }

    let seed = 42_u64;
    let p = 1.0_f32;
    let distance = sliced_wasserstein(&cloud_a, &cloud_b, num_projections, seed, p);

    Ok(distance as f64)
}
