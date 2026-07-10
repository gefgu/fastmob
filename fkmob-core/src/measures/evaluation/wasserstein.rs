pub fn empirical_wasserstein_1d(a: &[f64], b: &[f64]) -> Result<f64, String> {
    if a.is_empty() || b.is_empty() {
        return Err("samples must be non-empty".to_string());
    }

    let mut sa = a.to_vec();
    let mut sb = b.to_vec();
    sa.sort_unstable_by(|x, y| x.total_cmp(y));
    sb.sort_unstable_by(|x, y| x.total_cmp(y));

    if sa.len() == sb.len() {
        let n = sa.len() as f64;
        let sum: f64 = sa.iter().zip(sb.iter()).map(|(x, y)| (x - y).abs()).sum();
        return Ok(sum / n);
    }

    let na = sa.len() as f64;
    let nb = sb.len() as f64;
    let mut ia = 0usize;
    let mut ib = 0usize;
    let mut cdf_a: f64 = 0.0;
    let mut cdf_b: f64 = 0.0;
    let mut prev = sa[0].min(sb[0]);
    let mut distance = 0.0;

    while ia < sa.len() || ib < sb.len() {
        let next_a = if ia < sa.len() { sa[ia] } else { f64::INFINITY };
        let next_b = if ib < sb.len() { sb[ib] } else { f64::INFINITY };
        let x = next_a.min(next_b);
        distance += (cdf_a - cdf_b).abs() * (x - prev);

        while ia < sa.len() && sa[ia] == x {
            cdf_a += 1.0 / na;
            ia += 1;
        }
        while ib < sb.len() && sb[ib] == x {
            cdf_b += 1.0 / nb;
            ib += 1;
        }
        prev = x;
    }

    Ok(distance)
}
