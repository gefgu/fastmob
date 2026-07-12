use rand::Rng;

pub(crate) fn sample_tpl_rng(rng: &mut impl Rng, xmin: f64, alpha: f64, lambda_: f64) -> f64 {
    loop {
        let u: f64 = rng.gen_range(0.0_f64..1.0);
        let x = xmin - (1.0 / lambda_) * (1.0 - u).ln();
        if rng.gen_range(0.0_f64..1.0) < (x / xmin).powf(-alpha) {
            return x;
        }
    }
}

pub fn cdf_choice(rng: &mut impl Rng, cdf: &[f64]) -> usize {
    let n = cdf.len();
    if n == 0 {
        return 0;
    }
    let total = cdf[n - 1];
    if total <= 0.0 || !total.is_finite() {
        return rng.gen_range(0..n);
    }
    let threshold = rng.gen_range(0.0_f64..1.0) * total;
    cdf.partition_point(|value| value.is_finite() && *value < threshold)
        .min(n - 1)
}

pub fn weighted_choice_excluding(
    rng: &mut impl Rng,
    visited_locs: &[usize],
    visit_counts: &[u32],
    total_visits: f64,
    exclude: usize,
) -> usize {
    let total = total_visits - visit_counts[exclude] as f64;
    if total <= 0.0 {
        return exclude;
    }
    let threshold = rng.gen_range(0.0_f64..1.0) * total;
    let mut cumsum = 0.0;
    let mut fallback = exclude;
    for &loc in visited_locs {
        if loc == exclude {
            continue;
        }
        fallback = loc;
        cumsum += visit_counts[loc] as f64;
        if cumsum > threshold {
            return loc;
        }
    }
    fallback
}

#[inline(always)]
pub fn derive_agent_seed(master_seed: u64, agent: usize, stream: u64) -> u64 {
    let mut value = master_seed
        ^ ((agent as u64).wrapping_add(1)).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ stream.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

#[inline(always)]
pub(crate) fn estimate_records_per_agent(start_ts: i64, end_ts: i64, xmin: f64) -> usize {
    if end_ts <= start_ts || xmin <= 0.0 || !xmin.is_finite() {
        return 2;
    }
    let duration_h = (end_ts - start_ts) as f64 / 3600.0;
    ((duration_h / xmin).ceil() as usize + 2).clamp(2, 4096)
}

pub(crate) fn validate_starting_locs_length(
    starts: Option<&[i64]>,
    n_agents: usize,
) -> Result<(), String> {
    match starts {
        Some(values) if values.len() < n_agents => {
            Err("starting_locs length must be at least n_agents".to_string())
        }
        _ => Ok(()),
    }
}
