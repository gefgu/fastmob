use crate::utils::haversine::haversine_km;
use crate::utils::validate_indexed_coord_ends;

pub fn square_displacement_km2(lat0: f64, lng0: f64, lat_t: f64, lng_t: f64) -> f64 {
    let d = haversine_km(lat0, lng0, lat_t, lng_t);
    d * d
}

pub fn mean_square_displacement_indexed_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    timestamps_s: &[f64],
    indices: &[usize],
    ends: &[usize],
    delta_s: f64,
) -> Result<f64, String> {
    validate_indexed_coord_ends(latitudes, longitudes, indices, ends)?;
    if latitudes.len() != timestamps_s.len() {
        return Err(
            "timestamps_s must have the same length as latitudes and longitudes".to_string(),
        );
    }
    let mut total = 0.0f64;
    let mut count = 0usize;

    for i in 0..ends.len() {
        let start = if i == 0 { 0 } else { ends[i - 1] };
        let end = ends[i];
        if start >= end {
            continue;
        }
        let first_idx = indices[start];
        let t_limit = timestamps_s[first_idx] + delta_s;

        // indices[start..end] are in chronological order (time-ordered ranges)
        let mut rt_idx = first_idx;
        for &idx in &indices[start..end] {
            if timestamps_s[idx] <= t_limit {
                rt_idx = idx;
            } else {
                break;
            }
        }

        let d = haversine_km(
            latitudes[first_idx],
            longitudes[first_idx],
            latitudes[rt_idx],
            longitudes[rt_idx],
        );
        total += d * d;
        count += 1;
    }

    if count == 0 {
        Ok(0.0)
    } else {
        Ok(total / count as f64)
    }
}
