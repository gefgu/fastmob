use rand::SeedableRng;
use rand::distributions::{Distribution, WeightedIndex};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;

use crate::models::od::validate_equal_lengths;
use crate::utils::haversine::haversine_km;

type RadiationResult = Result<(Vec<usize>, Vec<usize>, Vec<f64>), String>;

pub fn model_radiation_probabilities_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    relevances: &[f64],
    tot_outflows: &[f64],
) -> RadiationResult {
    let n = validate_equal_lengths(&[
        ("longitudes", longitudes.len()),
        ("relevances", relevances.len()),
        ("tot_outflows", tot_outflows.len()),
        ("latitudes", latitudes.len()),
    ])?;
    let total_relevance: f64 = relevances.iter().sum();

    let per_origin: Vec<Vec<(usize, usize, f64)>> = (0..n)
        .into_par_iter()
        .map(|origin| {
            if tot_outflows[origin] <= 0.0 {
                return Vec::new();
            }
            let origin_relevance = relevances[origin];
            let normalization_factor = 1.0 / (1.0 - origin_relevance / total_relevance);
            let mut destinations_and_distances: Vec<(usize, f64)> = (0..n)
                .filter(|&destination| destination != origin)
                .map(|destination| {
                    (
                        destination,
                        haversine_km(
                            latitudes[origin],
                            longitudes[origin],
                            latitudes[destination],
                            longitudes[destination],
                        ),
                    )
                })
                .collect();
            destinations_and_distances.sort_by(|left, right| {
                left.1
                    .partial_cmp(&right.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut sum_inside = 0.0;
            let mut rows = Vec::with_capacity(n.saturating_sub(1));
            for (destination, _) in destinations_and_distances {
                let destination_relevance = relevances[destination];
                let prob = normalization_factor * (origin_relevance * destination_relevance)
                    / ((origin_relevance + sum_inside)
                        * (origin_relevance + sum_inside + destination_relevance));
                sum_inside += destination_relevance;
                rows.push((origin, destination, prob));
            }
            rows
        })
        .collect();

    let total_len: usize = per_origin.iter().map(Vec::len).sum();
    let mut origins = Vec::with_capacity(total_len);
    let mut destinations = Vec::with_capacity(total_len);
    let mut probabilities = Vec::with_capacity(total_len);
    for rows in per_origin {
        for (origin, destination, probability) in rows {
            origins.push(origin);
            destinations.push(destination);
            probabilities.push(probability);
        }
    }
    Ok((origins, destinations, probabilities))
}

/// Sample integer radiation flows without materialising a dense probability matrix
/// in Python. Sampling is deterministic for a supplied seed.
pub fn model_radiation_sample_flows_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    relevances: &[f64],
    tot_outflows: &[f64],
    seed: u64,
) -> RadiationResult {
    let (origins, destinations, probabilities) =
        model_radiation_probabilities_impl(latitudes, longitudes, relevances, tot_outflows)?;
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut quantities = vec![0_u64; probabilities.len()];
    let mut row_start = 0;
    while row_start < origins.len() {
        let origin = origins[row_start];
        let mut row_end = row_start + 1;
        while row_end < origins.len() && origins[row_end] == origin {
            row_end += 1;
        }
        let weights = &probabilities[row_start..row_end];
        let total: f64 = weights.iter().sum();
        if total.is_finite() && total > 0.0 {
            let distribution = WeightedIndex::new(weights.iter().map(|weight| weight / total))
                .map_err(|error| error.to_string())?;
            for _ in 0..tot_outflows[origin].max(0.0) as u64 {
                quantities[row_start + distribution.sample(&mut rng)] += 1;
            }
        }
        row_start = row_end;
    }
    let mut sampled_origins = Vec::new();
    let mut sampled_destinations = Vec::new();
    let mut sampled_values = Vec::new();
    for (index, quantity) in quantities.into_iter().enumerate() {
        if quantity > 0 {
            sampled_origins.push(origins[index]);
            sampled_destinations.push(destinations[index]);
            sampled_values.push(quantity as f64);
        }
    }
    Ok((sampled_origins, sampled_destinations, sampled_values))
}
