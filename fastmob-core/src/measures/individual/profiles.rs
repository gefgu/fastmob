//! Shared mobility-profile metrics and Routiner/Regular/Scouter labelling.
//!
//! This module deliberately works on plain visit rows, so Rust services and
//! language bindings use the exact same implementation.

use linfa::prelude::*;
use linfa_clustering::GaussianMixtureModel;
use ndarray015::Array2;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rustc_hash::{FxHashMap, FxHashSet};

use super::{diversity::diversity_batch, entropy::trajectory_entropy_batch};

const FIVE_MINUTES_US: i64 = 5 * 60 * 1_000_000;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProfileVisit {
    pub user_id: String,
    pub start_us: i64,
    pub end_us: i64,
    pub location_id: String,
    pub purpose: Option<String>,
}

#[derive(Debug, Clone, Copy)]
pub struct ProfileOptions {
    pub n_clusters: usize,
    pub max_iterations: usize,
}

impl Default for ProfileOptions {
    fn default() -> Self {
        Self {
            n_clusters: 3,
            max_iterations: 100,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MobilityProfile {
    pub user_id: String,
    pub intermittency: f64,
    pub degree_of_return: f64,
    pub mean_return: f64,
    pub mean_exploration: f64,
    pub regularity: f64,
    pub diversity: f64,
    pub stationarity: f64,
    pub entropy: f64,
    /// Canonical fastmob names: routiners, regulars, scouters.
    pub profile: String,
}

fn token(visit: &ProfileVisit) -> String {
    match &visit.purpose {
        Some(purpose) if !purpose.is_empty() => format!("{}_{}", visit.location_id, purpose),
        _ => visit.location_id.clone(),
    }
}

fn expanded_tokens(visits: &[ProfileVisit]) -> Vec<String> {
    let mut out = Vec::new();
    let mut seen = FxHashSet::default();
    for visit in visits {
        let mut ts = ((visit.start_us + FIVE_MINUTES_US - 1) / FIVE_MINUTES_US) * FIVE_MINUTES_US;
        let end = (visit.end_us / FIVE_MINUTES_US) * FIVE_MINUTES_US;
        while ts <= end {
            if seen.insert(ts) {
                out.push(token(visit));
            }
            ts += FIVE_MINUTES_US;
        }
    }
    if out.is_empty() {
        out.extend(visits.iter().map(token));
    }
    out
}

fn intermittency(tokens: &[String]) -> Option<(f64, f64, f64, f64)> {
    if tokens.is_empty() {
        return None;
    }
    let mut counts: FxHashMap<&str, usize> = FxHashMap::default();
    for value in tokens {
        *counts.entry(value).or_default() += 1;
    }
    let threshold = tokens.len() as f64 / counts.len() as f64 * 0.8;
    let known_at_start: FxHashSet<&str> = counts
        .iter()
        .filter_map(|(&value, &count)| (count as f64 >= threshold).then_some(value))
        .collect();
    let mut visited = FxHashSet::default();
    let mut states = Vec::with_capacity(tokens.len());
    for value in tokens {
        states.push(visited.contains(value.as_str()) || known_at_start.contains(value.as_str()));
        visited.insert(value.as_str());
    }
    let mut returns = Vec::new();
    let mut explorations = Vec::new();
    let mut state = states[0];
    let mut length = 0usize;
    for next in states {
        if next == state {
            length += 1;
            continue;
        }
        if state {
            returns.push(length);
        } else {
            explorations.push(length);
        }
        state = next;
        length = 1;
    }
    if state {
        returns.push(length);
    } else {
        explorations.push(length);
    }
    let mean = |values: &[usize]| values.iter().sum::<usize>() as f64 / values.len().max(1) as f64;
    let mean_return = mean(&returns);
    let mean_exploration = mean(&explorations);
    Some((
        mean_return + mean_exploration,
        mean_return.atan2(mean_exploration),
        mean_return,
        mean_exploration,
    ))
}

fn label_profiles(rows: &mut [MobilityProfile], options: ProfileOptions) -> Result<(), String> {
    let clusters = options.n_clusters;
    if clusters != 3 {
        return Err("mobility profile labels require exactly 3 clusters".into());
    }
    if rows.len() < clusters {
        return Err(format!(
            "need at least {clusters} users with finite profiling metrics, got {}",
            rows.len()
        ));
    }
    let n = rows.len() as f64;
    let mean_i = rows.iter().map(|r| r.intermittency).sum::<f64>() / n;
    let mean_d = rows.iter().map(|r| r.degree_of_return).sum::<f64>() / n;
    let std_i = (rows
        .iter()
        .map(|r| (r.intermittency - mean_i).powi(2))
        .sum::<f64>()
        / n)
        .sqrt()
        .max(1e-12);
    let std_d = (rows
        .iter()
        .map(|r| (r.degree_of_return - mean_d).powi(2))
        .sum::<f64>()
        / n)
        .sqrt()
        .max(1e-12);
    let points: Vec<[f64; 2]> = rows
        .iter()
        .map(|r| {
            [
                (r.intermittency - mean_i) / std_i,
                (r.degree_of_return - mean_d) / std_d,
            ]
        })
        .collect();
    let values = points
        .iter()
        .flat_map(|point| point.map(|value| value as f32))
        .collect();
    let observations =
        Array2::from_shape_vec((rows.len(), 2), values).map_err(|err| err.to_string())?;
    let dataset = DatasetBase::from(observations);
    let model = GaussianMixtureModel::params(clusters)
        .max_n_iterations(options.max_iterations as u64)
        .with_rng(Xoshiro256PlusPlus::seed_from_u64(0))
        .fit(&dataset)
        .map_err(|err| err.to_string())?;
    let assignments = model.predict(&dataset).into_raw_vec();
    let mut means = (0..3)
        .map(|cluster| {
            let selected: Vec<_> = rows
                .iter()
                .zip(&assignments)
                .filter_map(|(row, &assigned)| {
                    (assigned == cluster).then_some(row.degree_of_return)
                })
                .collect::<Vec<_>>();
            (
                cluster,
                if selected.is_empty() {
                    f64::NEG_INFINITY
                } else {
                    selected.iter().sum::<f64>() / selected.len() as f64
                },
            )
        })
        .collect::<Vec<_>>();
    means.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut names = ["scouters"; 3];
    for (rank, (cluster, _)) in means.into_iter().enumerate() {
        names[cluster] = ["routiners", "regulars", "scouters"][rank];
    }
    for (row, &cluster) in rows.iter_mut().zip(&assignments) {
        row.profile = names[cluster].to_string();
    }
    Ok(())
}

pub fn compute_profiles(
    visits: &[ProfileVisit],
    options: ProfileOptions,
) -> Result<Vec<MobilityProfile>, String> {
    let mut grouped: std::collections::BTreeMap<&str, Vec<&ProfileVisit>> =
        std::collections::BTreeMap::new();
    for visit in visits {
        grouped.entry(&visit.user_id).or_default().push(visit);
    }
    let mut profiles = Vec::with_capacity(grouped.len());
    for (user_id, rows) in grouped.iter_mut() {
        rows.sort_by_key(|visit| visit.start_us);
        let rows: Vec<ProfileVisit> = rows.iter().map(|visit| (*visit).clone()).collect();
        let Some((intermittency, degree_of_return, mean_return, mean_exploration)) =
            intermittency(&expanded_tokens(&rows))
        else {
            continue;
        };
        if !intermittency.is_finite() || !degree_of_return.is_finite() {
            continue;
        }
        let tokens: Vec<String> = rows.iter().map(token).collect();
        let diversity = diversity_batch(tokens.clone(), vec![(0, tokens.len())])?
            .into_iter()
            .next()
            .unwrap_or(0.0);
        let entropy = trajectory_entropy_batch(tokens, vec![(0, rows.len())], true)?
            .into_iter()
            .next()
            .unwrap_or(0.0);
        let mut pairs = FxHashSet::default();
        let mut dwell = 0.0;
        let mut start = i64::MAX;
        let mut end = i64::MIN;
        for visit in &rows {
            pairs.insert((
                visit.location_id.as_str(),
                visit.purpose.as_deref().unwrap_or(""),
            ));
            if visit.end_us > visit.start_us {
                dwell += (visit.end_us - visit.start_us) as f64 / 60_000_000.0;
                start = start.min(visit.start_us);
                end = end.max(visit.end_us);
            }
        }
        let stationarity = if end > start {
            dwell / ((end - start) as f64 / 60_000_000.0)
        } else {
            0.0
        };
        profiles.push(MobilityProfile {
            user_id: (*user_id).to_string(),
            intermittency,
            degree_of_return,
            mean_return,
            mean_exploration,
            regularity: 1.0 - pairs.len() as f64 / rows.len().max(1) as f64,
            diversity,
            stationarity,
            entropy,
            profile: String::new(),
        });
    }
    label_profiles(&mut profiles, options)?;
    Ok(profiles)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn visit(user: &str, hour: i64, location: &str) -> ProfileVisit {
        ProfileVisit {
            user_id: user.into(),
            start_us: hour * 3_600_000_000,
            end_us: hour * 3_600_000_000 + 1_800_000_000,
            location_id: location.into(),
            purpose: Some("stay".into()),
        }
    }
    #[test]
    fn computes_and_labels_profiles() {
        let mut visits = Vec::new();
        for (idx, user) in ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
            .iter()
            .enumerate()
        {
            for hour in 0..(4 + idx) {
                let location = if hour % (idx % 3 + 2) == 0 {
                    "home"
                } else {
                    "work"
                };
                visits.push(visit(user, hour as i64, location));
            }
        }
        let profiles = compute_profiles(&visits, ProfileOptions::default()).unwrap();
        assert_eq!(profiles.len(), 9);
        assert!(
            profiles
                .iter()
                .all(|row| row.entropy.is_finite() && !row.profile.is_empty())
        );
    }
}
