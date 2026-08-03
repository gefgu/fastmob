use h3o::{CellIndex, LatLng};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::preprocessing::h3::INVALID_CELL;

const NO_ROW: usize = usize::MAX;

#[derive(Clone, Copy, Debug)]
struct Obs {
    location: u64,
    time_key: u64,
    row_idx: usize,
}

#[derive(Clone, Copy, Debug)]
struct MetricObs {
    location: u64,
    value: f64,
}

#[derive(Clone, Debug)]
enum UserValues {
    Observations(Vec<Obs>),
    Metrics(Vec<MetricObs>),
}

enum PreparedCandidate<'a> {
    Observations {
        values: &'a [Obs],
        locations: FxHashMap<u64, usize>,
        location_times: FxHashMap<(u64, u64), usize>,
    },
    Metrics {
        locations: FxHashSet<u64>,
        metrics: FxHashMap<u64, f64>,
    },
}

enum PreparedInstance<'a> {
    Observations {
        values: &'a [Obs],
        locations: FxHashMap<u64, usize>,
        location_times: FxHashMap<(u64, u64), usize>,
    },
    Metrics {
        values: &'a [MetricObs],
        locations: FxHashSet<u64>,
    },
}

#[derive(Clone, Debug)]
struct ForceRow {
    location: u64,
    row_idx: usize,
    user_idx: usize,
    instance: usize,
    elem: usize,
    prob: f64,
}

pub struct PrivacyRiskResult {
    pub user_indices: Vec<usize>,
    pub risks: Vec<f64>,
    pub force_lats: Vec<f64>,
    pub force_lngs: Vec<f64>,
    pub row_indices: Vec<usize>,
    pub force_user_indices: Vec<usize>,
    pub instances: Vec<usize>,
    pub elems: Vec<usize>,
    pub probs: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttackKind {
    Location,
    Sequence,
    Time,
    UniqueLocation,
    Frequency,
    Probability,
    Proportion,
    HomeWork,
}

impl TryFrom<&str> for AttackKind {
    type Error = String;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        match value {
            "location" => Ok(Self::Location),
            "sequence" => Ok(Self::Sequence),
            "time" => Ok(Self::Time),
            "unique_location" => Ok(Self::UniqueLocation),
            "frequency" => Ok(Self::Frequency),
            "probability" => Ok(Self::Probability),
            "proportion" => Ok(Self::Proportion),
            "home_work" => Ok(Self::HomeWork),
            _ => Err("unknown privacy attack kind".to_string()),
        }
    }
}

fn loc_time_key(obs: Obs) -> (u64, u64) {
    (obs.location, obs.time_key)
}

fn starts_from_ends(ends: &[usize]) -> Vec<usize> {
    let mut starts = Vec::with_capacity(ends.len());
    let mut start = 0usize;
    for &end in ends {
        starts.push(start);
        start = end;
    }
    starts
}

fn validate_inputs(
    latitudes: &[f64],
    longitudes: &[f64],
    time_keys: Option<&[u64]>,
    indices: Option<&[usize]>,
    ends: &[usize],
    valid_rows: Option<&[bool]>,
    location_ids: &[u64],
) -> Result<(), String> {
    if latitudes.len() != longitudes.len() {
        return Err("latitudes and longitudes must have the same length".to_string());
    }
    if let Some(time_keys) = time_keys
        && time_keys.len() != latitudes.len()
    {
        return Err("time_keys and coordinates must have the same length".to_string());
    }
    if let Some(valid_rows) = valid_rows
        && valid_rows.len() != latitudes.len()
    {
        return Err("valid_rows and coordinates must have the same length".to_string());
    }
    if location_ids.len() != latitudes.len() {
        return Err("location_ids and coordinates must have the same length".to_string());
    }

    let mut previous = 0usize;
    let max_end = indices.map_or(latitudes.len(), <[usize]>::len);
    for &end in ends {
        if end < previous {
            return Err("range ends must be monotonically non-decreasing".to_string());
        }
        if end > max_end {
            return Err("range end must be within input bounds".to_string());
        }
        previous = end;
    }
    if let Some(indices) = indices {
        for &idx in indices {
            if idx >= latitudes.len() {
                return Err("index must be within coordinate array bounds".to_string());
            }
        }
    }
    Ok(())
}

fn row_is_valid(
    latitudes: &[f64],
    longitudes: &[f64],
    valid_rows: Option<&[bool]>,
    idx: usize,
) -> bool {
    valid_rows.is_none_or(|rows| rows[idx])
        && latitudes[idx].is_finite()
        && longitudes[idx].is_finite()
}

fn build_observation_users(
    latitudes: &[f64],
    longitudes: &[f64],
    time_keys: Option<&[u64]>,
    indices: Option<&[usize]>,
    ends: &[usize],
    valid_rows: Option<&[bool]>,
    location_ids: &[u64],
) -> Vec<Vec<Obs>> {
    let starts = starts_from_ends(ends);
    starts
        .into_iter()
        .zip(ends.iter().copied())
        .map(|(start, end)| {
            (start..end)
                .filter_map(|pos| {
                    let idx = indices.map_or(pos, |values| values[pos]);
                    (row_is_valid(latitudes, longitudes, valid_rows, idx)
                        && location_ids[idx] != INVALID_CELL)
                        .then_some(Obs {
                            location: location_ids[idx],
                            time_key: time_keys.map_or(0, |values| values[idx]),
                            row_idx: idx,
                        })
                })
                .collect()
        })
        .collect()
}

fn cell_center(cell: u64) -> (f64, f64) {
    let center = LatLng::from(
        CellIndex::try_from(cell).expect("privacy location ID must be a valid H3 cell"),
    );
    (center.lat(), center.lng())
}

fn sorted_location_counts(obs: &[Obs]) -> Vec<(u64, u64)> {
    let mut counts: FxHashMap<u64, u64> = FxHashMap::default();
    for item in obs {
        *counts.entry(item.location).or_insert(0) += 1;
    }
    let mut values: Vec<_> = counts
        .into_iter()
        .map(|(location, count)| {
            let (lat, lng) = cell_center(location);
            (location, count, lat, lng)
        })
        .collect();
    values.sort_unstable_by(|a, b| {
        a.1.cmp(&b.1)
            .then(a.2.total_cmp(&b.2))
            .then(a.3.total_cmp(&b.3))
    });
    values
        .into_iter()
        .map(|(location, count, _, _)| (location, count))
        .collect()
}

fn top_two_metric_values(obs: &[Obs]) -> Vec<MetricObs> {
    sorted_location_counts(obs)
        .into_iter()
        .take(2)
        .map(|(location, count)| MetricObs {
            location,
            value: count as f64,
        })
        .collect()
}

fn metric_values_for_user(obs: &[Obs], kind: AttackKind) -> Vec<MetricObs> {
    let counts = sorted_location_counts(obs);
    let total: u64 = counts.iter().map(|(_, count)| *count).sum();
    counts
        .into_iter()
        .map(|(location, count)| MetricObs {
            location,
            value: if kind == AttackKind::Probability {
                if total == 0 {
                    0.0
                } else {
                    count as f64 / total as f64
                }
            } else {
                count as f64
            },
        })
        .collect()
}

fn build_values(users: &[Vec<Obs>], kind: AttackKind) -> Vec<UserValues> {
    users
        .iter()
        .map(|obs| match kind {
            AttackKind::UniqueLocation
            | AttackKind::Frequency
            | AttackKind::Probability
            | AttackKind::Proportion => UserValues::Metrics(metric_values_for_user(obs, kind)),
            AttackKind::HomeWork => UserValues::Metrics(top_two_metric_values(obs)),
            _ => UserValues::Observations(obs.clone()),
        })
        .collect()
}

fn combination_positions(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 || n == 0 || k > n {
        return Vec::new();
    }
    let mut out = Vec::new();
    let mut combo: Vec<usize> = (0..k).collect();
    loop {
        out.push(combo.clone());
        let mut i = k;
        while i > 0 {
            i -= 1;
            if combo[i] != i + n - k {
                break;
            }
        }
        if combo[0] == n - k && i == 0 {
            break;
        }
        combo[i] += 1;
        for j in i + 1..k {
            combo[j] = combo[j - 1] + 1;
        }
    }
    out
}

fn loc_multiset(obs: &[Obs]) -> FxHashMap<u64, usize> {
    let mut counts = FxHashMap::default();
    for item in obs {
        *counts.entry(item.location).or_insert(0) += 1;
    }
    counts
}

fn loc_time_multiset(obs: &[Obs]) -> FxHashMap<(u64, u64), usize> {
    let mut counts = FxHashMap::default();
    for item in obs {
        *counts.entry(loc_time_key(*item)).or_insert(0) += 1;
    }
    counts
}

fn metric_map(values: &[MetricObs]) -> FxHashMap<u64, f64> {
    let mut out = FxHashMap::default();
    for value in values {
        out.insert(value.location, value.value);
    }
    out
}

fn prepare_candidates(values: &[UserValues]) -> Vec<PreparedCandidate<'_>> {
    values
        .iter()
        .map(|value| match value {
            UserValues::Observations(observations) => PreparedCandidate::Observations {
                values: observations,
                locations: loc_multiset(observations),
                location_times: loc_time_multiset(observations),
            },
            UserValues::Metrics(metrics) => PreparedCandidate::Metrics {
                locations: metrics.iter().map(|item| item.location).collect(),
                metrics: metric_map(metrics),
            },
        })
        .collect()
}

fn prepare_instance(values: &UserValues) -> PreparedInstance<'_> {
    match values {
        UserValues::Observations(items) => PreparedInstance::Observations {
            values: items,
            locations: loc_multiset(items),
            location_times: loc_time_multiset(items),
        },
        UserValues::Metrics(items) => PreparedInstance::Metrics {
            values: items,
            locations: items.iter().map(|item| item.location).collect(),
        },
    }
}

fn matches_location(required: &FxHashMap<u64, usize>, available: &FxHashMap<u64, usize>) -> bool {
    required
        .iter()
        .all(|(key, count)| available.get(key).copied().unwrap_or(0) >= *count)
}

fn matches_time(
    required: &FxHashMap<(u64, u64), usize>,
    available: &FxHashMap<(u64, u64), usize>,
) -> bool {
    required
        .iter()
        .all(|(key, count)| available.get(key).copied().unwrap_or(0) >= *count)
}

fn matches_sequence(instance: &[Obs], candidate: &[Obs]) -> bool {
    if instance.is_empty() {
        return true;
    }
    let mut pos = 0usize;
    for cand in candidate {
        let inst = instance[pos];
        if inst.location == cand.location {
            pos += 1;
            if pos == instance.len() {
                return true;
            }
        }
    }
    false
}

fn matches_unique_or_home(required: &FxHashSet<u64>, available: &FxHashSet<u64>) -> bool {
    required.iter().all(|key| available.contains(key))
}

fn matches_metric_tolerance(
    instance: &[MetricObs],
    candidate: &FxHashMap<u64, f64>,
    tolerance: f64,
) -> bool {
    instance.iter().all(|item| {
        candidate
            .get(&item.location)
            .is_some_and(|candidate_value| {
                item.value >= candidate_value * (1.0 - tolerance)
                    && item.value <= candidate_value * (1.0 + tolerance)
            })
    })
}

fn matches_proportion(
    instance: &[MetricObs],
    candidate: &FxHashMap<u64, f64>,
    tolerance: f64,
) -> bool {
    let max_instance = instance.iter().map(|item| item.value).fold(0.0, f64::max);
    let mut max_candidate = 0.0f64;
    for item in instance {
        let Some(candidate_value) = candidate.get(&item.location) else {
            return false;
        };
        max_candidate = max_candidate.max(*candidate_value);
    }
    if max_instance <= 0.0 || max_candidate <= 0.0 {
        return false;
    }
    instance.iter().all(|item| {
        let Some(candidate_value) = candidate.get(&item.location) else {
            return false;
        };
        let instance_prop = item.value / max_instance;
        let candidate_prop = candidate_value / max_candidate;
        instance_prop >= candidate_prop * (1.0 - tolerance)
            && instance_prop <= candidate_prop * (1.0 + tolerance)
    })
}

fn candidate_matches(
    kind: AttackKind,
    instance: &PreparedInstance<'_>,
    candidate: &PreparedCandidate<'_>,
    tolerance: f64,
) -> bool {
    match (kind, instance, candidate) {
        (
            AttackKind::Location,
            PreparedInstance::Observations {
                locations: required,
                ..
            },
            PreparedCandidate::Observations { locations, .. },
        ) => matches_location(required, locations),
        (
            AttackKind::Sequence,
            PreparedInstance::Observations {
                values: instance, ..
            },
            PreparedCandidate::Observations { values, .. },
        ) => matches_sequence(instance, values),
        (
            AttackKind::Time,
            PreparedInstance::Observations {
                location_times: required,
                ..
            },
            PreparedCandidate::Observations { location_times, .. },
        ) => matches_time(required, location_times),
        (
            AttackKind::UniqueLocation | AttackKind::HomeWork,
            PreparedInstance::Metrics {
                locations: required,
                ..
            },
            PreparedCandidate::Metrics { locations, .. },
        ) => matches_unique_or_home(required, locations),
        (
            AttackKind::Frequency | AttackKind::Probability,
            PreparedInstance::Metrics {
                values: instance, ..
            },
            PreparedCandidate::Metrics { metrics, .. },
        ) => matches_metric_tolerance(instance, metrics, tolerance),
        (
            AttackKind::Proportion,
            PreparedInstance::Metrics {
                values: instance, ..
            },
            PreparedCandidate::Metrics { metrics, .. },
        ) => matches_proportion(instance, metrics, tolerance),
        _ => false,
    }
}

fn instance_from_positions(values: &UserValues, positions: &[usize]) -> UserValues {
    match values {
        UserValues::Observations(items) => {
            UserValues::Observations(positions.iter().map(|&pos| items[pos]).collect())
        }
        UserValues::Metrics(items) => {
            UserValues::Metrics(positions.iter().map(|&pos| items[pos]).collect())
        }
    }
}

fn append_force_rows(
    rows: &mut Vec<ForceRow>,
    kind: AttackKind,
    user_idx: usize,
    instance_id: usize,
    prob: f64,
    instance: &UserValues,
) {
    match instance {
        UserValues::Observations(items) => {
            for (idx, item) in items.iter().enumerate() {
                rows.push(ForceRow {
                    location: item.location,
                    row_idx: item.row_idx,
                    user_idx,
                    instance: instance_id,
                    elem: idx + 1,
                    prob,
                });
            }
        }
        UserValues::Metrics(items) => {
            for (idx, item) in items.iter().enumerate() {
                rows.push(ForceRow {
                    location: item.location,
                    row_idx: NO_ROW,
                    user_idx,
                    instance: instance_id,
                    elem: idx + 1,
                    prob: if kind == AttackKind::Probability {
                        item.value
                    } else {
                        prob
                    },
                });
            }
        }
    }
}

fn push_result(
    normal_user_indices: &mut Vec<usize>,
    risks: &mut Vec<f64>,
    force_rows: &mut Vec<ForceRow>,
    force_instances: bool,
    kind: AttackKind,
    user_idx: usize,
    instance_id: usize,
    prob: f64,
    instance: &UserValues,
    max_prob: &mut f64,
) {
    if force_instances {
        append_force_rows(force_rows, kind, user_idx, instance_id, prob, instance);
    } else if prob > *max_prob {
        *max_prob = prob;
    }
    if !force_instances {
        let _ = normal_user_indices;
        let _ = risks;
    }
}

fn flatten_result(
    normal_user_indices: Vec<usize>,
    risks: Vec<f64>,
    force_rows: Vec<ForceRow>,
) -> PrivacyRiskResult {
    let mut lats = Vec::with_capacity(force_rows.len());
    let mut lngs = Vec::with_capacity(force_rows.len());
    let mut row_indices = Vec::with_capacity(force_rows.len());
    let mut force_user_indices = Vec::with_capacity(force_rows.len());
    let mut instances = Vec::with_capacity(force_rows.len());
    let mut elems = Vec::with_capacity(force_rows.len());
    let mut probs = Vec::with_capacity(force_rows.len());

    for row in force_rows {
        let (lat, lng) = cell_center(row.location);
        lats.push(lat);
        lngs.push(lng);
        row_indices.push(row.row_idx);
        force_user_indices.push(row.user_idx);
        instances.push(row.instance);
        elems.push(row.elem);
        probs.push(row.prob);
    }

    PrivacyRiskResult {
        user_indices: normal_user_indices,
        risks,
        force_lats: lats,
        force_lngs: lngs,
        row_indices,
        force_user_indices,
        instances,
        elems,
        probs,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn privacy_assess_risk_impl(
    latitudes: &[f64],
    longitudes: &[f64],
    time_keys: Option<&[u64]>,
    indices: Option<&[usize]>,
    ends: &[usize],
    target_user_indices: &[usize],
    attack_kind: AttackKind,
    knowledge_length: usize,
    tolerance: f64,
    force_instances: bool,
    valid_rows: Option<&[bool]>,
    location_ids: &[u64],
) -> Result<PrivacyRiskResult, String> {
    if knowledge_length == 0 {
        return Err("knowledge_length must be greater than zero".to_string());
    }
    validate_inputs(
        latitudes,
        longitudes,
        time_keys,
        indices,
        ends,
        valid_rows,
        location_ids,
    )?;

    let obs_users = build_observation_users(
        latitudes,
        longitudes,
        time_keys,
        indices,
        ends,
        valid_rows,
        location_ids,
    );
    let values = build_values(&obs_users, attack_kind);
    let candidates = prepare_candidates(&values);

    let mut normal_user_indices = Vec::new();
    let mut risks = Vec::new();
    let mut force_rows = Vec::new();

    for &user_idx in target_user_indices {
        if user_idx >= values.len() {
            continue;
        }
        let n = match &values[user_idx] {
            UserValues::Observations(items) => items.len(),
            UserValues::Metrics(items) => items.len(),
        };
        if n == 0 {
            continue;
        }

        let positions = if attack_kind == AttackKind::HomeWork {
            vec![(0..n).collect()]
        } else {
            combination_positions(n, knowledge_length.min(n))
        };

        let mut max_prob = 0.0f64;
        for (instance_idx, combo) in positions.iter().enumerate() {
            let instance = instance_from_positions(&values[user_idx], combo);
            let prepared_instance = prepare_instance(&instance);
            let match_count = candidates
                .iter()
                .filter(|candidate| {
                    candidate_matches(attack_kind, &prepared_instance, candidate, tolerance)
                })
                .count();
            if match_count == 0 {
                continue;
            }
            let prob = 1.0 / match_count as f64;
            push_result(
                &mut normal_user_indices,
                &mut risks,
                &mut force_rows,
                force_instances,
                attack_kind,
                user_idx,
                instance_idx + 1,
                prob,
                &instance,
                &mut max_prob,
            );
        }
        if !force_instances && max_prob > 0.0 {
            normal_user_indices.push(user_idx);
            risks.push(max_prob);
        }
    }

    Ok(flatten_result(normal_user_indices, risks, force_rows))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h3_location_ids_match_without_coordinate_equality() {
        let result = privacy_assess_risk_impl(
            &[37.769377, 37.769378],
            &[-122.388519, -122.388518],
            None,
            None,
            &[1, 2],
            &[0, 1],
            AttackKind::Location,
            1,
            0.0,
            false,
            None,
            &[0x8c283082e73b9ff, 0x8c283082e73b9ff],
        )
        .unwrap();

        assert_eq!(result.user_indices, vec![0, 1]);
        assert_eq!(result.risks, vec![0.5, 0.5]);
    }

    #[test]
    fn location_ids_must_match_coordinate_length() {
        let error = match privacy_assess_risk_impl(
            &[1.0],
            &[2.0],
            None,
            None,
            &[1],
            &[0],
            AttackKind::Location,
            1,
            0.0,
            false,
            None,
            &[],
        ) {
            Ok(_) => panic!("mismatched location IDs must fail validation"),
            Err(error) => error,
        };

        assert!(error.contains("location_ids"));
    }

    #[test]
    fn invalid_h3_ids_are_excluded() {
        let result = privacy_assess_risk_impl(
            &[37.769377],
            &[-122.388519],
            None,
            None,
            &[1],
            &[0],
            AttackKind::Location,
            1,
            0.0,
            false,
            None,
            &[INVALID_CELL],
        )
        .unwrap();

        assert!(result.user_indices.is_empty());
    }
}
