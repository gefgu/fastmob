//! Sparse collective interest-network construction.

use rustc_hash::FxHashMap;

/// Count users that visited each unordered pair of distinct locations.
///
/// `user_codes` and `location_ranks` must contain one row per distinct
/// user-location membership. Location ranks are the row positions in the
/// caller's global location catalogue, which gives edges a stable orientation
/// without imposing an order on the location IDs themselves.
pub fn interest_network_impl(
    user_codes: &[u32],
    location_ranks: &[u32],
) -> Result<Vec<(u32, u32, u64)>, String> {
    if user_codes.len() != location_ranks.len() {
        return Err("user_codes and location_ranks must have the same length".to_string());
    }

    let mut locations_by_user: FxHashMap<u32, Vec<u32>> = FxHashMap::default();
    for (&user, &location) in user_codes.iter().zip(location_ranks) {
        locations_by_user.entry(user).or_default().push(location);
    }

    let mut counts: FxHashMap<(u32, u32), u64> = FxHashMap::default();
    for locations in locations_by_user.values_mut() {
        locations.sort_unstable();
        locations.dedup();
        for (index, &left) in locations.iter().enumerate() {
            for &right in &locations[index + 1..] {
                *counts.entry((left, right)).or_insert(0) += 1;
            }
        }
    }

    let mut edges: Vec<_> = counts
        .into_iter()
        .map(|((left, right), count)| (left, right, count))
        .collect();
    edges.sort_unstable_by_key(|&(left, right, _)| (left, right));
    Ok(edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_distinct_users_per_catalogue_ordered_pair() {
        let edges = interest_network_impl(&[0, 0, 1, 1, 2], &[2, 5, 2, 7, 5]).unwrap();
        assert_eq!(edges, vec![(2, 5, 1), (2, 7, 1)]);
    }

    #[test]
    fn duplicate_memberships_do_not_inflate_counts_or_make_self_loops() {
        let edges = interest_network_impl(&[0, 0, 0, 1, 1], &[3, 3, 9, 3, 9]).unwrap();
        assert_eq!(edges, vec![(3, 9, 2)]);
    }

    #[test]
    fn rejects_mismatched_inputs() {
        assert!(interest_network_impl(&[0], &[]).is_err());
    }
}
