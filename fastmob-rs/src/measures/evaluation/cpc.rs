//! Common Part of Commuters between two trajectories.

use fastmob_core::measures::evaluation::trajectory_cpc::trajectory_common_part_of_commuters_impl;

use crate::error::FastmobRsError;
use crate::prepare::PreparedTrajectory;

/// CPC between two prepared trajectories at one H3 resolution.
///
/// The kernel takes a permutation plus group ends because the Python path hands
/// it an unsorted frame. A `PreparedTrajectory` is already physically arranged,
/// so the permutation is the identity and the arrangement is shared with every
/// other measure rather than rebuilt here.
pub fn common_part_of_commuters(
    a: &PreparedTrajectory,
    b: &PreparedTrajectory,
    resolution: u8,
) -> Result<f64, FastmobRsError> {
    common_part_of_commuters_multi(a, b, &[resolution]).map(|values| values[0].1)
}

/// CPC at several H3 resolutions, sharing one identity permutation per side.
pub fn common_part_of_commuters_multi(
    a: &PreparedTrajectory,
    b: &PreparedTrajectory,
    resolutions: &[u8],
) -> Result<Vec<(u8, f64)>, FastmobRsError> {
    let indices_a: Vec<usize> = (0..a.height()).collect();
    let indices_b: Vec<usize> = (0..b.height()).collect();

    resolutions
        .iter()
        .map(|&resolution| {
            let value = trajectory_common_part_of_commuters_impl(
                a.lat(),
                a.lng(),
                &indices_a,
                a.ends(),
                b.lat(),
                b.lng(),
                &indices_b,
                b.ends(),
                resolution,
            )
            .map_err(FastmobRsError::Core)?;
            Ok((resolution, value))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use polars::prelude::*;

    use super::*;
    use crate::prepare::{Cols, prepare};

    fn trajectory(lats: [f64; 4], lngs: [f64; 4]) -> PreparedTrajectory {
        let df = df![
            "uid" => [1i64, 1, 2, 2],
            "datetime" => [0i64, 1, 0, 1],
            "lat" => lats.to_vec(),
            "lng" => lngs.to_vec(),
        ]
        .unwrap();
        prepare(&df, Cols::auto()).unwrap()
    }

    #[test]
    fn a_trajectory_against_itself_is_fully_common() {
        let traj = trajectory([48.85, 48.86, 40.71, 40.72], [2.35, 2.36, -74.01, -74.00]);
        let value = common_part_of_commuters(&traj, &traj, 7).unwrap();
        assert!((value - 1.0).abs() < 1e-12, "got {value}");
    }

    #[test]
    fn disjoint_regions_share_nothing() {
        let paris = trajectory([48.85, 48.86, 48.87, 48.88], [2.35, 2.36, 2.37, 2.38]);
        let tokyo = trajectory(
            [35.68, 35.69, 35.70, 35.71],
            [139.69, 139.70, 139.71, 139.72],
        );
        assert_eq!(common_part_of_commuters(&paris, &tokyo, 7).unwrap(), 0.0);
    }

    #[test]
    fn multi_resolution_shares_one_arrangement() {
        let traj = trajectory([48.85, 48.86, 40.71, 40.72], [2.35, 2.36, -74.01, -74.00]);
        let values = common_part_of_commuters_multi(&traj, &traj, &[6, 7, 8]).unwrap();
        assert_eq!(values.len(), 3);
        assert_eq!(
            values.iter().map(|(res, _)| *res).collect::<Vec<_>>(),
            vec![6, 7, 8]
        );
    }

    #[test]
    fn an_invalid_resolution_is_rejected() {
        let traj = trajectory([48.85, 48.86, 40.71, 40.72], [2.35, 2.36, -74.01, -74.00]);
        assert!(common_part_of_commuters(&traj, &traj, 16).is_err());
    }
}
