//! Top-Down Time Ratio (TD-TR) simplification, an iterative (explicit
//! stack, not recursive) port of MovingPandas' `TopDownTimeRatioGeneralizer`
//! (Meratnia & de By, 2004).
//!
//! Unlike Douglas-Peucker, which measures perpendicular distance from a
//! point to the *spatial* replacement line, TD-TR measures distance from a
//! point to its *spatiotemporal* projection: the position the trajectory
//! would be at that point's timestamp if it moved along a straight line at
//! constant speed between the segment's endpoints. That projection is
//! computed in the same local planar kilometre plane used by
//! Douglas-Peucker, matching MovingPandas' own choice of interpolating in
//! projected Cartesian space rather than along the great circle.

use crate::utils::haversine::project_local_planar_km;

/// Return the 0-based local indices retained by Top-Down Time Ratio
/// simplification for a single user's trajectory.
///
/// `epsilon_km` is the spatiotemporal distance tolerance, in kilometres,
/// between a point and its time-interpolated position on the candidate
/// replacement segment.
///
/// @usedBy `fastmob-core/src/preprocessing/simplify/mod.rs::simplify_user_slice`
/// (method = `top_down_time_ratio`).
pub fn top_down_time_ratio_indices(
    lats: &[f64],
    lngs: &[f64],
    times: &[f64],
    epsilon_km: f64,
) -> Vec<usize> {
    let n = lats.len();
    if n <= 2 {
        return (0..n).collect();
    }
    let coords = project_local_planar_km(lats, lngs);

    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;

    // Explicit stack of inclusive (start, end) index ranges still needing
    // subdivision, mirroring the recursive `td_tr` reference implementation
    // without recursion.
    let mut stack: Vec<(usize, usize)> = vec![(0, n - 1)];

    while let Some((start, end)) = stack.pop() {
        if end <= start + 1 {
            continue;
        }

        let t0 = times[start];
        let t1 = times[end];
        let de = t1 - t0;
        let (x0, y0) = coords[start];
        let (x1, y1) = coords[end];
        let dx = x1 - x0;
        let dy = y1 - y0;

        let mut farthest_idx = start + 1;
        let mut farthest_dist = -1.0f64;

        for idx in (start + 1)..end {
            let di = times[idx] - t0;
            let ratio = if de != 0.0 { di / de } else { 0.0 };
            let calc_x = x0 + dx * ratio;
            let calc_y = y0 + dy * ratio;
            let (px, py) = coords[idx];
            let ddx = px - calc_x;
            let ddy = py - calc_y;
            let dist = (ddx * ddx + ddy * ddy).sqrt();

            if dist > farthest_dist {
                farthest_dist = dist;
                farthest_idx = idx;
            }
        }

        if farthest_dist > epsilon_km {
            keep[farthest_idx] = true;
            stack.push((start, farthest_idx));
            stack.push((farthest_idx, end));
        }
    }

    (0..n).filter(|&i| keep[i]).collect()
}
