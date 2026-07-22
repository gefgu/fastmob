use geo::{Distance, Haversine, Point};
#[cfg(feature = "numkong")]
use numkong::Haversine as NumKongHaversine;

#[cfg(feature = "numkong")]
const GEO_HAVERSINE_RADIUS_M: f64 = 6_371_008.8;
#[cfg(feature = "numkong")]
const NUMKONG_HAVERSINE_RADIUS_M: f64 = 6_335_439.0;
#[cfg(feature = "numkong")]
const NUMKONG_TO_GEO_KM: f64 = GEO_HAVERSINE_RADIUS_M / NUMKONG_HAVERSINE_RADIUS_M / 1000.0;

pub fn haversine_km(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let p1 = Point::new(lon1, lat1);
    let p2 = Point::new(lon2, lat2);
    Haversine.distance(p1, p2) / 1000.0
}

pub fn adjacent_haversine_distances_km(
    latitudes: &[f64],
    longitudes: &[f64],
    start: usize,
    end: usize,
) -> Vec<f64> {
    if end - start < 2 {
        return Vec::new();
    }

    let mut distances = vec![0.0; end - start - 1];
    adjacent_haversine_distances_into_km(latitudes, longitudes, start, end, &mut distances);
    distances
}

pub fn adjacent_haversine_distances_into_km(
    latitudes: &[f64],
    longitudes: &[f64],
    start: usize,
    end: usize,
    distances: &mut [f64],
) {
    if end - start < 2 {
        return;
    }

    #[cfg(feature = "numkong")]
    {
        let latitudes_rad: Vec<f64> = latitudes[start..end]
            .iter()
            .map(|lat| lat.to_radians())
            .collect();
        let longitudes_rad: Vec<f64> = longitudes[start..end]
            .iter()
            .map(|lon| lon.to_radians())
            .collect();

        f64::haversine(
            &latitudes_rad[..latitudes_rad.len() - 1],
            &longitudes_rad[..longitudes_rad.len() - 1],
            &latitudes_rad[1..],
            &longitudes_rad[1..],
            distances,
        )
        .expect("adjacent coordinate slices have matching lengths");

        distances
            .iter_mut()
            .for_each(|distance| *distance *= NUMKONG_TO_GEO_KM);
    }

    // Pure-Rust fallback (no numkong): compute each adjacent distance with the
    // geo-backed single-pair haversine. Results are in the same geo radius as the
    // scaled SIMD path above.
    #[cfg(not(feature = "numkong"))]
    for (idx, distance) in distances.iter_mut().enumerate() {
        *distance = haversine_km(
            latitudes[start + idx],
            longitudes[start + idx],
            latitudes[start + idx + 1],
            longitudes[start + idx + 1],
        );
    }
}

pub fn adjacent_haversine_sum_km(
    latitudes: &[f64],
    longitudes: &[f64],
    start: usize,
    end: usize,
) -> f64 {
    adjacent_haversine_distances_km(latitudes, longitudes, start, end)
        .into_iter()
        .sum()
}

pub fn adjacent_haversine_max_km(
    latitudes: &[f64],
    longitudes: &[f64],
    start: usize,
    end: usize,
) -> f64 {
    adjacent_haversine_distances_km(latitudes, longitudes, start, end)
        .into_iter()
        .fold(0.0f64, f64::max)
}

/// Mean earth radius in kilometres, matching `geo`'s Haversine radius
/// (`GEO_HAVERSINE_RADIUS_M` above, expressed in km) so planar-projected
/// distances stay consistent with `haversine_km`.
const EARTH_RADIUS_KM: f64 = 6371.0088;

/// Project a slice of lat/lng points to local equirectangular planar
/// kilometre coordinates around the slice's mean latitude.
///
/// Returns one `(x_km, y_km)` pair per input point, in input order. `x_km`
/// grows eastward and `y_km` grows northward. The projection is only locally
/// accurate (valid for a single user's trajectory extent, not global data),
/// which is why algorithms that need real 2D vector geometry (Douglas-Peucker,
/// MaxDistance, Chan-Chin, Imai-Iri) project per-user slices rather than a
/// whole dataframe at once.
///
/// @usedBy `fastmob-core/src/preprocessing/simplify/{douglas_peucker,
/// distance_time_threshold,corridor}.rs` for building per-user planar
/// coordinates before computing point-to-line / point-to-segment distances.
pub fn project_local_planar_km(latitudes: &[f64], longitudes: &[f64]) -> Vec<(f64, f64)> {
    debug_assert_eq!(latitudes.len(), longitudes.len());
    let n = latitudes.len();
    if n == 0 {
        return Vec::new();
    }

    let mean_lat_rad = (latitudes.iter().sum::<f64>() / n as f64).to_radians();
    let cos_mean_lat = mean_lat_rad.cos();

    latitudes
        .iter()
        .zip(longitudes.iter())
        .map(|(&lat, &lng)| {
            let x = EARTH_RADIUS_KM * lng.to_radians() * cos_mean_lat;
            let y = EARTH_RADIUS_KM * lat.to_radians();
            (x, y)
        })
        .collect()
}

/// Shortest planar Euclidean distance from `point` to the segment `a`-`b`.
///
/// All inputs are `(x_km, y_km)` pairs already produced by
/// [`project_local_planar_km`]; the result is in the same km units. When `a`
/// and `b` coincide, this is simply the distance from `point` to `a`.
///
/// @usedBy `fastmob-core/src/preprocessing/simplify/{douglas_peucker,
/// distance_time_threshold,corridor}.rs` for perpendicular-distance and
/// corridor-feasibility checks against a candidate simplified segment.
pub fn point_to_segment_distance_km(point: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (px, py) = point;
    let (ax, ay) = a;
    let (bx, by) = b;

    let dx = bx - ax;
    let dy = by - ay;
    let len_sq = dx * dx + dy * dy;

    let (proj_x, proj_y) = if len_sq == 0.0 {
        (ax, ay)
    } else {
        let t = ((px - ax) * dx + (py - ay) * dy) / len_sq;
        let t_clamped = t.clamp(0.0, 1.0);
        (ax + t_clamped * dx, ay + t_clamped * dy)
    };

    let ddx = px - proj_x;
    let ddy = py - proj_y;
    (ddx * ddx + ddy * ddy).sqrt()
}
