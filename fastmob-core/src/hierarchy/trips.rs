use std::collections::BTreeMap;

use crate::utils::haversine::haversine_km;

pub const NULL_I64: i64 = i64::MIN;

pub struct TriplegLengthsResult {
    pub uid_codes: Vec<u64>,
    pub segment_ids: Vec<i64>,
    pub lengths_km: Vec<f64>,
}

pub struct TripsResult {
    pub uid_codes: Vec<u64>,
    pub started_at_us: Vec<i64>,
    pub finished_at_us: Vec<i64>,
    pub origin_staypoint_ids: Vec<i64>,
    pub destination_staypoint_ids: Vec<i64>,
    pub tripleg_offsets: Vec<usize>,
    pub tripleg_ids: Vec<i64>,
}

pub struct ToursResult {
    pub uid_codes: Vec<u64>,
    pub started_at_us: Vec<i64>,
    pub finished_at_us: Vec<i64>,
    pub location_ids: Vec<i64>,
    pub journey_offsets: Vec<usize>,
    pub journey_trip_ids: Vec<i64>,
}

#[allow(clippy::too_many_arguments)]
pub fn tripleg_lengths_attributed_impl(
    uid_codes: &[u64],
    segment_ids: &[i64],
    is_stop: &[bool],
    latitudes: &[f64],
    longitudes: &[f64],
) -> Result<TriplegLengthsResult, String> {
    let n = uid_codes.len();
    if segment_ids.len() != n || is_stop.len() != n || latitudes.len() != n || longitudes.len() != n
    {
        return Err("tripleg length input arrays must have the same length".to_string());
    }

    let mut lengths: BTreeMap<(u64, i64), f64> = BTreeMap::new();
    for i in 0..n {
        let first_of_user = i == 0 || uid_codes[i] != uid_codes[i - 1];
        if first_of_user {
            continue;
        }

        let first_of_segment = segment_ids[i] != segment_ids[i - 1];
        let attr_segment_id = if first_of_segment && is_stop[i] {
            segment_ids[i - 1]
        } else {
            segment_ids[i]
        };
        let distance = haversine_km(
            latitudes[i - 1],
            longitudes[i - 1],
            latitudes[i],
            longitudes[i],
        );
        *lengths
            .entry((uid_codes[i], attr_segment_id))
            .or_insert(0.0) += distance;
    }

    let mut uid_out = Vec::with_capacity(lengths.len());
    let mut segment_out = Vec::with_capacity(lengths.len());
    let mut length_out = Vec::with_capacity(lengths.len());
    for ((uid, segment_id), length) in lengths {
        uid_out.push(uid);
        segment_out.push(segment_id);
        length_out.push(length);
    }

    Ok(TriplegLengthsResult {
        uid_codes: uid_out,
        segment_ids: segment_out,
        lengths_km: length_out,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn trips_from_timeline_impl(
    uid_codes: &[u64],
    kind_codes: &[u8],
    activity: &[bool],
    staypoint_ids: &[i64],
    tripleg_ids: &[i64],
    started_at_us: &[i64],
    finished_at_us: &[i64],
) -> Result<TripsResult, String> {
    let n = uid_codes.len();
    if kind_codes.len() != n
        || activity.len() != n
        || staypoint_ids.len() != n
        || tripleg_ids.len() != n
        || started_at_us.len() != n
        || finished_at_us.len() != n
    {
        return Err("trip timeline input arrays must have the same length".to_string());
    }

    let mut origin = vec![NULL_I64; n];
    let mut destination = vec![NULL_I64; n];
    let mut local_trip_idx = vec![0i64; n];

    let mut last_seen = NULL_I64;
    for i in 0..n {
        if i == 0 || uid_codes[i] != uid_codes[i - 1] {
            last_seen = NULL_I64;
        }
        if kind_codes[i] == 0 && activity[i] {
            last_seen = staypoint_ids[i];
        }
        origin[i] = last_seen;
    }

    let mut next_seen = NULL_I64;
    for i in (0..n).rev() {
        if i + 1 < n && uid_codes[i] != uid_codes[i + 1] {
            next_seen = NULL_I64;
        }
        if kind_codes[i] == 0 && activity[i] {
            next_seen = staypoint_ids[i];
        }
        destination[i] = next_seen;
    }

    let mut running = 0i64;
    for i in 0..n {
        if i == 0 || uid_codes[i] != uid_codes[i - 1] {
            running = 0;
        }
        local_trip_idx[i] = running;
        if kind_codes[i] == 0 && activity[i] {
            running += 1;
        }
    }

    #[derive(Clone)]
    struct TripAccumulator {
        uid_code: u64,
        started_at_us: i64,
        finished_at_us: i64,
        origin_staypoint_id: i64,
        destination_staypoint_id: i64,
        tripleg_ids: Vec<i64>,
    }

    let mut trips: BTreeMap<(u64, i64), usize> = BTreeMap::new();
    let mut order: Vec<(u64, i64)> = Vec::new();
    let mut acc: Vec<TripAccumulator> = Vec::new();
    for i in 0..n {
        if kind_codes[i] != 1 {
            continue;
        }
        let key = (uid_codes[i], local_trip_idx[i]);
        let idx = if let Some(idx) = trips.get(&key) {
            *idx
        } else {
            let idx = acc.len();
            trips.insert(key, idx);
            order.push(key);
            acc.push(TripAccumulator {
                uid_code: uid_codes[i],
                started_at_us: started_at_us[i],
                finished_at_us: finished_at_us[i],
                origin_staypoint_id: origin[i],
                destination_staypoint_id: destination[i],
                tripleg_ids: Vec::new(),
            });
            idx
        };
        let trip = &mut acc[idx];
        trip.started_at_us = trip.started_at_us.min(started_at_us[i]);
        trip.finished_at_us = trip.finished_at_us.max(finished_at_us[i]);
        trip.tripleg_ids.push(tripleg_ids[i]);
    }

    let mut uid_out = Vec::with_capacity(order.len());
    let mut started_out = Vec::with_capacity(order.len());
    let mut finished_out = Vec::with_capacity(order.len());
    let mut origin_out = Vec::with_capacity(order.len());
    let mut destination_out = Vec::with_capacity(order.len());
    let mut offsets = Vec::with_capacity(order.len() + 1);
    let mut flat_triplegs = Vec::new();
    offsets.push(0);
    for key in order {
        let trip = &acc[trips[&key]];
        uid_out.push(trip.uid_code);
        started_out.push(trip.started_at_us);
        finished_out.push(trip.finished_at_us);
        origin_out.push(trip.origin_staypoint_id);
        destination_out.push(trip.destination_staypoint_id);
        flat_triplegs.extend_from_slice(&trip.tripleg_ids);
        offsets.push(flat_triplegs.len());
    }

    Ok(TripsResult {
        uid_codes: uid_out,
        started_at_us: started_out,
        finished_at_us: finished_out,
        origin_staypoint_ids: origin_out,
        destination_staypoint_ids: destination_out,
        tripleg_offsets: offsets,
        tripleg_ids: flat_triplegs,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn tours_from_trips_impl(
    uid_codes: &[u64],
    trip_ids: &[i64],
    started_at_us: &[i64],
    finished_at_us: &[i64],
    origin_location_ids: &[i64],
    destination_location_ids: &[i64],
) -> Result<ToursResult, String> {
    let n = uid_codes.len();
    if trip_ids.len() != n
        || started_at_us.len() != n
        || finished_at_us.len() != n
        || origin_location_ids.len() != n
        || destination_location_ids.len() != n
    {
        return Err("tour input arrays must have the same length".to_string());
    }

    let mut uid_out = Vec::new();
    let mut started_out = Vec::new();
    let mut finished_out = Vec::new();
    let mut location_out = Vec::new();
    let mut offsets = Vec::new();
    let mut journey = Vec::new();
    offsets.push(0);

    let mut i = 0usize;
    while i < n {
        let uid = uid_codes[i];
        let mut user_end = i + 1;
        while user_end < n && uid_codes[user_end] == uid {
            user_end += 1;
        }

        let mut j = i;
        while j < user_end {
            let origin_loc = origin_location_ids[j];
            if origin_loc == NULL_I64 {
                j += 1;
                continue;
            }

            let mut closing = None;
            for (k, &dest_loc) in destination_location_ids
                .iter()
                .enumerate()
                .take(user_end)
                .skip(j)
            {
                if dest_loc == origin_loc {
                    closing = Some(k);
                    break;
                }
            }

            if let Some(k) = closing {
                uid_out.push(uid);
                started_out.push(started_at_us[j]);
                finished_out.push(finished_at_us[k]);
                location_out.push(origin_loc);
                journey.extend_from_slice(&trip_ids[j..=k]);
                offsets.push(journey.len());
                j = k + 1;
            } else {
                j += 1;
            }
        }

        i = user_end;
    }

    Ok(ToursResult {
        uid_codes: uid_out,
        started_at_us: started_out,
        finished_at_us: finished_out,
        location_ids: location_out,
        journey_offsets: offsets,
        journey_trip_ids: journey,
    })
}
