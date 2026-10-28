//! Two-pass, batch-at-a-time radius of gyration over raw rows.
//!
//! The eager kernels in [`super::radius_of_gyration`] see every row of a user at
//! once. A streamed source hands over one batch at a time, and a user's rows may
//! be spread over many batches, so the computation is split into the two
//! reductions radius of gyration is made of:
//!
//! 1. per-user coordinate sums and counts, which fix each centre of mass;
//! 2. per-user sums of squared haversine distances to that centre.
//!
//! Radius of gyration is order-independent, so neither pass builds per-user
//! row ranges. Each batch is cut into row units that are processed in
//! parallel: a unit resolves its rows' user IDs to unit-local ids (a "same
//! user as the previous row" check first, so grouped input almost never
//! hashes), accumulates partials per local id, and the units are then merged
//! into the global per-user state in row order. Merging in row order is what
//! keeps user slots in first-appearance order, matching the eager output.

use std::hash::Hash;

use rayon::prelude::*;
use rustc_hash::FxHashMap;

use super::radius_of_gyration::{RogCenter, sq_haversine_to_center};

/// Rows per parallel work unit. Large enough to amortize scheduling, small
/// enough that a few-chunk batch still spreads over every core.
const UNIT_ROWS: usize = 1 << 15;

/// Borrowed user-ID value of one row; the key type the per-row loops compare.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UidKeyRef<'a> {
    Null,
    Int(i128),
    /// IEEE-754 total-order bits, so that NaN and -0.0 group consistently.
    Float(u64),
    Bool(bool),
    Bytes(&'a [u8]),
}

/// Owned user-ID value, as kept in the cross-batch codebook.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UidKey {
    Null,
    Int(i128),
    Float(u64),
    Bool(bool),
    Bytes(Box<[u8]>),
}

impl UidKeyRef<'_> {
    pub fn to_owned_key(self) -> UidKey {
        match self {
            UidKeyRef::Null => UidKey::Null,
            UidKeyRef::Int(v) => UidKey::Int(v),
            UidKeyRef::Float(v) => UidKey::Float(v),
            UidKeyRef::Bool(v) => UidKey::Bool(v),
            UidKeyRef::Bytes(v) => UidKey::Bytes(v.into()),
        }
    }
}

/// One contiguous slice of a batch: user IDs (via `keys`), coordinates, and an
/// optional validity mask for rows whose coordinates are Arrow-null.
pub struct RowChunk<'a, K> {
    pub keys: K,
    pub latitudes: &'a [f64],
    pub longitudes: &'a [f64],
    pub valid: Option<&'a [bool]>,
}

impl<K> RowChunk<'_, K> {
    fn validate(&self) -> Result<(), String> {
        if self.latitudes.len() != self.longitudes.len() {
            return Err("latitudes and longitudes must have the same length".to_string());
        }
        if self.valid.is_some_and(|v| v.len() != self.latitudes.len()) {
            return Err("validity mask must match the coordinate length".to_string());
        }
        Ok(())
    }

    #[inline]
    fn coordinate(&self, i: usize) -> Option<(f64, f64)> {
        let (lat, lng) = (self.latitudes[i], self.longitudes[i]);
        (self.valid.is_none_or(|v| v[i]) && lat.is_finite() && lng.is_finite())
            .then_some((lat, lng))
    }
}

/// Resolves a unit's rows to dense unit-local ids, in first-appearance order.
struct LocalIds<'a, T> {
    ids: FxHashMap<UidKeyRef<'a>, u32>,
    keys: Vec<UidKeyRef<'a>>,
    values: Vec<T>,
    previous: Option<(UidKeyRef<'a>, u32)>,
}

impl<'a, T> LocalIds<'a, T> {
    fn new() -> Self {
        Self {
            ids: FxHashMap::default(),
            keys: Vec::new(),
            values: Vec::new(),
            previous: None,
        }
    }

    /// Local id of `key`; `on_new` runs once per unseen key.
    #[inline]
    fn id(&mut self, key: UidKeyRef<'a>, on_new: impl FnOnce(UidKeyRef<'a>) -> T) -> u32 {
        if let Some((previous, id)) = self.previous {
            if previous == key {
                return id;
            }
        }
        let next = self.keys.len() as u32;
        let id = *self.ids.entry(key).or_insert(next);
        if id == next {
            self.keys.push(key);
            self.values.push(on_new(key));
        }
        self.previous = Some((key, id));
        id
    }
}

#[derive(Clone, Copy, Default)]
struct Moments {
    lat: f64,
    lng: f64,
    n: u64,
}

#[derive(Clone, Copy, Default)]
struct Squares {
    slot: u32,
    sq: f64,
    n: u64,
}

/// `(chunk, start, end)` work units of at most [`UNIT_ROWS`] rows, in row order.
fn units<K>(chunks: &[RowChunk<'_, K>]) -> Vec<(usize, usize, usize)> {
    chunks
        .iter()
        .enumerate()
        .flat_map(|(c, chunk)| {
            let len = chunk.latitudes.len();
            (0..len)
                .step_by(UNIT_ROWS)
                .map(move |start| (c, start, (start + UNIT_ROWS).min(len)))
        })
        .collect()
}

#[derive(Debug, Default)]
pub struct RogRowStream {
    codebook: FxHashMap<UidKey, u32>,
    labels: Vec<UidKey>,
    lat_sum: Vec<f64>,
    lng_sum: Vec<f64>,
    count: Vec<u64>,
    sq_sum: Vec<f64>,
    sq_count: Vec<u64>,
    centers: Option<Vec<RogCenter>>,
}

impl RogRowStream {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn num_users(&self) -> usize {
        self.labels.len()
    }

    /// Pass 1: accumulate per-user coordinate sums over one batch.
    pub fn accumulate_moments<'a, K>(&mut self, chunks: &[RowChunk<'a, K>]) -> Result<(), String>
    where
        K: Fn(usize) -> UidKeyRef<'a> + Sync,
    {
        if self.centers.is_some() {
            return Err("moment pass already finalized; call accumulate_sq instead".to_string());
        }
        chunks.iter().try_for_each(RowChunk::validate)?;

        let partials: Vec<LocalIds<'a, Moments>> = units(chunks)
            .into_par_iter()
            .map(|(c, start, end)| {
                let chunk = &chunks[c];
                let mut local = LocalIds::new();
                for i in start..end {
                    let id = local.id((chunk.keys)(i), |_| Moments::default());
                    if let Some((lat, lng)) = chunk.coordinate(i) {
                        let m = &mut local.values[id as usize];
                        m.lat += lat;
                        m.lng += lng;
                        m.n += 1;
                    }
                }
                local
            })
            .collect();

        for local in partials {
            for (key, m) in local.keys.into_iter().zip(local.values) {
                let slot = self.slot_or_insert(key)?;
                self.lat_sum[slot] += m.lat;
                self.lng_sum[slot] += m.lng;
                self.count[slot] += m.n;
            }
        }
        Ok(())
    }

    fn slot_or_insert(&mut self, key: UidKeyRef<'_>) -> Result<usize, String> {
        let owned = key.to_owned_key();
        if let Some(&slot) = self.codebook.get(&owned) {
            return Ok(slot as usize);
        }
        let slot = u32::try_from(self.labels.len())
            .map_err(|_| "more than u32::MAX distinct user IDs".to_string())?;
        self.codebook.insert(owned.clone(), slot);
        self.labels.push(owned);
        self.lat_sum.push(0.0);
        self.lng_sum.push(0.0);
        self.count.push(0);
        Ok(slot as usize)
    }

    /// Fix each user's centre of mass; ends pass 1.
    pub fn finalize_centers(&mut self) -> Result<(), String> {
        if self.centers.is_some() {
            return Err("finalize_centers was already called".to_string());
        }
        let centers = (0..self.labels.len())
            .map(|slot| {
                RogCenter::from_sums(
                    self.lat_sum[slot],
                    self.lng_sum[slot],
                    self.count[slot] as usize,
                )
            })
            .collect();
        self.centers = Some(centers);
        self.sq_sum = vec![0.0; self.labels.len()];
        self.sq_count = vec![0; self.labels.len()];
        Ok(())
    }

    /// Pass 2: accumulate per-user squared distances to the centres over one batch.
    pub fn accumulate_sq<'a, K>(&mut self, chunks: &[RowChunk<'a, K>]) -> Result<(), String>
    where
        K: Fn(usize) -> UidKeyRef<'a> + Sync,
    {
        let centers = self
            .centers
            .as_deref()
            .ok_or_else(|| "call finalize_centers before accumulate_sq".to_string())?;
        chunks.iter().try_for_each(RowChunk::validate)?;
        let codebook = &self.codebook;

        let partials: Vec<Result<Vec<Squares>, String>> = units(chunks)
            .into_par_iter()
            .map(|(c, start, end)| {
                let chunk = &chunks[c];
                let mut local = LocalIds::<Squares>::new();
                let mut unseen = false;
                for i in start..end {
                    let id = local.id((chunk.keys)(i), |key| {
                        let slot = codebook.get(&key.to_owned_key()).copied();
                        unseen |= slot.is_none();
                        Squares {
                            slot: slot.unwrap_or(0),
                            ..Squares::default()
                        }
                    });
                    if unseen {
                        return Err("a user ID was not seen in the first pass; \
                                    the source must yield the same rows on every pass"
                            .to_string());
                    }
                    if let Some((lat, lng)) = chunk.coordinate(i) {
                        let s = &mut local.values[id as usize];
                        s.sq += sq_haversine_to_center(lat, lng, &centers[s.slot as usize]);
                        s.n += 1;
                    }
                }
                Ok(local.values)
            })
            .collect();

        for local in partials {
            for s in local? {
                self.sq_sum[s.slot as usize] += s.sq;
                self.sq_count[s.slot as usize] += s.n;
            }
        }
        Ok(())
    }

    /// User labels in first-appearance order, with each user's radius of
    /// gyration (km) and validity (`false` for users with no valid coordinates).
    pub fn finalize(&self) -> Result<(&[UidKey], Vec<f64>, Vec<bool>), String> {
        if self.centers.is_none() {
            return Err("call finalize_centers and accumulate_sq before finalize".to_string());
        }
        if self.sq_count != self.count {
            return Err(
                "the two passes saw different rows; the source must yield the same rows on every pass"
                    .to_string(),
            );
        }
        let values = self
            .count
            .iter()
            .zip(&self.sq_sum)
            .map(|(&n, &sq)| if n == 0 { 0.0 } else { (sq / n as f64).sqrt() })
            .collect();
        let validity = self.count.iter().map(|&n| n > 0).collect();
        Ok((&self.labels, values, validity))
    }
}

#[cfg(test)]
mod tests {
    use super::super::radius_of_gyration::radius_of_gyration_indexed_impl;
    use super::*;

    fn chunk<'a>(
        uids: &'a [i64],
        lats: &'a [f64],
        lngs: &'a [f64],
    ) -> RowChunk<'a, impl Fn(usize) -> UidKeyRef<'a> + Sync + 'a> {
        RowChunk {
            keys: move |i: usize| UidKeyRef::Int(uids[i] as i128),
            latitudes: lats,
            longitudes: lngs,
            valid: None,
        }
    }

    fn eager(uids: &[i64], lats: &[f64], lngs: &[f64], order: &[i64]) -> (Vec<f64>, Vec<bool>) {
        let mut indices = Vec::new();
        let mut ends = Vec::new();
        for &user in order {
            indices.extend((0..uids.len()).filter(|&i| uids[i] == user));
            ends.push(indices.len());
        }
        radius_of_gyration_indexed_impl(lats, lngs, &indices, &ends, None)
    }

    fn assert_close(a: &[f64], b: &[f64]) {
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b) {
            assert!((x - y).abs() <= 1e-9 * y.abs().max(1.0), "{x} != {y}");
        }
    }

    /// Interleaved users, one all-NaN user, split across two batches and
    /// several chunks per batch.
    #[test]
    fn split_batches_match_eager() {
        let uids = [7i64, 3, 7, 9, 3, 7, 3];
        let lats = [45.0, 10.0, 45.5, f64::NAN, 10.2, 46.0, 11.0];
        let lngs = [9.0, 20.0, 9.3, 1.0, 20.5, 9.9, 21.0];
        let (expected, expected_valid) = eager(&uids, &lats, &lngs, &[7, 3, 9]);

        let first = [
            chunk(&uids[..2], &lats[..2], &lngs[..2]),
            chunk(&uids[2..4], &lats[2..4], &lngs[2..4]),
        ];
        let second = [chunk(&uids[4..], &lats[4..], &lngs[4..])];
        let mut stream = RogRowStream::new();
        stream.accumulate_moments(&first).unwrap();
        stream.accumulate_moments(&second).unwrap();
        stream.finalize_centers().unwrap();
        stream.accumulate_sq(&second).unwrap();
        stream.accumulate_sq(&first).unwrap();
        let (labels, values, valid) = stream.finalize().unwrap();
        assert_eq!(labels, [UidKey::Int(7), UidKey::Int(3), UidKey::Int(9)]);
        assert_eq!(valid, expected_valid);
        assert_close(&values, &expected);
    }

    /// More rows than one work unit, so units are merged across threads.
    #[test]
    fn multi_unit_chunk_matches_eager() {
        let n = UNIT_ROWS * 3 + 17;
        let uids: Vec<i64> = (0..n).map(|i| ((i * 7919) % 13) as i64).collect();
        let lats: Vec<f64> = (0..n).map(|i| -60.0 + (i % 1000) as f64 * 0.12).collect();
        let lngs: Vec<f64> = (0..n).map(|i| -170.0 + (i % 977) as f64 * 0.34).collect();
        let mut order = Vec::new();
        for &u in &uids {
            if !order.contains(&u) {
                order.push(u);
            }
        }
        let (expected, _) = eager(&uids, &lats, &lngs, &order);

        let batch = [chunk(&uids, &lats, &lngs)];
        let mut stream = RogRowStream::new();
        stream.accumulate_moments(&batch).unwrap();
        stream.finalize_centers().unwrap();
        stream.accumulate_sq(&batch).unwrap();
        let (labels, values, _) = stream.finalize().unwrap();
        let expected_labels: Vec<UidKey> = order.iter().map(|&u| UidKey::Int(u as i128)).collect();
        assert_eq!(labels, expected_labels.as_slice());
        assert_close(&values, &expected);
    }

    #[test]
    fn unseen_user_or_row_mismatch_in_second_pass_errors() {
        let (uids, lats, lngs) = ([1i64, 1], [1.0, 2.0], [1.0, 2.0]);
        let mut stream = RogRowStream::new();
        stream
            .accumulate_moments(&[chunk(&uids, &lats, &lngs)])
            .unwrap();
        stream.finalize_centers().unwrap();
        let other = [2i64];
        assert!(
            stream
                .accumulate_sq(&[chunk(&other, &lats[..1], &lngs[..1])])
                .is_err()
        );
        stream
            .accumulate_sq(&[chunk(&uids[..1], &lats[..1], &lngs[..1])])
            .unwrap();
        assert!(stream.finalize().is_err());
    }
}
