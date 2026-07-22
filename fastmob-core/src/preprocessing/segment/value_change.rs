//! ValueChange segmentation, ported from MovingPandas'
//! `ValueChangeSplitter` (`trajectory_splitter.py`).
//!
//! Also backs the `Temporal` method (MovingPandas' `TemporalSplitter`): the
//! Python wrapper truncates the datetime column into an integer bucket id
//! per row (see `fastmob/preprocessing/_segment.py::_prepare_temporal`) and
//! delegates to this same generic bucket-id-change kernel instead of a
//! dedicated Rust kernel, avoiding a redundant duplicate implementation.

/// ValueChange segment ids for one user's chronologically-sorted slice.
///
/// Point `0` always starts segment `0`. For `i >= 1`, a new segment starts
/// at `i` whenever `bucket_ids[i] != bucket_ids[i-1]`.
///
/// @usedBy `fastmob-core/src/preprocessing/segment/mod.rs::segment_user_slice`
/// (method = `value_change`, which also serves `temporal`).
pub fn value_change_segment_ids(bucket_ids: &[i64]) -> Vec<u32> {
    let n = bucket_ids.len();
    let mut out = vec![0u32; n];
    if n == 0 {
        return out;
    }

    let mut group = 0u32;
    for i in 1..n {
        if bucket_ids[i] != bucket_ids[i - 1] {
            group += 1;
        }
        out[i] = group;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_and_single_point_inputs() {
        assert_eq!(value_change_segment_ids(&[]), Vec::<u32>::new());
        assert_eq!(value_change_segment_ids(&[7]), vec![0]);
    }

    #[test]
    fn a_value_change_creates_three_segments() {
        let bucket_ids = vec![1, 1, 2, 2, 2, 3, 1];
        let ids = value_change_segment_ids(&bucket_ids);
        assert_eq!(ids, vec![0, 0, 1, 1, 1, 2, 3]);
    }

    #[test]
    fn constant_values_never_split() {
        let bucket_ids = vec![5, 5, 5, 5];
        let ids = value_change_segment_ids(&bucket_ids);
        assert_eq!(ids, vec![0, 0, 0, 0]);
    }
}
