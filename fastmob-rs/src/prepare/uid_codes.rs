//! Value-ordered `u32` user codes.
//!
//! Every downstream step (arranging, group boundaries, output labels) works on
//! these codes rather than the original uid column, which buys two things:
//!
//! 1. **Speed.** The arrange step compares `u32`s instead of strings or
//!    multi-column row encodings, and the counting sort in [`super::arrange`]
//!    needs a bounded integer key.
//! 2. **Correctness on composite string uids.** A naive `cast(Int64)` on ids
//!    like `"10_2980"` silently nulls every value; a follow-up
//!    `unwrap_or(i64::MIN)` then collapses every user into one group. That was
//!    a real, diagnosed bug in citybehavex's port (504 users collapsed to 1,
//!    creating 503 spurious inter-user jumps), so no path here ever parses a
//!    uid as a number.
//!
//! Codes are **value-ordered**: `code(a) < code(b)` iff `a < b` under the
//! column's natural ordering. That makes the arranged group order a pure
//! function of the data rather than of incoming row order, so results are
//! reproducible and stable under [`super::trajectory::PreparedTrajectory::filter`].

use polars::prelude::*;
use rustc_hash::FxHashMap;

use crate::error::FastmobRsError;

/// Value-ordered codes plus the number of distinct codes, i.e. the exclusive
/// upper bound the counting sort needs.
pub(crate) struct UidCodes {
    pub codes: Vec<u32>,
    pub num_codes: usize,
}

/// Direct offset coding stays cheap as long as the key space is not wildly
/// larger than the row count; past that the counting-sort histogram would
/// dominate, so fall back to dense ranking.
fn offset_coding_is_worthwhile(span: u128, len: usize) -> bool {
    span <= u32::MAX as u128 && span <= (4 * len as u128).max(1024)
}

pub(crate) fn build(series: &Series, len: usize) -> Result<UidCodes, FastmobRsError> {
    if len == 0 {
        return Ok(UidCodes {
            codes: Vec::new(),
            num_codes: 0,
        });
    }

    if series.dtype().is_integer()
        && let Some(codes) = integer_codes(series, len)?
    {
        return Ok(codes);
    }

    string_codes(series, len)
}

/// Fast path: shift integer uids down by their minimum. Order-preserving by
/// construction and needs no hashing at all, which matters because this runs
/// once per row.
fn integer_codes(series: &Series, len: usize) -> Result<Option<UidCodes>, FastmobRsError> {
    let cast = series.cast(&DataType::Int64)?;
    let values = cast.i64()?;
    if values.null_count() != 0 {
        return Ok(None);
    }

    let (Some(min), Some(max)) = (values.min(), values.max()) else {
        return Ok(None);
    };
    let span = (max as i128 - min as i128) as u128;
    if !offset_coding_is_worthwhile(span, len) {
        return dense_rank_integer_codes(values);
    }

    let mut codes = Vec::with_capacity(len);
    for chunk in values.downcast_iter() {
        codes.extend(chunk.values().iter().map(|&v| (v - min) as u32));
    }
    Ok(Some(UidCodes {
        codes,
        num_codes: span as usize + 1,
    }))
}

/// Sparse integer uids: rank the (few) distinct values, then remap. Keeps the
/// counting-sort histogram proportional to the user count rather than to the
/// numeric range.
fn dense_rank_integer_codes(values: &Int64Chunked) -> Result<Option<UidCodes>, FastmobRsError> {
    let mut uniques: Vec<i64> = values.into_no_null_iter().collect();
    uniques.sort_unstable();
    uniques.dedup();

    let ranks: FxHashMap<i64, u32> = uniques
        .iter()
        .enumerate()
        .map(|(rank, &value)| (value, rank as u32))
        .collect();

    let codes = values
        .into_no_null_iter()
        .map(|value| ranks[&value])
        .collect::<Vec<_>>();
    Ok(Some(UidCodes {
        codes,
        num_codes: uniques.len(),
    }))
}

/// Everything else (strings, and integers with nulls) goes through the string
/// representation, which is the only encoding guaranteed to exist for an
/// arbitrary uid dtype. Nulls sort last, matching Polars' `nulls_last` default
/// for sorted uid columns.
fn string_codes(series: &Series, len: usize) -> Result<UidCodes, FastmobRsError> {
    let cast = series.cast(&DataType::String)?;
    let values = cast.str()?;

    let mut uniques: Vec<Option<&str>> = Vec::new();
    let mut seen: FxHashMap<Option<&str>, ()> = FxHashMap::default();
    for value in values.into_iter() {
        if seen.insert(value, ()).is_none() {
            uniques.push(value);
        }
    }
    uniques.sort_unstable_by(|a, b| match (a, b) {
        (Some(a), Some(b)) => a.cmp(b),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    });

    let ranks: FxHashMap<Option<&str>, u32> = uniques
        .iter()
        .enumerate()
        .map(|(rank, &value)| (value, rank as u32))
        .collect();

    let mut codes = Vec::with_capacity(len);
    for value in values.into_iter() {
        codes.push(ranks[&value]);
    }
    Ok(UidCodes {
        codes,
        num_codes: uniques.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codes_of(series: Series) -> Vec<u32> {
        let len = series.len();
        build(&series, len).unwrap().codes
    }

    #[test]
    fn integer_uids_are_offset_coded_in_value_order() {
        let codes = codes_of(Series::new("uid".into(), [10i64, 12, 10, 11]));
        assert_eq!(codes, vec![0, 2, 0, 1]);
    }

    #[test]
    fn sparse_integer_uids_fall_back_to_dense_ranks() {
        let codes = codes_of(Series::new(
            "uid".into(),
            [5_000_000_000i64, 1, 5_000_000_000, 7],
        ));
        assert_eq!(codes, vec![2, 0, 2, 1]);
    }

    #[test]
    fn composite_string_uids_keep_distinct_codes() {
        // The exact shape that collapsed 504 users into one in citybehavex.
        let codes = codes_of(Series::new(
            "uid".into(),
            ["10_2980", "10_2981", "10_2980", "9_1"],
        ));
        // Lexicographic order: "10_2980" < "10_2981" < "9_1".
        assert_eq!(codes, vec![0, 1, 0, 2]);
        assert_eq!(codes.iter().collect::<rustc_hash::FxHashSet<_>>().len(), 3);
    }

    #[test]
    fn num_codes_bounds_every_code() {
        let series = Series::new("uid".into(), ["b", "a", "c", "a"]);
        let built = build(&series, series.len()).unwrap();
        assert!(built.codes.iter().all(|&c| (c as usize) < built.num_codes));
        assert_eq!(built.num_codes, 3);
    }

    #[test]
    fn empty_input_produces_no_codes() {
        let series = Series::new("uid".into(), Vec::<i64>::new());
        let built = build(&series, 0).unwrap();
        assert!(built.codes.is_empty());
        assert_eq!(built.num_codes, 0);
    }
}
