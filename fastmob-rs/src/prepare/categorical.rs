//! Factorizing a string-ish column into dense codes plus a category list.
//!
//! Categories are sorted lexicographically, matching Python's
//! `sorted(set(values), key=str)`, so a matrix indexed by code has the same
//! row/column order on both sides of the FFI boundary.

use polars::prelude::*;

use crate::error::FastmobRsError;

/// Dense `u64` codes alongside the category each code indexes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Categorical {
    pub categories: Vec<String>,
    pub codes: Vec<u64>,
}

/// The label given to null entries, matching citybehavex's prior behaviour and
/// keeping every row codeable rather than dropping it.
const NULL_LABEL: &str = "UNKNOWN";

/// Factorize a column into lexicographically ordered categories and codes.
pub fn factorize(series: &Series) -> Result<Categorical, FastmobRsError> {
    let cast = series.cast(&DataType::String)?;
    let values = cast.str()?;

    let mut categories: Vec<&str> = Vec::new();
    let mut seen: rustc_hash::FxHashSet<&str> = rustc_hash::FxHashSet::default();
    for value in values.into_iter() {
        let value = value.unwrap_or(NULL_LABEL);
        if seen.insert(value) {
            categories.push(value);
        }
    }
    categories.sort_unstable();

    let ranks: rustc_hash::FxHashMap<&str, u64> = categories
        .iter()
        .enumerate()
        .map(|(rank, &value)| (value, rank as u64))
        .collect();
    let codes = values
        .into_iter()
        .map(|value| ranks[value.unwrap_or(NULL_LABEL)])
        .collect();

    Ok(Categorical {
        categories: categories.into_iter().map(str::to_string).collect(),
        codes,
    })
}

/// Contiguous per-group end offsets for a column already grouped by value.
///
/// Unlike [`crate::prepare`], this never reorders anything: it reads the runs
/// that are already there. Use it for frames a caller has arranged themselves
/// in an order the measure must preserve.
pub fn presorted_group_ends(series: &Series) -> Result<Vec<usize>, FastmobRsError> {
    let codes = super::uid_codes::build(series, series.len())?;
    Ok(super::arrange::ends_from_arranged_codes(&codes.codes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn categories_are_lexicographic_and_codes_index_them() {
        let series = Series::new("a".into(), ["work", "home", "work", "errand"]);
        let result = factorize(&series).unwrap();
        assert_eq!(result.categories, vec!["errand", "home", "work"]);
        assert_eq!(result.codes, vec![2, 1, 2, 0]);
    }

    #[test]
    fn nulls_become_an_unknown_category_rather_than_being_dropped() {
        let series = Series::new("a".into(), [Some("work"), None, Some("home")]);
        let result = factorize(&series).unwrap();
        assert_eq!(result.categories, vec!["UNKNOWN", "home", "work"]);
        assert_eq!(result.codes, vec![2, 0, 1]);
    }

    #[test]
    fn non_string_columns_are_cast_first() {
        let series = Series::new("a".into(), [10i64, 2, 10]);
        let result = factorize(&series).unwrap();
        // Lexicographic on the string form: "10" sorts before "2".
        assert_eq!(result.categories, vec!["10", "2"]);
        assert_eq!(result.codes, vec![0, 1, 0]);
    }

    #[test]
    fn empty_input_has_no_categories() {
        let series = Series::new("a".into(), Vec::<&str>::new());
        let result = factorize(&series).unwrap();
        assert!(result.categories.is_empty());
        assert!(result.codes.is_empty());
    }

    #[test]
    fn group_ends_follow_the_existing_runs() {
        let series = Series::new("uid".into(), [5i64, 5, 1, 1, 1, 9]);
        assert_eq!(presorted_group_ends(&series).unwrap(), vec![2, 5, 6]);
    }

    #[test]
    fn group_ends_do_not_reorder_out_of_order_input() {
        // "b" appears twice in two separate runs; both are kept as-is.
        let series = Series::new("uid".into(), ["b", "a", "b"]);
        assert_eq!(presorted_group_ends(&series).unwrap(), vec![1, 2, 3]);
    }

    #[test]
    fn group_ends_of_an_empty_column_are_empty() {
        let series = Series::new("uid".into(), Vec::<i64>::new());
        assert!(presorted_group_ends(&series).unwrap().is_empty());
    }
}
