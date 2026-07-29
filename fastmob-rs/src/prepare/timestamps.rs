//! Datetime coercion and extraction of a plain `i64` ordering key.

use polars::prelude::*;

use crate::error::FastmobRsError;

/// Coerce a datetime-ish column to `Datetime`, turning unparsable values into
/// nulls rather than erroring (they are dropped by the cleaning step).
///
/// Mirrors the Python `_to_datetime` helper, and matches the behaviour
/// citybehavex's `comparison::util::to_datetime_expr` relies on.
pub(crate) fn datetime_expr(schema: &Schema, name: &str) -> Expr {
    match schema.get(name) {
        Some(DataType::String) => col(name).str().to_datetime(
            Some(TimeUnit::Microseconds),
            None,
            StrptimeOptions {
                format: None,
                strict: false,
                exact: true,
                cache: true,
            },
            lit("raise"),
        ),
        Some(DataType::Datetime(_, _)) | Some(DataType::Date) => col(name),
        Some(dtype) if dtype.is_integer() => col(name),
        _ => col(name).cast(DataType::Datetime(TimeUnit::Microseconds, None)),
    }
}

/// Extract the physical `i64` ordering key from a datetime column.
///
/// Only ordering matters here, so the physical representation is used as-is
/// without normalising the time unit — every row in a column shares one unit.
pub(crate) fn ordering_key(series: &Series, name: &str) -> Result<Vec<i64>, FastmobRsError> {
    match series.dtype() {
        DataType::Datetime(_, _) => {
            let values = series.datetime()?;
            Ok(values.phys.into_no_null_iter().collect())
        }
        DataType::Date => {
            let values = series.date()?;
            Ok(values.phys.into_no_null_iter().map(i64::from).collect())
        }
        dtype if dtype.is_integer() => {
            let cast = series.cast(&DataType::Int64)?;
            Ok(cast.i64()?.into_no_null_iter().collect())
        }
        dtype => Err(FastmobRsError::UnsupportedDtype(
            name.to_string(),
            dtype.clone(),
        )),
    }
}

/// Millisecond Unix timestamps, matching the Python `_extract_timestamps_ms`
/// contract that time-based measures are written against.
pub(crate) fn milliseconds(series: &Series, name: &str) -> Result<Vec<i64>, FastmobRsError> {
    match series.dtype() {
        DataType::Datetime(unit, _) => {
            let values = series.datetime()?;
            let divisor = match unit {
                TimeUnit::Nanoseconds => 1_000_000,
                TimeUnit::Microseconds => 1_000,
                TimeUnit::Milliseconds => 1,
            };
            Ok(values
                .phys
                .into_no_null_iter()
                .map(|value| value.div_euclid(divisor))
                .collect())
        }
        DataType::Date => {
            let values = series.date()?;
            Ok(values
                .phys
                .into_no_null_iter()
                .map(|value| i64::from(value) * 86_400_000)
                .collect())
        }
        _ => ordering_key(series, name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_columns_are_used_directly() {
        let series = Series::new("datetime".into(), [3i64, 1, 2]);
        assert_eq!(ordering_key(&series, "datetime").unwrap(), vec![3, 1, 2]);
    }

    #[test]
    fn datetime_columns_use_their_physical_representation() {
        let series = Series::new("datetime".into(), [3i64, 1, 2])
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        assert_eq!(ordering_key(&series, "datetime").unwrap(), vec![3, 1, 2]);
    }

    #[test]
    fn milliseconds_rescale_from_the_column_unit() {
        let series = Series::new("datetime".into(), [3_000i64, 1_000])
            .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
            .unwrap();
        assert_eq!(milliseconds(&series, "datetime").unwrap(), vec![3, 1]);
    }

    #[test]
    fn unsupported_dtypes_are_rejected() {
        let series = Series::new("datetime".into(), [1.5f64]);
        assert!(matches!(
            ordering_key(&series, "datetime"),
            Err(FastmobRsError::UnsupportedDtype(_, _))
        ));
    }
}
