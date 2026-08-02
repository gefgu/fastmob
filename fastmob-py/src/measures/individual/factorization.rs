use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use arrow_array::types::*;
use arrow_array::{Array, BooleanArray, LargeStringArray, PrimitiveArray, StringArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use rustc_hash::FxHashMap;

use crate::utils::{extract_arrow_array, u64_results_into_arrow};

#[derive(Clone, Copy, Debug)]
struct TotalF64(u64);

impl TotalF64 {
    fn new(value: f64) -> Self {
        if value == 0.0 {
            Self(0)
        } else if value.is_nan() {
            Self(f64::NAN.to_bits())
        } else {
            Self(value.to_bits())
        }
    }

    fn value(self) -> f64 {
        f64::from_bits(self.0)
    }
}

impl PartialEq for TotalF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for TotalF64 {}
impl Hash for TotalF64 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}
impl PartialOrd for TotalF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for TotalF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value().total_cmp(&other.value())
    }
}

fn factorize_values<K, F>(len: usize, sort: bool, value_at: F) -> (Vec<u64>, Vec<u64>)
where
    K: Clone + Eq + Hash + Ord,
    F: Fn(usize) -> Option<K>,
{
    // A bounded initial allocation removes repeated growth for common medium-
    // cardinality inputs without reserving the full row count for a low-cardinality column.
    let mut seen: FxHashMap<Option<K>, u64> =
        FxHashMap::with_capacity_and_hasher(len.min(1 << 20), Default::default());
    let mut codes = Vec::with_capacity(len);
    let mut representatives = Vec::new();
    for index in 0..len {
        let value = value_at(index);
        let code = match seen.get(&value) {
            Some(&code) => code,
            None => {
                let code = representatives.len() as u64;
                seen.insert(value, code);
                representatives.push(index as u64);
                code
            }
        };
        codes.push(code);
    }
    if sort {
        sort_factorized(&mut codes, &mut representatives, &value_at);
    }
    (codes, representatives)
}

/// Convert first-seen factorization output to value-sorted codes without a second hash pass.
fn sort_factorized<K, F>(codes: &mut [u64], representatives: &mut Vec<u64>, value_at: &F)
where
    K: Ord,
    F: Fn(usize) -> Option<K>,
{
    let mut order: Vec<usize> = (0..representatives.len()).collect();
    order.sort_unstable_by(|&left, &right| {
        match (
            value_at(representatives[left] as usize),
            value_at(representatives[right] as usize),
        ) {
            (Some(left), Some(right)) => left.cmp(&right),
            (Some(_), None) => Ordering::Less,
            (None, Some(_)) => Ordering::Greater,
            (None, None) => Ordering::Equal,
        }
    });

    let mut ranks = vec![0u64; order.len()];
    for (new_code, &old_code) in order.iter().enumerate() {
        ranks[old_code] = new_code as u64;
    }
    for code in codes {
        *code = ranks[*code as usize];
    }
    *representatives = order
        .into_iter()
        .map(|old_code| representatives[old_code])
        .collect();
}

fn factorize_strings<'a, F>(len: usize, sort: bool, value_at: F) -> (Vec<u64>, Vec<u64>)
where
    F: Fn(usize) -> Option<&'a str>,
{
    if sort {
        return factorize_values(len, true, value_at);
    }

    let mut seen: FxHashMap<&str, u64> =
        FxHashMap::with_capacity_and_hasher(len.min(1 << 20), Default::default());
    let mut null_code = None;
    let mut codes = Vec::with_capacity(len);
    let mut representatives = Vec::new();
    for index in 0..len {
        let value = value_at(index);
        let code = if let Some(value) = value {
            match seen.get(value) {
                Some(&code) => code,
                None => {
                    let code = seen.len() as u64 + u64::from(null_code.is_some());
                    seen.insert(value, code);
                    representatives.push(index as u64);
                    code
                }
            }
        } else if let Some(code) = null_code {
            code
        } else {
            let code = seen.len() as u64;
            null_code = Some(code);
            representatives.push(index as u64);
            code
        };
        codes.push(code);
    }
    (codes, representatives)
}

fn factorize_dense_integers<T>(
    array: &PrimitiveArray<T>,
    sort: bool,
) -> Option<(Vec<u64>, Vec<u64>)>
where
    T: ArrowPrimitiveType,
    T::Native: Into<i128>,
{
    if array.is_empty() {
        return Some((Vec::new(), Vec::new()));
    }
    let mut range: Option<(i128, i128)> = None;
    for index in 0..array.len() {
        if array.is_valid(index) {
            let value = array.value(index).into();
            range = Some(match range {
                Some((min, max)) => (min.min(value), max.max(value)),
                None => (value, value),
            });
        }
    }
    let Some((min, max)) = range else {
        return Some((vec![0; array.len()], vec![0]));
    };
    let span = usize::try_from(max - min + 1).ok()?;
    if span > array.len().saturating_mul(4).min(1_000_000) {
        return None;
    }

    let mut codes_by_value = vec![u64::MAX; span];
    let mut representatives = Vec::new();
    if sort {
        let mut representative_by_value = vec![usize::MAX; span];
        let mut null_representative = None;
        for index in 0..array.len() {
            if array.is_valid(index) {
                let offset = (array.value(index).into() - min) as usize;
                if representative_by_value[offset] == usize::MAX {
                    representative_by_value[offset] = index;
                }
            } else {
                null_representative.get_or_insert(index);
            }
        }
        for (offset, &representative) in representative_by_value.iter().enumerate() {
            if representative != usize::MAX {
                codes_by_value[offset] = representatives.len() as u64;
                representatives.push(representative as u64);
            }
        }
        let null_code = null_representative.map(|representative| {
            let code = representatives.len() as u64;
            representatives.push(representative as u64);
            code
        });
        let codes = (0..array.len())
            .map(|index| {
                if array.is_valid(index) {
                    codes_by_value[(array.value(index).into() - min) as usize]
                } else {
                    null_code.expect("a null row has a null code")
                }
            })
            .collect();
        return Some((codes, representatives));
    } else {
        let mut null_code = None;
        let mut codes = Vec::with_capacity(array.len());
        for index in 0..array.len() {
            let code = if array.is_valid(index) {
                let offset = (array.value(index).into() - min) as usize;
                if codes_by_value[offset] == u64::MAX {
                    codes_by_value[offset] = representatives.len() as u64;
                    representatives.push(index as u64);
                }
                codes_by_value[offset]
            } else if let Some(code) = null_code {
                code
            } else {
                let code = representatives.len() as u64;
                null_code = Some(code);
                representatives.push(index as u64);
                code
            };
            codes.push(code);
        }
        return Some((codes, representatives));
    }
}

fn factorize_boolean(array: &BooleanArray, sort: bool) -> (Vec<u64>, Vec<u64>) {
    let state_at = |index: usize| match array.is_valid(index) {
        false => 2usize,
        true if array.value(index) => 1,
        true => 0,
    };
    let mut representatives_by_state = [usize::MAX; 3];
    for index in 0..array.len() {
        let state = state_at(index);
        if representatives_by_state[state] == usize::MAX {
            representatives_by_state[state] = index;
        }
    }
    let order: &[usize] = if sort { &[0, 1, 2] } else { &[] };
    let mut codes_by_state = [u64::MAX; 3];
    let mut representatives = Vec::new();
    if sort {
        for &state in order {
            if representatives_by_state[state] != usize::MAX {
                codes_by_state[state] = representatives.len() as u64;
                representatives.push(representatives_by_state[state] as u64);
            }
        }
    } else {
        for index in 0..array.len() {
            let state = state_at(index);
            if codes_by_state[state] == u64::MAX {
                codes_by_state[state] = representatives.len() as u64;
                representatives.push(index as u64);
            }
        }
    }
    let codes = (0..array.len())
        .map(|index| codes_by_state[state_at(index)])
        .collect();
    (codes, representatives)
}

macro_rules! factorize_primitive {
    ($array:expr, $type:ty, $sort:expr) => {{
        let array = $array
            .as_any()
            .downcast_ref::<PrimitiveArray<$type>>()
            .expect("Arrow data type and primitive array must agree");
        factorize_values(array.len(), $sort, |index| {
            array.is_valid(index).then(|| array.value(index))
        })
    }};
}

macro_rules! factorize_integer {
    ($array:expr, $type:ty, $sort:expr) => {{
        let array = $array
            .as_any()
            .downcast_ref::<PrimitiveArray<$type>>()
            .expect("Arrow data type and primitive array must agree");
        factorize_dense_integers(array, $sort).unwrap_or_else(|| {
            factorize_values(array.len(), $sort, |index| {
                array.is_valid(index).then(|| array.value(index))
            })
        })
    }};
}

#[pyfunction]
#[pyo3(signature = (values, sort = false))]
pub fn factorize_arrow(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    sort: bool,
) -> PyResult<(PyArray, PyArray)> {
    let values = extract_arrow_array(values, "values")?;
    let (array, _field) = values.into_inner();
    use arrow_schema::DataType;

    let result = py
        .detach(|| -> Result<_, String> {
            let result = match array.data_type() {
                DataType::Int8 => factorize_integer!(&array, Int8Type, sort),
                DataType::Int16 => factorize_integer!(&array, Int16Type, sort),
                DataType::Int32 => factorize_integer!(&array, Int32Type, sort),
                DataType::Int64 => factorize_integer!(&array, Int64Type, sort),
                DataType::UInt8 => factorize_integer!(&array, UInt8Type, sort),
                DataType::UInt16 => factorize_integer!(&array, UInt16Type, sort),
                DataType::UInt32 => factorize_integer!(&array, UInt32Type, sort),
                DataType::UInt64 => factorize_integer!(&array, UInt64Type, sort),
                DataType::Date32 => factorize_primitive!(&array, Date32Type, sort),
                DataType::Date64 => factorize_primitive!(&array, Date64Type, sort),
                DataType::Timestamp(unit, _) => match unit {
                    arrow_schema::TimeUnit::Second => {
                        factorize_primitive!(&array, TimestampSecondType, sort)
                    }
                    arrow_schema::TimeUnit::Millisecond => {
                        factorize_primitive!(&array, TimestampMillisecondType, sort)
                    }
                    arrow_schema::TimeUnit::Microsecond => {
                        factorize_primitive!(&array, TimestampMicrosecondType, sort)
                    }
                    arrow_schema::TimeUnit::Nanosecond => {
                        factorize_primitive!(&array, TimestampNanosecondType, sort)
                    }
                },
                DataType::Float32 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<PrimitiveArray<Float32Type>>()
                        .unwrap();
                    factorize_values(array.len(), sort, |index| {
                        array
                            .is_valid(index)
                            .then(|| TotalF64::new(array.value(index) as f64))
                    })
                }
                DataType::Float64 => {
                    let array = array
                        .as_any()
                        .downcast_ref::<PrimitiveArray<Float64Type>>()
                        .unwrap();
                    factorize_values(array.len(), sort, |index| {
                        array
                            .is_valid(index)
                            .then(|| TotalF64::new(array.value(index)))
                    })
                }
                DataType::Boolean => {
                    let array = array.as_any().downcast_ref::<BooleanArray>().unwrap();
                    factorize_boolean(array, sort)
                }
                DataType::Utf8 => {
                    let array = array.as_any().downcast_ref::<StringArray>().unwrap();
                    factorize_strings(array.len(), sort, |index| {
                        array.is_valid(index).then(|| array.value(index))
                    })
                }
                DataType::LargeUtf8 => {
                    let array = array.as_any().downcast_ref::<LargeStringArray>().unwrap();
                    factorize_strings(array.len(), sort, |index| {
                        array.is_valid(index).then(|| array.value(index))
                    })
                }
                data_type => {
                    return Err(format!(
                        "unsupported Arrow dtype for factorization: {data_type}"
                    ));
                }
            };
            Ok(result)
        })
        .map_err(PyValueError::new_err)?;
    Ok((
        u64_results_into_arrow(result.0),
        u64_results_into_arrow(result.1),
    ))
}
