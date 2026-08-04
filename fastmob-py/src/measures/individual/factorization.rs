use std::cmp::Ordering;
use std::hash::{Hash, Hasher};

use arrow_array::types::*;
use arrow_array::{
    Array, BooleanArray, DictionaryArray, LargeStringArray, PrimitiveArray, StringArray,
};
use arrow_buffer::ArrowNativeType;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3_arrow::PyArray;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

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

const CARDINALITY_SAMPLE: usize = 4096;
const MAX_INITIAL_CAPACITY: usize = 1 << 22;

/// Choose a capacity that avoids both a repeated growth cascade for high-cardinality
/// columns and a huge, mostly-empty allocation for categoricals.
fn estimate_cardinality<K, F>(len: usize, value_at: &F) -> usize
where
    K: Clone + Eq + Hash,
    F: Fn(usize) -> Option<K>,
{
    if len <= CARDINALITY_SAMPLE {
        return len;
    }
    let stride = len / CARDINALITY_SAMPLE;
    let mut sample = FxHashSet::with_capacity_and_hasher(CARDINALITY_SAMPLE, Default::default());
    for sample_index in 0..CARDINALITY_SAMPLE {
        sample.insert(value_at(sample_index * stride));
    }
    let distinct = sample.len();
    if distinct < CARDINALITY_SAMPLE / 4 {
        distinct.saturating_mul(4).max(1)
    } else {
        (len / 4).clamp(CARDINALITY_SAMPLE, MAX_INITIAL_CAPACITY)
    }
}

fn factorize_values<K, F>(len: usize, sort: bool, value_at: F) -> (Vec<u64>, Vec<u64>)
where
    K: Clone + Eq + Hash + Ord,
    F: Fn(usize) -> Option<K>,
{
    let mut seen: FxHashMap<Option<K>, u64> = FxHashMap::with_capacity_and_hasher(
        estimate_cardinality(len, &value_at),
        Default::default(),
    );
    let mut codes = Vec::with_capacity(len);
    let mut representatives = Vec::new();
    for index in 0..len {
        let value = value_at(index);
        let code = match seen.entry(value) {
            std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let code = representatives.len() as u64;
                entry.insert(code);
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
    let keys: Vec<Option<K>> = representatives
        .iter()
        .map(|&representative| value_at(representative as usize))
        .collect();
    let mut order: Vec<usize> = (0..representatives.len()).collect();
    order.sort_unstable_by(|&left, &right| match (&keys[left], &keys[right]) {
        (Some(left), Some(right)) => left.cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    });

    let mut ranks = vec![0u64; order.len()];
    for (new_code, &old_code) in order.iter().enumerate() {
        ranks[old_code] = new_code as u64;
    }
    codes
        .par_iter_mut()
        .for_each(|code| *code = ranks[*code as usize]);
    *representatives = order
        .into_iter()
        .map(|old_code| representatives[old_code])
        .collect();
}

fn factorize_strings<'a, F>(len: usize, sort: bool, value_at: F) -> (Vec<u64>, Vec<u64>)
where
    F: Fn(usize) -> Option<&'a str>,
{
    let mut seen: FxHashMap<&str, u64> = FxHashMap::with_capacity_and_hasher(
        estimate_cardinality(len, &value_at),
        Default::default(),
    );
    let mut null_code = None;
    let mut codes = Vec::with_capacity(len);
    let mut representatives = Vec::new();
    for index in 0..len {
        let value = value_at(index);
        let code = if let Some(value) = value {
            match seen.entry(value) {
                std::collections::hash_map::Entry::Occupied(entry) => *entry.get(),
                std::collections::hash_map::Entry::Vacant(entry) => {
                    let code = representatives.len() as u64;
                    entry.insert(code);
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
    if sort {
        sort_factorized(&mut codes, &mut representatives, &value_at);
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
    let range = if array.null_count() == 0 {
        let mut values = array.values().iter().map(|&value| value.into());
        let first = values.next().expect("non-empty array has a first value");
        Some(values.fold((first, first), |(min, max), value| {
            (min.min(value), max.max(value))
        }))
    } else {
        let mut values = (0..array.len())
            .filter(|&index| array.is_valid(index))
            .map(|index| array.value(index).into());
        values.next().map(|first| {
            values.fold((first, first), |(min, max), value| {
                (min.min(value), max.max(value))
            })
        })
    };
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
    let mut codes_by_state = [u64::MAX; 3];
    let mut representatives = Vec::new();
    if sort {
        let mut representatives_by_state = [usize::MAX; 3];
        for index in 0..array.len() {
            let state = state_at(index);
            if representatives_by_state[state] == usize::MAX {
                representatives_by_state[state] = index;
            }
        }
        for state in [0, 1, 2] {
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

/// Factorize a string dictionary through its key array.  The dictionary values are
/// factorized only once (including duplicate values), then the row scan is made of
/// small integer gathers rather than string hashes.
fn factorize_string_dictionary<K>(array: &DictionaryArray<K>, sort: bool) -> (Vec<u64>, Vec<u64>)
where
    K: ArrowDictionaryKeyType,
    K::Native: ArrowNativeType,
{
    let values = array
        .values()
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("dictionary dispatch and values must agree");
    let (dictionary_codes, dictionary_representatives) =
        factorize_strings(values.len(), sort, |index| {
            values.is_valid(index).then(|| values.value(index))
        });
    let dictionary_null_code = (0..values.len())
        .find(|&index| !values.is_valid(index))
        .map(|index| dictionary_codes[index] as usize);
    let virtual_null_code = dictionary_representatives.len();
    let null_code = dictionary_null_code.unwrap_or(virtual_null_code);
    let logical_code_count =
        dictionary_representatives.len() + usize::from(dictionary_null_code.is_none());
    let mut logical_codes = Vec::with_capacity(array.len());
    for index in 0..array.len() {
        let code = if array.is_valid(index) {
            dictionary_codes[array.keys().value(index).as_usize()] as usize
        } else {
            null_code
        };
        logical_codes.push(code);
    }

    let mut representatives_by_value = vec![usize::MAX; logical_code_count];
    for (index, &code) in logical_codes.iter().enumerate() {
        if representatives_by_value[code] == usize::MAX {
            representatives_by_value[code] = index;
        }
    }
    let mut output_code_by_value = vec![u64::MAX; logical_code_count];
    let mut representatives = Vec::new();
    if sort {
        for code in 0..dictionary_representatives.len() {
            if code == null_code {
                continue;
            }
            if representatives_by_value[code] != usize::MAX {
                output_code_by_value[code] = representatives.len() as u64;
                representatives.push(representatives_by_value[code] as u64);
            }
        }
        if representatives_by_value[null_code] != usize::MAX {
            output_code_by_value[null_code] = representatives.len() as u64;
            representatives.push(representatives_by_value[null_code] as u64);
        }
    } else {
        for (index, &code) in logical_codes.iter().enumerate() {
            if output_code_by_value[code] == u64::MAX {
                output_code_by_value[code] = representatives.len() as u64;
                representatives.push(index as u64);
            }
        }
    }
    let codes = logical_codes
        .into_iter()
        .map(|code| output_code_by_value[code])
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

macro_rules! factorize_string_dictionary {
    ($array:expr, $key_type:ty, $sort:expr) => {{
        let array = $array
            .as_any()
            .downcast_ref::<DictionaryArray<$key_type>>()
            .expect("Arrow data type and dictionary array must agree");
        factorize_string_dictionary(array, $sort)
    }};
}

/// Factorize a supported Arrow array without crossing the Python boundary.
///
/// This is shared by Arrow-native kernels that need categorical codes but must
/// keep dataframe adapters out of their hot path.
pub(crate) fn factorize_array(
    array: &dyn Array,
    sort: bool,
) -> Result<(Vec<u64>, Vec<u64>), String> {
    use arrow_schema::DataType;

    let result = match array.data_type() {
        DataType::Int8 => factorize_integer!(array, Int8Type, sort),
        DataType::Int16 => factorize_integer!(array, Int16Type, sort),
        DataType::Int32 => factorize_integer!(array, Int32Type, sort),
        DataType::Int64 => factorize_integer!(array, Int64Type, sort),
        DataType::UInt8 => factorize_integer!(array, UInt8Type, sort),
        DataType::UInt16 => factorize_integer!(array, UInt16Type, sort),
        DataType::UInt32 => factorize_integer!(array, UInt32Type, sort),
        DataType::UInt64 => factorize_integer!(array, UInt64Type, sort),
        DataType::Date32 => factorize_primitive!(array, Date32Type, sort),
        DataType::Date64 => factorize_primitive!(array, Date64Type, sort),
        DataType::Timestamp(unit, _) => match unit {
            arrow_schema::TimeUnit::Second => {
                factorize_primitive!(array, TimestampSecondType, sort)
            }
            arrow_schema::TimeUnit::Millisecond => {
                factorize_primitive!(array, TimestampMillisecondType, sort)
            }
            arrow_schema::TimeUnit::Microsecond => {
                factorize_primitive!(array, TimestampMicrosecondType, sort)
            }
            arrow_schema::TimeUnit::Nanosecond => {
                factorize_primitive!(array, TimestampNanosecondType, sort)
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
        DataType::Dictionary(key_type, value_type)
            if matches!(value_type.as_ref(), DataType::Utf8) =>
        {
            match key_type.as_ref() {
                DataType::Int8 => factorize_string_dictionary!(array, Int8Type, sort),
                DataType::Int16 => factorize_string_dictionary!(array, Int16Type, sort),
                DataType::Int32 => factorize_string_dictionary!(array, Int32Type, sort),
                DataType::Int64 => factorize_string_dictionary!(array, Int64Type, sort),
                DataType::UInt8 => factorize_string_dictionary!(array, UInt8Type, sort),
                DataType::UInt16 => factorize_string_dictionary!(array, UInt16Type, sort),
                DataType::UInt32 => factorize_string_dictionary!(array, UInt32Type, sort),
                DataType::UInt64 => factorize_string_dictionary!(array, UInt64Type, sort),
                key_type => {
                    return Err(format!(
                        "unsupported dictionary key dtype for factorization: {key_type}"
                    ));
                }
            }
        }
        data_type => {
            return Err(format!(
                "unsupported Arrow dtype for factorization: {data_type}"
            ));
        }
    };
    Ok(result)
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
    let result = py
        .detach(|| factorize_array(array.as_ref(), sort))
        .map_err(PyValueError::new_err)?;
    Ok((
        u64_results_into_arrow(result.0),
        u64_results_into_arrow(result.1),
    ))
}
