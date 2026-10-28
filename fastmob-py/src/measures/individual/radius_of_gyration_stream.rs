//! PyO3 binding for the streamed radius of gyration
//! (`fastmob_core::measures::individual::radius_of_gyration_stream`).
//!
//! Each call takes one batch as Arrow C streams (`PyChunkedArray`), so a Polars
//! Series is consumed chunk by chunk with no copy -- in particular, its string
//! user IDs stay `string_view`, where exporting them as `large_string` would
//! copy every value. Rows are keyed straight off the Arrow buffers.

use std::sync::Arc;

use arrow_array::types::*;
use arrow_array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, Float64Array, Int64Array,
    LargeBinaryArray, LargeStringArray, PrimitiveArray, StringArray, StringViewArray, UInt64Array,
};
use arrow_schema::DataType;
use fastmob_core::measures::individual::radius_of_gyration_stream::{
    RogRowStream, RowChunk, UidKey, UidKeyRef,
};
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_arrow::{PyArray as ArrowPyArray, PyChunkedArray};

use super::factorization::TotalF64;
use crate::utils::f64_results_into_arrow;

type PyRogStreamResult<'py> = (Py<PyAny>, Py<PyAny>, Bound<'py, PyArray1<bool>>);

fn arrow_f64_output(py: Python<'_>, values: Vec<f64>) -> PyResult<Py<PyAny>> {
    Ok(Py::new(py, f64_results_into_arrow(values))?.into_any())
}

#[derive(Clone, Copy)]
enum Pass {
    Moments,
    Squares,
}

/// Slice every column at the union of all columns' chunk boundaries, so piece
/// `p` of each column covers the same rows. Columns of one Polars frame share
/// boundaries already, in which case this only clones `Arc`s.
fn align_chunks(columns: &[&[ArrayRef]]) -> PyResult<Vec<Vec<ArrayRef>>> {
    let lengths: Vec<usize> = columns
        .iter()
        .map(|chunks| chunks.iter().map(|c| c.len()).sum())
        .collect();
    if lengths.windows(2).any(|w| w[0] != w[1]) {
        return Err(PyValueError::new_err(
            "user-ID and coordinate columns must have the same length",
        ));
    }
    let mut cuts: Vec<usize> = columns
        .iter()
        .flat_map(|chunks| {
            chunks.iter().scan(0usize, |end, c| {
                *end += c.len();
                Some(*end)
            })
        })
        .collect();
    cuts.sort_unstable();
    cuts.dedup();

    let mut pieces = vec![Vec::with_capacity(cuts.len()); columns.len()];
    for (column, chunks) in columns.iter().enumerate() {
        let (mut chunk, mut chunk_start, mut start) = (0usize, 0usize, 0usize);
        for &end in &cuts {
            if end == start {
                continue;
            }
            while chunk_start + chunks[chunk].len() <= start {
                chunk_start += chunks[chunk].len();
                chunk += 1;
            }
            pieces[column].push(chunks[chunk].slice(start - chunk_start, end - start));
            start = end;
        }
    }
    Ok(pieces)
}

struct Coordinates {
    latitudes: Float64Array,
    longitudes: Float64Array,
    valid: Option<Vec<bool>>,
}

fn coordinates(lat: &ArrayRef, lng: &ArrayRef) -> PyResult<Coordinates> {
    let as_f64 = |array: &ArrayRef, name: &str| {
        array
            .as_any()
            .downcast_ref::<Float64Array>()
            .cloned()
            .ok_or_else(|| PyTypeError::new_err(format!("{name} must be a float64 Arrow array")))
    };
    let (latitudes, longitudes) = (as_f64(lat, "latitudes")?, as_f64(lng, "longitudes")?);
    let valid = (latitudes.null_count() + longitudes.null_count() > 0).then(|| {
        (0..latitudes.len())
            .map(|i| latitudes.is_valid(i) && longitudes.is_valid(i))
            .collect()
    });
    Ok(Coordinates {
        latitudes,
        longitudes,
        valid,
    })
}

fn run<'a, K>(
    stream: &mut RogRowStream,
    pass: Pass,
    chunks: &[RowChunk<'a, K>],
) -> Result<(), String>
where
    K: Fn(usize) -> UidKeyRef<'a> + Sync,
{
    match pass {
        Pass::Moments => stream.accumulate_moments(chunks),
        Pass::Squares => stream.accumulate_sq(chunks),
    }
}

/// Build one `RowChunk` per aligned piece with a key accessor for array type
/// `$array`, then run the pass. One closure expression per arm, so every chunk
/// of a batch shares one concrete key type.
macro_rules! run_keyed {
    ($stream:expr, $pass:expr, $uids:expr, $coords:expr, $array:ty, |$a:ident, $i:ident| $key:expr) => {{
        let arrays: Vec<&$array> = $uids
            .iter()
            .map(|u| u.as_any().downcast_ref::<$array>().unwrap())
            .collect();
        let chunks: Vec<_> = arrays
            .iter()
            .zip($coords)
            .map(|(&$a, c)| RowChunk {
                keys: move |$i: usize| {
                    if $a.is_null($i) {
                        UidKeyRef::Null
                    } else {
                        $key
                    }
                },
                latitudes: c.latitudes.values(),
                longitudes: c.longitudes.values(),
                valid: c.valid.as_deref(),
            })
            .collect();
        run($stream, $pass, &chunks)
    }};
}

macro_rules! run_integer {
    ($stream:expr, $pass:expr, $uids:expr, $coords:expr, $ty:ty) => {
        run_keyed!(
            $stream,
            $pass,
            $uids,
            $coords,
            PrimitiveArray<$ty>,
            |a, i| { UidKeyRef::Int(a.value(i).into()) }
        )
    };
}

macro_rules! run_bytes {
    ($stream:expr, $pass:expr, $uids:expr, $coords:expr, $ty:ty) => {
        run_keyed!($stream, $pass, $uids, $coords, $ty, |a, i| {
            UidKeyRef::Bytes(a.value(i).as_ref())
        })
    };
}

fn dispatch(
    stream: &mut RogRowStream,
    pass: Pass,
    data_type: &DataType,
    uids: &[ArrayRef],
    coords: &[Coordinates],
) -> PyResult<Result<(), String>> {
    use arrow_schema::TimeUnit;

    let result = match data_type {
        DataType::Int8 => run_integer!(stream, pass, uids, coords, Int8Type),
        DataType::Int16 => run_integer!(stream, pass, uids, coords, Int16Type),
        DataType::Int32 => run_integer!(stream, pass, uids, coords, Int32Type),
        DataType::Int64 => run_integer!(stream, pass, uids, coords, Int64Type),
        DataType::UInt8 => run_integer!(stream, pass, uids, coords, UInt8Type),
        DataType::UInt16 => run_integer!(stream, pass, uids, coords, UInt16Type),
        DataType::UInt32 => run_integer!(stream, pass, uids, coords, UInt32Type),
        DataType::UInt64 => run_integer!(stream, pass, uids, coords, UInt64Type),
        DataType::Date32 => run_integer!(stream, pass, uids, coords, Date32Type),
        DataType::Date64 => run_integer!(stream, pass, uids, coords, Date64Type),
        DataType::Timestamp(TimeUnit::Second, _) => {
            run_integer!(stream, pass, uids, coords, TimestampSecondType)
        }
        DataType::Timestamp(TimeUnit::Millisecond, _) => {
            run_integer!(stream, pass, uids, coords, TimestampMillisecondType)
        }
        DataType::Timestamp(TimeUnit::Microsecond, _) => {
            run_integer!(stream, pass, uids, coords, TimestampMicrosecondType)
        }
        DataType::Timestamp(TimeUnit::Nanosecond, _) => {
            run_integer!(stream, pass, uids, coords, TimestampNanosecondType)
        }
        DataType::Float32 => {
            run_keyed!(
                stream,
                pass,
                uids,
                coords,
                PrimitiveArray<Float32Type>,
                |a, i| UidKeyRef::Float(TotalF64::new(a.value(i).into()).0)
            )
        }
        DataType::Float64 => {
            run_keyed!(
                stream,
                pass,
                uids,
                coords,
                PrimitiveArray<Float64Type>,
                |a, i| UidKeyRef::Float(TotalF64::new(a.value(i)).0)
            )
        }
        DataType::Boolean => run_keyed!(stream, pass, uids, coords, BooleanArray, |a, i| {
            UidKeyRef::Bool(a.value(i))
        }),
        DataType::Utf8 => run_bytes!(stream, pass, uids, coords, StringArray),
        DataType::LargeUtf8 => run_bytes!(stream, pass, uids, coords, LargeStringArray),
        DataType::Utf8View => run_bytes!(stream, pass, uids, coords, StringViewArray),
        DataType::Binary => run_bytes!(stream, pass, uids, coords, BinaryArray),
        DataType::LargeBinary => run_bytes!(stream, pass, uids, coords, LargeBinaryArray),
        DataType::BinaryView => run_bytes!(stream, pass, uids, coords, BinaryViewArray),
        DataType::Null => {
            run_keyed!(
                stream,
                pass,
                uids,
                coords,
                arrow_array::NullArray,
                |_a, _i| UidKeyRef::Null
            )
        }
        other => {
            return Err(PyTypeError::new_err(format!(
                "unsupported user-ID Arrow type for radius_of_gyration_lazy: {other}"
            )));
        }
    };
    Ok(result)
}

/// Labels in slot order as an Arrow array of the canonical type for the
/// user-ID family (int64/uint64, float64, bool, large_string, large_binary);
/// the Python wrapper casts it back to the input's exact type.
fn labels_into_arrow(labels: &[UidKey], data_type: Option<&DataType>) -> ArrayRef {
    let get = |want: fn(&UidKey) -> bool| labels.iter().map(move |k| Some(k).filter(|k| want(k)));
    let is_int = |k: &UidKey| matches!(k, UidKey::Int(_));
    match data_type {
        Some(dt) if dt.is_integer() || dt.is_temporal() => {
            let fits_i64 = labels
                .iter()
                .all(|k| !matches!(k, UidKey::Int(v) if i64::try_from(*v).is_err()));
            let int = |k: &UidKey| match k {
                UidKey::Int(v) => *v,
                _ => unreachable!(),
            };
            if fits_i64 {
                Arc::new(Int64Array::from_iter(
                    get(is_int).map(|k| k.map(|k| int(k) as i64)),
                ))
            } else {
                Arc::new(UInt64Array::from_iter(
                    get(is_int).map(|k| k.map(|k| int(k) as u64)),
                ))
            }
        }
        Some(DataType::Float32 | DataType::Float64) => {
            Arc::new(Float64Array::from_iter(labels.iter().map(|k| match k {
                UidKey::Float(bits) => Some(TotalF64(*bits).value()),
                _ => None,
            })))
        }
        Some(DataType::Boolean) => {
            Arc::new(BooleanArray::from_iter(labels.iter().map(|k| match k {
                UidKey::Bool(v) => Some(*v),
                _ => None,
            })))
        }
        Some(DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View) => {
            Arc::new(LargeStringArray::from_iter(labels.iter().map(
                |k| match k {
                    // Keys were read from a string array, so the bytes are valid UTF-8.
                    UidKey::Bytes(v) => Some(std::str::from_utf8(v).expect("string user ID")),
                    _ => None,
                },
            )))
        }
        Some(DataType::Binary | DataType::LargeBinary | DataType::BinaryView) => Arc::new(
            LargeBinaryArray::from_iter(labels.iter().map(|k| match k {
                UidKey::Bytes(v) => Some(v.as_ref()),
                _ => None,
            })),
        ),
        _ => Arc::new(arrow_array::NullArray::new(labels.len())),
    }
}

/// Batch-at-a-time radius-of-gyration accumulator.
///
/// Pass 1: `accumulate_moments` per batch, then `finalize_centers`.
/// Pass 2: `accumulate_sq` per batch over the same rows, then `finalize`.
/// Each call takes the batch's user-ID column (or `None` for a single
/// individual) and float64 latitude/longitude columns, as any objects that
/// export an Arrow C stream (Polars Series, PyArrow chunked arrays, ...).
#[pyclass(name = "RadiusOfGyrationStream")]
pub struct PyRadiusOfGyrationStream {
    state: RogRowStream,
    uid_type: Option<DataType>,
}

impl PyRadiusOfGyrationStream {
    fn accumulate(
        &mut self,
        py: Python<'_>,
        pass: Pass,
        uids: Option<PyChunkedArray>,
        latitudes: PyChunkedArray,
        longitudes: PyChunkedArray,
    ) -> PyResult<()> {
        let data_type = match &uids {
            Some(uids) => uids.field().data_type().clone(),
            None => DataType::Null,
        };
        match &self.uid_type {
            Some(locked) if *locked != data_type => {
                return Err(PyTypeError::new_err(format!(
                    "user-ID type changed between batches ({locked} -> {data_type})"
                )));
            }
            None => self.uid_type = Some(data_type.clone()),
            _ => {}
        }

        let uid_chunks: Vec<ArrayRef> = match &uids {
            Some(uids) => uids.chunks().to_vec(),
            None => latitudes
                .chunks()
                .iter()
                .map(|c| Arc::new(arrow_array::NullArray::new(c.len())) as ArrayRef)
                .collect(),
        };
        let pieces = align_chunks(&[&uid_chunks, latitudes.chunks(), longitudes.chunks()])?;
        let coords: Vec<Coordinates> = pieces[1]
            .iter()
            .zip(&pieces[2])
            .map(|(lat, lng)| coordinates(lat, lng))
            .collect::<PyResult<_>>()?;
        let uid_pieces = &pieces[0];
        let state = &mut self.state;
        py.detach(|| dispatch(state, pass, &data_type, uid_pieces, &coords))?
            .map_err(PyValueError::new_err)
    }
}

#[pymethods]
impl PyRadiusOfGyrationStream {
    #[new]
    fn new() -> Self {
        Self {
            state: RogRowStream::new(),
            uid_type: None,
        }
    }

    fn __len__(&self) -> usize {
        self.state.num_users()
    }

    #[pyo3(signature = (uids, latitudes, longitudes))]
    fn accumulate_moments(
        &mut self,
        py: Python<'_>,
        uids: Option<PyChunkedArray>,
        latitudes: PyChunkedArray,
        longitudes: PyChunkedArray,
    ) -> PyResult<()> {
        self.accumulate(py, Pass::Moments, uids, latitudes, longitudes)
    }

    fn finalize_centers(&mut self) -> PyResult<()> {
        self.state.finalize_centers().map_err(PyValueError::new_err)
    }

    #[pyo3(signature = (uids, latitudes, longitudes))]
    fn accumulate_sq(
        &mut self,
        py: Python<'_>,
        uids: Option<PyChunkedArray>,
        latitudes: PyChunkedArray,
        longitudes: PyChunkedArray,
    ) -> PyResult<()> {
        self.accumulate(py, Pass::Squares, uids, latitudes, longitudes)
    }

    /// `(labels, radius_of_gyration, validity)` in first-appearance order.
    fn finalize<'py>(&self, py: Python<'py>) -> PyResult<PyRogStreamResult<'py>> {
        let (labels, values, validity) = self.state.finalize().map_err(PyValueError::new_err)?;
        let labels = labels_into_arrow(labels, self.uid_type.as_ref());
        Ok((
            Py::new(py, ArrowPyArray::from_array_ref(labels))?.into_any(),
            arrow_f64_output(py, values)?,
            validity.into_pyarray(py),
        ))
    }
}
