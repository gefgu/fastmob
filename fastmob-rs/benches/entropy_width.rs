//! Compare the entropy kernel with u64 and u32 location codes.

use std::fs::File;
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::time::Instant;

use flate2::read::GzDecoder;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHashSet};

const ROWS: usize = 4_000_000;
const RUNS: usize = 10;

fn dataset_path() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("tests/shared/data/loc-brightkite_totalCheckins.txt.gz")
}

fn load() -> (Vec<u64>, Vec<u32>, Vec<(usize, usize)>) {
    let file = File::open(dataset_path()).expect("Brightkite cache is missing");
    let reader = BufReader::new(GzDecoder::new(file));
    let mut rows = Vec::with_capacity(ROWS);

    for line in reader.lines() {
        if rows.len() == ROWS {
            break;
        }
        let line = line.expect("read Brightkite row");
        let mut fields = line.split('\t');
        let (Some(uid), Some(datetime), Some(_lat), Some(_lng), Some(location)) = (
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
        ) else {
            continue;
        };
        let Ok(uid) = uid.parse::<u64>() else {
            continue;
        };
        rows.push((uid, datetime.to_owned(), location.to_owned()));
    }
    rows.sort_unstable_by(|a, b| (a.0, &a.1).cmp(&(b.0, &b.1)));

    let mut locations = FxHashMap::<String, u64>::default();
    let mut next = 0u64;
    let mut ids64 = Vec::with_capacity(rows.len());
    let mut ranges = Vec::new();
    let mut current_uid = None;
    let mut start = 0;
    for (index, (uid, _, location)) in rows.into_iter().enumerate() {
        if current_uid != Some(uid) {
            if index > start {
                ranges.push((start, index));
            }
            start = index;
            current_uid = Some(uid);
        }
        let code = *locations.entry(location).or_insert_with(|| {
            let value = next;
            next += 1;
            value
        });
        ids64.push(code);
    }
    if start < ids64.len() {
        ranges.push((start, ids64.len()));
    }
    assert!(next <= u32::MAX as u64, "u32 location-code overflow");
    let ids32 = ids64.iter().map(|&value| value as u32).collect();
    (ids64, ids32, ranges)
}

fn entropy_u32(ids: &[u32], ranges: &[(usize, usize)]) -> Vec<f64> {
    ranges
        .par_iter()
        .map(|&(start, end)| {
            let sequence = &ids[start..end];
            let n = sequence.len();
            if n <= 1 {
                return 0.0;
            }
            let mut col_max = vec![1u32; n];
            let mut prev_row = vec![1u32; n];
            let mut curr_row = vec![1u32; n];
            for i in 1..n {
                for j in (i + 1)..n {
                    if sequence[i - 1] == sequence[j - 1] {
                        curr_row[j] = prev_row[j - 1] + 1;
                    } else {
                        curr_row[j] = 1;
                    }
                    if curr_row[j] > col_max[j] {
                        col_max[j] = curr_row[j];
                    }
                }
                std::mem::swap(&mut prev_row, &mut curr_row);
            }
            let lambdas: u64 = col_max.iter().map(|&value| value as u64).sum();
            if lambdas == 0 {
                0.0
            } else {
                (n as f64 / lambdas as f64) * (n as f64).log2()
            }
        })
        .collect()
}

fn entropy_u64(ids: &[u64], ranges: &[(usize, usize)]) -> Vec<f64> {
    ranges
        .par_iter()
        .map(|&(start, end)| {
            let sequence = &ids[start..end];
            let n = sequence.len();
            if n <= 1 {
                return 0.0;
            }
            let mut col_max = vec![1usize; n];
            let mut prev_row = vec![1usize; n];
            let mut curr_row = vec![1usize; n];
            for i in 1..n {
                for j in (i + 1)..n {
                    curr_row[j] = if sequence[i - 1] == sequence[j - 1] {
                        prev_row[j - 1] + 1
                    } else {
                        1
                    };
                    if curr_row[j] > col_max[j] {
                        col_max[j] = curr_row[j];
                    }
                }
                std::mem::swap(&mut prev_row, &mut curr_row);
            }
            let lambdas: usize = col_max.iter().sum();
            (n as f64 / lambdas as f64) * (n as f64).log2()
        })
        .collect()
}

fn measure<F: FnMut() -> Vec<f64>>(label: &str, mut run: F) {
    let mut times = Vec::new();
    for _ in 0..RUNS {
        let started = Instant::now();
        let output = black_box(run());
        black_box(output);
        times.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    println!(
        "{label}: {}",
        times
            .iter()
            .map(|value| format!("{value:.3} ms"))
            .collect::<Vec<_>>()
            .join(", ")
    );
}

fn main() {
    let (ids64, ids32, ranges) = load();
    println!(
        "rows={}, users={}, unique_locations={}",
        ids64.len(),
        ranges.len(),
        ids64.iter().copied().collect::<FxHashSet<_>>().len()
    );
    let mut ids64_runs: Vec<_> = (0..RUNS).map(|_| ids64.clone()).collect();
    let mut ids32_runs: Vec<_> = (0..RUNS).map(|_| ids32.clone()).collect();
    let mut range_runs: Vec<_> = (0..RUNS).map(|_| ranges.clone()).collect();
    measure("u64", || {
        let ids = ids64_runs.pop().unwrap();
        let run_ranges = range_runs.pop().unwrap();
        entropy_u64(&ids, &run_ranges)
    });
    measure("u32", || entropy_u32(&ids32_runs.pop().unwrap(), &ranges));
}
