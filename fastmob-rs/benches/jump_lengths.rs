//! fastmob-rs vs. citybehavex's current orchestration, on real Brightkite data.
//!
//! Three groups are measured:
//!
//! * `jumps_rog` — the end-to-end shape citybehavex actually calls: clean,
//!   arrange, then compute both jump lengths and radius of gyration. This is
//!   the number that decides whether the migration is worth doing.
//! * `prepare` — the arrange step alone, isolating the counting-sort strategy
//!   against Polars' whole-frame sort.
//! * `reused_prepare` — jump lengths alone over an already-prepared
//!   trajectory, showing what a caller saves on every measure after the first.

mod support;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

use fastmob_rs::{Cols, jump_lengths_flat, prepare, radius_of_gyration_flat};

use support::{brightkite, citybehavex_baseline};

const SIZES: &[usize] = &[100_000, 1_000_000, 4_000_000];

/// fastmob-rs equivalent of the baseline's `jumps_rog`, including the same
/// post-filtering, so the two sides produce identical output.
fn fastmob_rs_jumps_rog(df: &polars::prelude::DataFrame) -> (Vec<f64>, Vec<f64>) {
    let prep = prepare(df, Cols::auto()).expect("prepare");
    let jumps = jump_lengths_flat(&prep)
        .expect("jump lengths")
        .into_iter()
        .filter(|value| *value > 0.0)
        .collect();
    let (rog_all, valid) = radius_of_gyration_flat(&prep);
    let rog = rog_all
        .into_iter()
        .zip(valid)
        .filter_map(|(value, is_valid)| is_valid.then_some(value))
        .collect();
    (jumps, rog)
}

fn bench_jumps_rog(c: &mut Criterion) {
    for (label, parsed) in [
        ("jumps_rog_typed_datetime", true),
        ("jumps_rog_string_datetime", false),
    ] {
        let mut group = c.benchmark_group(label);
        group.sample_size(10);
        for &size in SIZES {
            let df = if parsed {
                brightkite::load_parsed(size)
            } else {
                brightkite::load(size)
            };
            group.bench_with_input(BenchmarkId::new("fastmob_rs", size), &df, |b, df| {
                b.iter(|| black_box(fastmob_rs_jumps_rog(df)));
            });
            group.bench_with_input(BenchmarkId::new("citybehavex", size), &df, |b, df| {
                b.iter(|| {
                    black_box(
                        citybehavex_baseline::jumps_rog(df, "uid", "lat", "lng", "datetime")
                            .expect("baseline jumps_rog"),
                    )
                });
            });
        }
        group.finish();
    }
}

fn bench_prepare(c: &mut Criterion) {
    let mut group = c.benchmark_group("prepare_typed_datetime");
    group.sample_size(10);
    for &size in SIZES {
        let df = brightkite::load_parsed(size);
        group.bench_with_input(BenchmarkId::new("fastmob_rs", size), &df, |b, df| {
            b.iter(|| black_box(prepare(df, Cols::auto()).expect("prepare")));
        });
    }
    group.finish();
}

fn bench_reused_prepare(c: &mut Criterion) {
    let mut group = c.benchmark_group("reused_prepare");
    group.sample_size(10);
    for &size in SIZES {
        let df = brightkite::load_parsed(size);
        let prep = prepare(&df, Cols::auto()).expect("prepare");
        group.bench_with_input(BenchmarkId::new("jump_lengths", size), &prep, |b, prep| {
            b.iter(|| black_box(jump_lengths_flat(prep).expect("jump lengths")));
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_jumps_rog,
    bench_prepare,
    bench_reused_prepare
);
criterion_main!(benches);
