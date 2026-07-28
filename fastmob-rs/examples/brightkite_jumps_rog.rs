//! Runs the fastmob-rs jumps/RoG path over a Brightkite prefix and reports
//! timings plus a raw dump of the results, so `scripts/bench_fastmob_rs.py`
//! can time the Python path on the same input and diff the values bit-for-bit.
//!
//! Usage: `cargo run --release --example brightkite_jumps_rog -- <rows> <out_dir>`

#[path = "../benches/support/brightkite.rs"]
mod brightkite;

use std::io::Write;
use std::time::Instant;

use fastmob_rs::{Cols, jump_lengths_flat, prepare, radius_of_gyration_flat};

fn dump(path: &std::path::Path, values: &[f64]) {
    let mut file = std::io::BufWriter::new(
        std::fs::File::create(path)
            .unwrap_or_else(|err| panic!("create {}: {err}", path.display())),
    );
    for value in values {
        file.write_all(&value.to_le_bytes()).expect("write value");
    }
    file.flush().expect("flush dump");
}

fn main() {
    let mut args = std::env::args().skip(1);
    let rows: usize = args
        .next()
        .expect("usage: brightkite_jumps_rog <rows> <out_dir>")
        .parse()
        .expect("row count");
    let out_dir = std::path::PathBuf::from(
        args.next()
            .expect("usage: brightkite_jumps_rog <rows> <out_dir>"),
    );
    std::fs::create_dir_all(&out_dir).expect("create output directory");

    // Match the shape citybehavex reads from parquet: datetime already typed.
    let df = brightkite::load_parsed(rows);

    let started = Instant::now();
    let prep = prepare(&df, Cols::auto()).expect("prepare");
    let prepare_secs = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let jumps: Vec<f64> = jump_lengths_flat(&prep)
        .expect("jump lengths")
        .into_iter()
        .filter(|value| *value > 0.0)
        .collect();
    let jump_secs = started.elapsed().as_secs_f64();

    let started = Instant::now();
    let (rog_all, valid) = radius_of_gyration_flat(&prep);
    let rog: Vec<f64> = rog_all
        .into_iter()
        .zip(valid)
        .filter_map(|(value, is_valid)| is_valid.then_some(value))
        .collect();
    let rog_secs = started.elapsed().as_secs_f64();

    dump(&out_dir.join("jumps.f64"), &jumps);
    dump(&out_dir.join("rog.f64"), &rog);

    println!(
        r#"{{"rows":{},"cleaned_rows":{},"users":{},"prepare_secs":{},"jump_lengths_secs":{},"radius_of_gyration_secs":{},"total_secs":{},"n_jumps":{},"n_rog":{}}}"#,
        rows,
        prep.height(),
        prep.num_users(),
        prepare_secs,
        jump_secs,
        rog_secs,
        prepare_secs + jump_secs + rog_secs,
        jumps.len(),
        rog.len(),
    );
}
