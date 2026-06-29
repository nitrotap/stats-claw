//! Criterion batch-throughput benchmark for the algorithms family.
//!
//! Gated workload: **PELT change-point detection** on a long piecewise-constant
//! signal. The Python baseline is `ruptures.Pelt(model="l2")`, a pure-Python
//! dynamic program — exactly the reference the equivalence suite checks against.
//! PELT is an inherently sequential dynamic program that neither library
//! vectorizes, so the interpreted Python loop pays full per-step overhead while
//! the Rust implementation runs natively. `cargo bench --bench algorithms`
//! reports samples/sec (signal length processed per second), which the
//! published-results file turns into a factor against the Python baseline.
//!
//! Bench-only: never linked into the shipped crate.
//
// `missing_docs` is relaxed for this bench: `criterion_group!`/`criterion_main!`
// expand to undocumented glue functions. Not shipped code.
#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use stats_claw::algorithms::change_point::pelt_l2;

/// Length of the piecewise-constant signal; matched by the Python baseline.
const N: usize = 2_000;

/// PELT L2 penalty; matched by the Python baseline.
const PENALTY: f64 = 10.0;

/// Minimum segment size; matched by the Python baseline.
const MIN_SIZE: usize = 5;

/// Builds a deterministic piecewise-constant signal with several level shifts so
/// PELT does real segmentation work.
fn signal(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            // Five segments with distinct levels; a small deterministic ripple
            // keeps each segment non-degenerate.
            let seg = (i * 5) / n;
            let level = f64::from(u32::try_from(seg).unwrap_or(0)) * 3.0;
            let t = f64::from(u32::try_from(i % 7).unwrap_or(0));
            level + t.mul_add(0.1, -0.3)
        })
        .collect()
}

/// Benchmarks PELT change-point detection on the long signal.
fn algorithms_batch(c: &mut Criterion) {
    let xs = signal(N);

    let mut group = c.benchmark_group("algorithms_batch");
    group.throughput(Throughput::Elements(u64::try_from(N).unwrap_or(u64::MAX)));
    group.bench_function("pelt_l2_changepoint", |b| {
        b.iter(|| {
            let segments = pelt_l2(black_box(&xs), PENALTY, MIN_SIZE);
            black_box(segments.len())
        });
    });
    group.finish();
}

criterion_group!(benches, algorithms_batch);
criterion_main!(benches);
