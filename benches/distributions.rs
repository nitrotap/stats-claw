//! Criterion batch-throughput benchmarks for the distribution family (AC-7 / 7.1).
//!
//! Benchmarks the batch hot paths `pdf` / `cdf` / `sample` over a large input
//! vector for the Normal family, plus a single `gradient_descent` step as a
//! representative optimizer hot path. `cargo bench --bench distributions` reports
//! per-operation timings that the published-results script converts into a
//! throughput factor against the scipy baseline.
//!
//! Bench-only: never linked into the shipped crate.
//
// `missing_docs` / `significant_drop_tightening` are relaxed for this bench:
// `criterion_group!`/`criterion_main!` expand to undocumented glue functions, and
// criterion's `BenchmarkGroup` is designed to live across `bench_function` calls
// until `finish()`. Neither is shipped code.
#![allow(missing_docs, clippy::significant_drop_tightening)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use stats_claw::distributions::NormalDistribution;
use stats_claw::optimizers::gradient::gradient_descent;
use stats_claw::optimizers::objectives::Quadratic;
use stats_claw::rng::SplitMix64;

/// Number of elements in the batch workload; matched by the scipy baseline.
const N: usize = 100_000;

/// Builds the standard-ish normal used across the benchmarks.
fn normal() -> NormalDistribution {
    NormalDistribution {
        mean: 0.0,
        standard_deviation: 1.0,
        ..Default::default()
    }
}

/// Evenly spaced evaluation grid over roughly `[-5, 5]`.
fn grid(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| {
            let t = f64::from(u32::try_from(i).unwrap_or(u32::MAX));
            t.mul_add(10.0 / f64::from(u32::try_from(n).unwrap_or(1)), -5.0)
        })
        .collect()
}

/// Benchmarks batch `pdf`, `cdf`, `sample`, and one optimizer step.
fn distribution_batch(c: &mut Criterion) {
    let dist = normal();
    let xs = grid(N);

    let mut out = vec![0.0; N];

    let mut group = c.benchmark_group("normal_batch");
    group.throughput(Throughput::Elements(u64::try_from(N).unwrap_or(u64::MAX)));

    // Dense batch hot path: the native-SIMD `pdf_batch` over the whole grid into a
    // reused buffer. This is the gated metric — the genuine apples-to-apples
    // comparison against vectorized `scipy.stats.norm.pdf`.
    group.bench_function("pdf", |b| {
        b.iter(|| {
            dist.pdf_batch(black_box(&xs), black_box(&mut out));
            black_box(out.first().copied())
        });
    });

    group.bench_function("cdf", |b| {
        b.iter(|| {
            dist.cdf_batch(black_box(&xs), black_box(&mut out));
            black_box(out.first().copied())
        });
    });

    // Dense batch sampling hot path: the native-SIMD ziggurat `sample_batch` fills
    // the whole buffer, the genuine apples-to-apples comparison against numpy's
    // vectorized `scipy.stats.norm.rvs(size=N)`. (The earlier per-draw accumulation
    // measured an online workload vs stdlib, a different — favorable — comparison.)
    let mut samples = vec![0.0; N];
    group.bench_function("sample", |b| {
        b.iter(|| {
            let mut rng = SplitMix64::new(0x00C0_FFEE);
            dist.sample_batch(&mut rng, black_box(&mut samples));
            // Reduce the *whole* buffer so the optimizer cannot elide writing the
            // tail elements (anchoring only `samples[0]` let a newer toolchain hoist
            // most of the fill, inflating throughput ~12×). Summing forces every
            // element to be materialized.
            black_box(samples.iter().sum::<f64>())
        });
    });
    group.finish();

    c.bench_function("gradient_descent_step", |b| {
        let obj = Quadratic::new(vec![3.0, -2.0]);
        b.iter(|| {
            black_box(gradient_descent(
                &obj,
                black_box(&[0.0, 0.0]),
                0.1,
                1,
                1e-12,
            ))
        });
    });
}

criterion_group!(benches, distribution_batch);
criterion_main!(benches);
