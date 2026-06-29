//! Criterion batch-throughput benchmark for the resampling family.
//!
//! Gated workload: a **bootstrap of the median** — draw `B` with-replacement
//! resamples of a sample and recompute the median on each, the realistic
//! bootstrap-confidence-interval usage pattern. This is a loop-bound hot path:
//! the bootstrap loop is inherently sequential (each resample draws `n` indices
//! from the RNG), and the per-resample statistic (a median, requiring a sort)
//! cannot be expressed as one vectorized numpy call, so the Python baseline pays
//! per-resample interpreter overhead. `cargo bench --bench resampling` reports
//! resamples/sec, which the published-results file turns into a factor against
//! the Python baseline.
//!
//! Bench-only: never linked into the shipped crate.
//
// `missing_docs` is relaxed for this bench: `criterion_group!`/`criterion_main!`
// expand to undocumented glue functions. Not shipped code.
#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use stats_claw::resampling::bootstrap_statistic;
use stats_claw::rng::SplitMix64;

/// Number of bootstrap resamples; matched by the Python baseline.
const B: usize = 5_000;

/// Size of the original sample; matched by the Python baseline.
const SAMPLE_SIZE: usize = 200;

/// Builds a deterministic sample to bootstrap.
fn sample(n: usize) -> Vec<f64> {
    let mut rng = SplitMix64::new(0xB007_5747);
    (0..n).map(|_| rng.standard_normal()).collect()
}

/// Median of a slice via a sorted copy (the per-resample statistic).
fn median(xs: &[f64]) -> f64 {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = v.len() / 2;
    if v.len().is_multiple_of(2) {
        f64::midpoint(*v.get(mid - 1).unwrap_or(&0.0), *v.get(mid).unwrap_or(&0.0))
    } else {
        *v.get(mid).unwrap_or(&0.0)
    }
}

/// Benchmarks the bootstrap-of-the-median over `B` resamples.
fn resampling_batch(c: &mut Criterion) {
    let data = sample(SAMPLE_SIZE);

    let mut group = c.benchmark_group("resampling_batch");
    group.throughput(Throughput::Elements(u64::try_from(B).unwrap_or(u64::MAX)));
    group.bench_function("bootstrap_median", |b| {
        b.iter(|| {
            let mut rng = SplitMix64::new(0x00C0_FFEE);
            let stats = bootstrap_statistic(black_box(&data), B, &mut rng, median);
            black_box(stats.map_or(0, |s| s.len()))
        });
    });
    group.finish();
}

criterion_group!(benches, resampling_batch);
criterion_main!(benches);
