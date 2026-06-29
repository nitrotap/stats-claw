//! Criterion batch-throughput benchmark for the statistical-test family
//! (AC-7 / 7.1).
//!
//! Gated workload: **many independent two-sample t-tests** — run
//! `t_test_ind` over a batch of independent sample-pairs. This is the realistic
//! usage pattern for screening many features / hypotheses at once, and a
//! loop-bound hot path: the Python baseline (`scipy.stats.ttest_ind` called once
//! per pair) pays per-call dispatch and object-construction overhead that numpy
//! cannot fuse across independent tests, while the Rust loop runs natively.
//! `cargo bench --bench stat_tests` reports tests/sec, which the
//! published-results file turns into a factor against the Python baseline.
//!
//! Bench-only: never linked into the shipped crate.
//
// `missing_docs` is relaxed for this bench: `criterion_group!`/`criterion_main!`
// expand to undocumented glue functions. Not shipped code.
#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use stats_claw::rng::SplitMix64;
use stats_claw::tests_stat::parametric::t_test_ind;
use stats_claw::tests_stat::Alternative;

/// Number of independent sample-pairs (tests) in the batch; matched by the
/// Python baseline.
const N_TESTS: usize = 5_000;

/// Observations per sample in each pair; matched by the Python baseline.
const SAMPLE_SIZE: usize = 64;

/// Builds `N_TESTS` deterministic sample-pairs, each `SAMPLE_SIZE` observations,
/// with a small mean shift so the test does real work.
fn pairs(n: usize, m: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut rng = SplitMix64::new(0x5EED_1234);
    (0..n)
        .map(|_| {
            let a: Vec<f64> = (0..m).map(|_| rng.standard_normal()).collect();
            let b: Vec<f64> = (0..m).map(|_| rng.standard_normal() + 0.3).collect();
            (a, b)
        })
        .collect()
}

/// Benchmarks the batch of independent two-sample t-tests.
fn stat_tests_batch(c: &mut Criterion) {
    let data = pairs(N_TESTS, SAMPLE_SIZE);

    let mut group = c.benchmark_group("stat_tests_batch");
    group.throughput(Throughput::Elements(
        u64::try_from(N_TESTS).unwrap_or(u64::MAX),
    ));
    group.bench_function("t_test_ind_batch", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for (left, right) in &data {
                if let Ok(result) =
                    t_test_ind(black_box(left), black_box(right), Alternative::TwoSided)
                {
                    acc += result.statistic;
                }
            }
            black_box(acc)
        });
    });
    group.finish();
}

criterion_group!(benches, stat_tests_batch);
criterion_main!(benches);
