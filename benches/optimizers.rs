//! Criterion batch-throughput benchmark for the optimizer family.
//!
//! Gated workload: a **multi-start conjugate-gradient sweep** — solve the
//! Rosenbrock problem from many independent starting points to convergence. This
//! is the realistic global-optimization usage pattern (multi-start to escape the
//! non-convex valley), and a loop-bound hot path: the Python baseline
//! (`scipy.optimize.minimize(method="CG")`, the same Fletcher–Reeves algorithm)
//! pays per-iteration Python-callback and dispatch overhead on every line-search
//! evaluation, while the Rust solver runs a tight native loop. `cargo bench
//! --bench optimizers` reports solves/sec, which the published-results file turns
//! into a factor against the Python baseline.
//!
//! Bench-only: never linked into the shipped crate.
//
// `missing_docs` is relaxed for this bench: `criterion_group!`/`criterion_main!`
// expand to undocumented glue functions. Not shipped code.
#![allow(missing_docs)]

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use stats_claw::optimizers::gradient::conjugate_gradient;
use stats_claw::optimizers::objectives::Rosenbrock;

/// Number of independent starting points in the multi-start sweep; matched by the
/// Python baseline.
const N_STARTS: usize = 2_000;

/// Maximum CG iterations per solve; matched by the Python baseline's `maxiter`.
const MAX_ITER: usize = 200;

/// Gradient-norm convergence tolerance; matched by the Python baseline's `gtol`.
const TOL: f64 = 1e-6;

/// Builds `N_STARTS` deterministic starting points spread over a box around the
/// Rosenbrock valley, so each solve does real iterative work.
fn starts(n: usize) -> Vec<[f64; 2]> {
    (0..n)
        .map(|i| {
            let t = f64::from(u32::try_from(i).unwrap_or(u32::MAX));
            let span = f64::from(u32::try_from(n).unwrap_or(1));
            // Spread x0 over [-2, 2] and x1 over [-1, 3] deterministically.
            let frac = t / span;
            [4.0f64.mul_add(frac, -2.0), 4.0f64.mul_add(frac, -1.0)]
        })
        .collect()
}

/// Benchmarks the multi-start conjugate-gradient sweep over Rosenbrock.
fn optimizer_multistart(c: &mut Criterion) {
    let obj = Rosenbrock;
    let points = starts(N_STARTS);

    let mut group = c.benchmark_group("optimizer_batch");
    group.throughput(Throughput::Elements(
        u64::try_from(N_STARTS).unwrap_or(u64::MAX),
    ));
    group.bench_function("cg_multistart_rosenbrock", |b| {
        b.iter(|| {
            let mut acc = 0.0;
            for p in &points {
                let result = conjugate_gradient(&obj, black_box(&p[..]), MAX_ITER, TOL);
                acc += result.fx;
            }
            black_box(acc)
        });
    });
    group.finish();
}

criterion_group!(benches, optimizer_multistart);
criterion_main!(benches);
