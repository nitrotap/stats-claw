<p align="center">
  <a href="https://github.com/nitrotap/stats-claw">
    <img src="https://raw.githubusercontent.com/nitrotap/stats-claw/main/assets/banner.png"
         alt="stats-claw — a Rust statistics tool" width="100%">
  </a>
</p>

<h1 align="center">stats-claw</h1>

<p align="center">
  <strong>Data science on the hot path.</strong><br>
  In-process statistical computing for Rust — distributions, hypothesis tests, and
  resampling — validated against <code>scipy</code>, with <strong>zero runtime dependencies</strong>.
</p>

<p align="center">
  <a href="https://crates.io/crates/stats-claw"><img alt="crates.io" src="https://img.shields.io/crates/v/stats-claw.svg?logo=rust&color=E8542F"></a>
  <a href="https://docs.rs/stats-claw"><img alt="docs.rs" src="https://img.shields.io/docsrs/stats-claw?logo=docs.rs"></a>
  <a href="https://github.com/nitrotap/stats-claw/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/nitrotap/stats-claw/actions/workflows/ci.yml/badge.svg"></a>
  <a href="#license"><img alt="license" src="https://img.shields.io/crates/l/stats-claw.svg"></a>
  <img alt="MSRV" src="https://img.shields.io/badge/rustc-1.93+-blue.svg">
</p>

---

Run inference where it has to live: inside a latency-critical Rust path, in-process and
deterministic, with no Python round-trip and nothing to link. The standout is a **classical
hypothesis-test suite with scipy-grade p-values** (exact and asymptotic modes) — the piece
the Rust ecosystem otherwise lacks.

```toml
[dependencies]
stats-claw = "0.1"
```

```rust
use stats_claw::distributions::{Cdf, Moments, NormalDistribution, Pdf, Quantile};

let z = NormalDistribution { mean: 0.0, standard_deviation: 1.0, ..Default::default() };

assert!((z.pdf(0.0) - 0.398_942_280_401_432_7).abs() < 1e-12); // 1/sqrt(2*pi)
assert!((z.cdf(0.0) - 0.5).abs() < 1e-12);
assert!((z.quantile(0.975) - 1.959_963_984_540_054).abs() < 1e-9);
assert_eq!(z.variance(), Some(1.0));
```

## Why stats-claw

- **Zero runtime dependencies.** std-only. No BLAS, no LAPACK, no transitive supply chain —
  it compiles in seconds and drops into any Rust binary, including constrained targets.
- **scipy-grade, and proven so.** Every numeric is checked against committed golden fixtures
  generated from the reference Python libraries, within documented tolerances (down to 1e-12).
- **Deterministic by construction.** A hand-written `SplitMix64` PRNG gives byte-identical
  sampling and resampling across platforms — reproducible tests, reproducible science.
- **The inference suite Rust was missing.** Exact and asymptotic p-values for the classical
  tests (t, ANOVA, χ², Fisher, Mann–Whitney, Wilcoxon, Kruskal–Wallis, KS, Shapiro), with
  log-space tails for extreme significance.

### Validated against reference libraries

| Area | Reference | Tolerance |
|------|-----------|-----------|
| Distributions (pdf/cdf/ppf/moments, log-space tails) | `scipy.stats` | 1e-9 – 1e-12 |
| Hypothesis tests (t/ANOVA, χ²/Fisher/McNemar, KS/Anderson/Shapiro, Mann–Whitney/Wilcoxon/Kruskal) | `scipy.stats` (incl. `method="exact"`) | ≤ 1e-8 |
| Regression (OLS, ridge) | `scikit-learn` | ~1e-12 |
| Clustering / decomposition / change-point | `scikit-learn`, `ruptures` | per-test |
| Association rules | `mlxtend` | 1e-12 |

## What's in the box

**Distributions** — `pdf`/`pmf`, `cdf`, `quantile`, `moments`, log-space `cdf`/`sf`, and
seeded sampling for 14 families: normal, uniform, Cauchy, Laplace, exponential, gamma, beta,
Weibull, log-normal, chi-squared, Student's t, F, binomial, and Poisson.

**Hypothesis tests** — parametric (one-sample / independent / paired / Welch t-tests, ANOVA,
variance), non-parametric (Mann–Whitney, Wilcoxon, Kruskal–Wallis, Friedman), categorical
(χ², Fisher, McNemar, Cochran), goodness-of-fit (KS, Anderson–Darling, Shapiro–Wilk), and
correlation — each returning the statistic, p-value (exact or asymptotic), degrees of freedom,
and effect size where defined.

**Resampling** — bootstrap, jackknife, permutation, cross-validation, and percentile
confidence / credible intervals, all driven by the deterministic PRNG.

**Streaming** — single-pass online estimators (`RunningMoments` via Welford, `P2Quantile`)
for latency-critical and unbounded-data workloads.

**Also included** — gradient and second-order optimizers (gradient descent, Adam, RMSProp,
AdaGrad, SGD, Newton, L-BFGS, conjugate-gradient) plus stochastic search (genetic, simulated
annealing); regression (OLS, ridge); clustering (k-means, DBSCAN, GMM, hierarchical, spectral,
mean-shift, affinity propagation); decomposition / embedding (PCA, ICA, factor analysis, t-SNE,
UMAP, LLE); and density, outlier, feature-selection, change-point (PELT), cardinality, and
association-rule utilities. These overlap mature crates like [`linfa`](https://github.com/rust-ml/linfa)
and [`smartcore`](https://github.com/smartcorelib/smartcore); the inference suite above is the focus.

## Usage

Distributions are plain parameter structs with `Default`; the behaviour lives in traits you
bring into scope (`Pdf`, `Pmf`, `Cdf`, `Quantile`, `Moments`, `LogCdf`, `Sample`).

### Distributions

```rust
use stats_claw::distributions::{Moments, Pmf, BinomialDistribution};

let coins = BinomialDistribution {
    number_of_trials: 10,
    success_probability: 0.5,
    ..Default::default()
};

assert_eq!(coins.mean(), Some(5.0));
assert!(coins.pmf(5) > coins.pmf(3)); // 5 heads is likelier than 3
```

### Deterministic sampling

```rust
use stats_claw::distributions::{NormalDistribution, Sample};
use stats_claw::rng::SplitMix64;

let dist = NormalDistribution { mean: 100.0, standard_deviation: 15.0, ..Default::default() };
let mut rng = SplitMix64::new(7); // same seed => same stream, on every platform

let draws: Vec<f64> = (0..1_000).map(|_| dist.sample(&mut rng)).collect();
assert_eq!(draws.len(), 1_000);
```

### Hypothesis testing

```rust
use stats_claw::tests_stat::{parametric::t_test_ind, Alternative};

let control = [5.1, 4.9, 5.3, 5.0, 4.8, 5.2];
let treated = [6.2, 6.0, 5.9, 6.4, 6.1, 6.3];

// Returns a TestResult; `?` works in any function returning stats_claw::error::Result.
let r = t_test_ind(&control, &treated, Alternative::TwoSided)?;

println!("t = {:.3}, p = {:.6}", r.statistic, r.p_value);
assert!(r.p_value < 0.05);             // significantly different means
assert_eq!(r.df, Some(10.0));          // n_a + n_b - 2
```

For ordinal or non-normal data, reach for the rank tests:

```rust
use stats_claw::tests_stat::{nonparametric::mann_whitney_u, Alternative};

let a = [1.0, 2.0, 3.0, 4.0];
let b = [10.0, 11.0, 12.0, 13.0];

let r = mann_whitney_u(&a, &b, Alternative::TwoSided, /* continuity */ true)?;
assert!(r.p_value < 0.05);
assert!(r.effect_size.is_some());      // rank-biserial correlation
```

### Bootstrap confidence intervals

```rust
use stats_claw::resampling::{bootstrap_statistic, percentile_ci};
use stats_claw::rng::SplitMix64;

let sample = [4.1, 5.5, 6.3, 4.8, 5.9, 6.1, 5.2, 4.6];
let mean = |xs: &[f64]| xs.iter().sum::<f64>() / xs.len() as f64;

let boots = bootstrap_statistic(&sample, 10_000, &mut SplitMix64::new(42), mean)?;
let (lo, hi) = percentile_ci(&boots, 0.05)?; // 95% interval
assert!(lo < hi);
```

### Streaming estimators

```rust
use stats_claw::streaming::RunningMoments;

let mut acc = RunningMoments::new();
for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
    acc.update(x); // O(1) memory, single pass
}

assert_eq!(acc.count(), 8);
assert!((acc.mean() - 5.0).abs() < 1e-12);
assert!(acc.variance() > 0.0); // Bessel-corrected (n-1) sample variance
```

### Regression

```rust
use stats_claw::algorithms::regression::ols;

let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]]; // one row per observation
let y = vec![2.0, 5.0, 8.0, 11.0];                        // y = 3x + 2

let model = ols(&x, &y)?;
assert!((model.intercept() - 2.0).abs() < 1e-9);
assert!((model.coefficients()[0] - 3.0).abs() < 1e-9);
```

## Performance

The hot paths are written for throughput: a SIMD ziggurat normal sampler (with a scalar
fallback and runtime feature detection), log-space tail evaluation to keep extreme p-values
accurate instead of underflowing to zero, and single-pass streaming estimators that hold O(1)
state. `unsafe` is confined to feature-gated SIMD intrinsics, each carrying a `// SAFETY:`
justification; everything else is safe Rust.

## Compatibility

- **MSRV:** Rust 1.93 (edition 2024). The minimum supported version is tested in CI.
- **Dependencies:** none at runtime (std-only).
- **Determinism:** identical results across operating systems and architectures.

## Status

Pre-stable (`0.x`): the public API is still being refined and may change before `1.0`,
following semantic versioning. The validation harness and tolerances are part of the test
suite, so numeric behaviour is held stable even while names settle.

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. Contributions are accepted under the same dual license.
