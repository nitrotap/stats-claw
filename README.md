# stats-claw

**Data science on the hot path.** In-process statistical computing for Rust —
distributions, hypothesis tests, and resampling — validated against `scipy` /
`scikit-learn` / `mlxtend`, with **zero runtime dependencies** (std-only).

Run inference where it has to live: inside a latency-critical Rust path, in-process and
deterministic, with no Python round-trip. The standout is a **classical hypothesis-test
suite with scipy-grade p-values** (exact and asymptotic modes) — the piece the Rust
ecosystem otherwise lacks.

## Validated against reference libraries

Every numeric is checked against committed golden fixtures from the established Python
libraries, within documented tolerances:

| Area | Reference | Tolerance |
|------|-----------|-----------|
| Distributions (pdf/cdf/ppf/moments, log-space tails) | `scipy.stats` | 1e-9–1e-12 |
| Hypothesis tests (t/ANOVA, χ²/Fisher/McNemar, KS/Anderson/Shapiro, Mann–Whitney/Wilcoxon/Kruskal) | `scipy.stats` (incl. `method="exact"`) | ≤1e-8 |
| Regression (OLS, ridge) | `scikit-learn` | ~1e-12 |
| Clustering / decomposition | `scikit-learn`, `ruptures` | per-test |
| Association rules | `mlxtend` | 1e-12 |

A hand-written deterministic PRNG gives byte-identical sampling across platforms.

## Highlights

- **Distributions** — pdf/pmf, cdf, quantile, moments, log-space cdf/sf, and seeded
  sampling for 14 families (normal, laplace, cauchy, uniform, exponential, gamma,
  weibull, lognormal, beta, chi-squared, Student's t, F, binomial, poisson).
- **Statistical tests** — parametric, non-parametric, categorical, goodness-of-fit, and
  correlation, with statistics, p-values (exact/asymptotic), and effect sizes.
- **Resampling** — bootstrap, jackknife, permutation, cross-validation, and
  confidence/credible intervals.
- **Also included** — optimizers, clustering/decomposition/change-point algorithms, and
  regression/density/outlier/feature-selection/association utilities. (These overlap
  mature crates like `linfa`/`smartcore`; the inference suite above is the focus.)
- **Streaming** — online estimators for latency-critical workloads.

## Quick start

```rust
use stats_claw::distributions::{Cdf, Moments, NormalDistribution, Pdf};

let n = NormalDistribution {
    mean: 0.0,
    standard_deviation: 1.0,
    ..Default::default()
};

// Peak of the standard normal is 1/sqrt(2*pi) ~= 0.398_942.
assert!((n.pdf(0.0) - 0.398_942_280_401_432_7).abs() < 1e-12);
assert!((n.cdf(0.0) - 0.5).abs() < 1e-12);
assert_eq!(n.variance(), Some(1.0));
```

## Status

Pre-stable (`0.x`): the public API is still being stabilized and may change before `1.0`.
MSRV 1.93.

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
