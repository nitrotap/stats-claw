# Changelog

All notable changes to this crate are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Pre-`1.0`, minor versions may
contain breaking changes.

## [0.1.0] — 2026-06-29

Initial release. Zero runtime dependencies (std-only). MSRV 1.93.

### Added
- **Distributions** (14 families): pdf/pmf, cdf, quantile, moments, log-space cdf/sf, and
  seeded sampling — validated against `scipy.stats` (1e-9–1e-12).
- **Hypothesis tests**: parametric (t-test, ANOVA, variance), non-parametric
  (Mann–Whitney, Wilcoxon, Kruskal–Wallis, Friedman), categorical (χ², Fisher, McNemar,
  Cochran), goodness-of-fit (KS, Anderson–Darling, Shapiro–Wilk), and correlation — with
  exact and asymptotic p-value modes and log-space tails, validated against `scipy.stats`
  (including `method="exact"`) to ≤1e-8.
- **Resampling**: bootstrap, jackknife, permutation, cross-validation, and
  confidence/credible intervals.
- **Special functions**: error, gamma, and beta families with numerically-stable
  log-space variants.
- **Additional numerics**: optimizers; clustering, decomposition, and change-point
  algorithms; and regression/density/outlier/feature-selection/association utilities,
  each validated against `scikit-learn` / `ruptures` / `mlxtend`.
- **Streaming** online estimators.
