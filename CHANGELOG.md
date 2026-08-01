# Changelog

All notable changes to this crate are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). Pre-`1.0`, minor versions may
contain breaking changes.

## [0.2.0] — 2026-07-31

Adds a formal-verification layer and several new public modules. Still zero runtime
dependencies (std-only), still MSRV 1.93.

### Added
- **68 Kani proof harnesses**, co-located with the code they verify as `#[cfg(kani)] mod
  verification` blocks. They model-check panic-freedom, index-bounds, and structural
  invariants: exhaustively over scalar input domains (`next_f64() ∈ [0, 1)` for all 2^64
  PRNG states, every ziggurat table index in bounds for every drawn `u64`, the HLL
  register index below `2^p` at every supported precision, the 2×2 solver total), and at
  small bounded collection sizes for the size-parameterised ones (permutation
  bijectivity, k-fold partitioning). Reproduce with `cargo kani -Z stubbing`; see
  `VERIFICATION.md` for the full inventory, the per-property size bounds, and the
  documented limits — notably that Kani over-approximates transcendentals, so numerical
  accuracy remains the equivalence suite's job.
- `VERIFICATION.md`, describing what is and is not proven, and the Miri
  undefined-behaviour gate covering the SIMD `unsafe` that model checking cannot reach.
- CI job running the proof suite. It is gated to the nightly schedule, manual dispatch,
  and `v*` release tags — not every push — and `publish` now waits on it.
- **Two-way ANOVA** (`tests_stat::parametric`) with interaction and partial η².
- **Naive Bayes classification** (`algorithms::classification`): Gaussian and Categorical,
  with a `ClassificationResult` carrier.
- **Divisive (DIANA) hierarchical clustering** (`algorithms::clustering`).
- **Likelihood module** (`likelihood::*`): maximum-likelihood fitting for the normal,
  binomial, Poisson, exponential, and categorical families.
- **Resampling configuration types** at `resampling::*` — `CrossValidation`,
  `JackknifeResampling`, `LeaveOneOutCrossValidation`, `MonteCarloResampling`,
  `StratifiedCrossValidation` — plus stratified k-fold, leave-one-out CV, and
  Monte-Carlo estimation with Phipson–Smyth corrected p-values.

### Changed
- `cfg(kani)` is registered in the manifest's `[lints.rust]`, so the proof harnesses are
  invisible to `cargo build`/`test`/`clippy` and produce no warnings. Kani is a
  development and CI tool only; it is not a dependency.
- `VERIFICATION.md` is included in the published tarball alongside the README and
  CHANGELOG.
- README documents the determinism guarantee more precisely: the PRNG stream is
  byte-identical across targets, and so is any plain-arithmetic reduction over it (Rust
  does not reassociate floating-point). The exception is `SplitMix64::standard_normal`,
  whose Box–Muller transform routes through the platform math library — accurate to well
  under an ULP, but not correctly rounded — so normal draws and the statistics accumulated
  from them can differ in the last ULP or two between targets, and even between
  optimisation levels of the same source. `standard_normal` now carries that caveat in its
  rustdoc.

### Fixed
- The Monte-Carlo regression tests no longer pin the seeded standard-normal mean, standard
  error, and p-value to exact bit patterns. Those pins asserted cross-target bit-identity
  of the platform math library's `ln`/`sin`/`cos` and so only held on the machine and build
  that recorded them: of the 100 000 draws in the pinned run, 15 900 differ by 1–2 ULP
  between `x86_64` and aarch64 (moving the pinned mean by 26 ULP), and 100 differ between
  an optimised and an unoptimised build on one machine. The normal-sampler tests now check
  the theoretical value within a four-standard-error band, and the bit-exact determinism
  pins moved onto the uniform stream, where bit-exactness is a guarantee this crate can
  actually keep.

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
