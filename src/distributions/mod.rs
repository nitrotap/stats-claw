//! Probability distributions as plain parameter structs with behaviour traits.
//!
//! This module defines the behaviour traits (`Pdf`/`Pmf`/`Cdf`/`Quantile`/
//! `Moments`/`Sample`) every distribution implements, plus the shared
//! `bisection_quantile` inverse-CDF solver and a small integer→float helper.
//! Each concrete distribution lives in a subgroup folder (`symmetric/`,
//! `positive/`, `sampling/`, `discrete/`) and implements these traits for its
//! parameter struct, keeping the family uniform.
//!
//! The math is validated against committed `scipy.stats` golden fixtures by the
//! `distributions_equiv` integration suite within the documented tolerances
//! (pdf/cdf/ppf abs ≤ 1e-10, rel ≤ 1e-9 in tails; moments rel ≤ 1e-12).
//!
//! # Examples
//!
//! Build a distribution from its parameter struct and evaluate it through the
//! behaviour traits. Here the standard normal `N(0, 1)`: its density peaks at
//! `1/√(2π) ≈ 0.398_942` and its CDF is `0.5` at the mean.
//!
//! ```
//! use stats_claw::distributions::{Cdf, Moments, Pdf, Quantile};
//! use stats_claw::distributions::NormalDistribution;
//!
//! let n = NormalDistribution {
//!     mean: 0.0,
//!     standard_deviation: 1.0,
//!     ..Default::default()
//! };
//!
//! assert!((n.pdf(0.0) - 0.398_942_280_401_432_7).abs() < 1e-12);
//! assert!((n.cdf(0.0) - 0.5).abs() < 1e-12);
//! assert!((n.quantile(0.5) - 0.0).abs() < 1e-9);
//! assert_eq!(n.mean(), Some(0.0));
//! assert_eq!(n.variance(), Some(1.0));
//! ```

use crate::rng::SplitMix64;

pub mod types;
pub use types::*;
pub mod discrete;
pub mod positive;
pub mod sampling;
mod simd;
pub mod symmetric;
mod ziggurat;

/// Continuous probability density at a point.
pub trait Pdf {
    /// Evaluates the probability density function at `x`.
    ///
    /// # Arguments
    ///
    /// * `x` — the point at which to evaluate the density; any finite `f64`.
    ///
    /// # Returns
    ///
    /// The density `f(x) ≥ 0`, in units of probability per unit of `x`. It is a
    /// height, not a probability, and may exceed 1.
    fn pdf(&self, x: f64) -> f64;
}

/// Discrete probability mass at an integer support point.
pub trait Pmf {
    /// Evaluates the probability mass function at integer `k`.
    ///
    /// # Arguments
    ///
    /// * `k` — a support point; returns `0.0` outside the distribution's support.
    ///
    /// # Returns
    ///
    /// The probability `P(X = k) ∈ [0, 1]`.
    fn pmf(&self, k: i64) -> f64;
}

/// Cumulative distribution function.
pub trait Cdf {
    /// Evaluates the cumulative distribution function at `x`.
    ///
    /// # Arguments
    ///
    /// * `x` — the upper limit; any finite `f64`.
    ///
    /// # Returns
    ///
    /// `P(X ≤ x) ∈ [0, 1]`, monotonically non-decreasing in `x`.
    fn cdf(&self, x: f64) -> f64;
}

/// Log-space cumulative and survival functions, for extreme-tail probabilities
/// that underflow to `0.0` (or saturate to `1.0`) in linear space.
///
/// Mirrors `scipy.stats.<dist>.logcdf` / `logsf`. The deep tail of a continuous
/// null distribution — where a statistical test's p-value lives — is exactly
/// where the linear [`Cdf`] loses all precision (`1 - cdf(x)` rounds to `0.0`
/// once `cdf(x) ≥ 1 - 2⁻⁵³`); these log-space evaluations stay finite and
/// accurate there, so a test can report an honest log p-value.
pub trait LogCdf {
    /// Evaluates the natural log of the CDF, `ln P(X ≤ x)`.
    ///
    /// # Arguments
    ///
    /// * `x` — the upper limit; any finite `f64`.
    ///
    /// # Returns
    ///
    /// `ln P(X ≤ x) ∈ (−∞, 0]`, finite in the left tail where `cdf(x)` underflows.
    fn logcdf(&self, x: f64) -> f64;

    /// Evaluates the natural log of the survival function, `ln P(X > x)`.
    ///
    /// # Arguments
    ///
    /// * `x` — the lower limit; any finite `f64`.
    ///
    /// # Returns
    ///
    /// `ln P(X > x) ∈ (−∞, 0]`, finite in the right tail where `1 - cdf(x)`
    /// underflows.
    fn logsf(&self, x: f64) -> f64;
}

/// Inverse cumulative distribution function (quantile / percent-point).
pub trait Quantile {
    /// Evaluates the quantile (inverse CDF) at probability `p`.
    ///
    /// # Arguments
    ///
    /// * `p` — a probability in `[0, 1]`; `0`/`1` map to the support endpoints
    ///   (`±∞` for unbounded support).
    ///
    /// # Returns
    ///
    /// The smallest `x` with `cdf(x) ≥ p`.
    fn quantile(&self, p: f64) -> f64;
}

/// First two moments, reporting `None` where the moment is undefined (matching
/// scipy's NaN for e.g. Cauchy or the T distribution with low degrees of
/// freedom).
pub trait Moments {
    /// Returns the theoretical mean, or `None` if it is undefined.
    fn mean(&self) -> Option<f64>;
    /// Returns the theoretical variance, or `None` if it is undefined.
    fn variance(&self) -> Option<f64>;
}

/// Draws a single variate from a seeded, reproducible RNG.
pub trait Sample {
    /// Draws one variate, advancing `rng`.
    ///
    /// # Arguments
    ///
    /// * `rng` — the deterministic generator; a fixed seed yields a fixed stream.
    ///
    /// # Returns
    ///
    /// A single draw from the distribution.
    fn sample(&self, rng: &mut SplitMix64) -> f64;
}

/// Generic inverse-CDF via bracketed bisection, for distributions whose quantile
/// has no closed form.
///
/// The bracket `[lo, hi]` must contain the target quantile and `cdf` must be
/// monotone non-decreasing on it. Two hundred halvings drive the residual well
/// below the round-trip tolerance for any double-precision bracket.
///
/// # Arguments
///
/// * `p` — target probability; `p ≤ 0` returns `lo`, `p ≥ 1` returns `hi`.
/// * `lo` — lower bracket bound (a value with `cdf(lo) ≤ p`).
/// * `hi` — upper bracket bound (a value with `cdf(hi) ≥ p`).
/// * `cdf` — the monotone CDF to invert.
///
/// # Returns
///
/// The bracket midpoint after convergence — an `x` with `cdf(x) ≈ p`.
pub(crate) fn bisection_quantile(p: f64, lo: f64, hi: f64, cdf: impl Fn(f64) -> f64) -> f64 {
    if p <= 0.0 {
        return lo;
    }
    if p >= 1.0 {
        return hi;
    }
    let (mut lo, mut hi) = (lo, hi);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Converts a small non-negative count to `f64` losslessly by domain.
///
/// Distribution parameters that arrive as integers (degrees of freedom, trial
/// counts, support indices) are far below `i32::MAX`, so routing through
/// `i32::try_from` then the lossless `f64::from` is exact and avoids the banned
/// `as` cast. Values outside `i32` range saturate to `i32::MAX`/`MIN`, which the
/// supported parameter domains never reach.
///
/// # Arguments
///
/// * `n` — an integer parameter (e.g. degrees of freedom or trial count).
///
/// # Returns
///
/// `n` as an `f64`.
pub(crate) fn count_to_f64(n: i64) -> f64 {
    let clamped = i32::try_from(n).unwrap_or(if n < 0 { i32::MIN } else { i32::MAX });
    f64::from(clamped)
}
