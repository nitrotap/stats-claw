//! Statistical hypothesis tests and their effect-size reporting.
//!
//! This module defines the shared result vocabulary ([`TestResult`],
//! [`Alternative`]) and houses the categorical, parametric, nonparametric,
//! goodness-of-fit, and correlation test families in subgroup folders. Each test
//! computes its statistic and p-value against the framework distributions
//! ([`crate::distributions`]) within the tolerances this crate commits to:
//! statistic relative error ≤ 1e-8, p-value absolute error ≤ 1e-8
//! (asymptotic ≤ 1e-6). Equivalence
//! is proven by the `tests/equiv.rs` suite (per-family modules under
//! `tests/stat/`) against committed scipy/statsmodels golden fixtures.

pub mod categorical;
pub mod correlation;
pub mod exact;
pub mod goodness_of_fit;
pub mod nonparametric;
pub mod parametric;
mod ranks;

use crate::distributions::ChiSquaredDistribution;
use crate::distributions::{Cdf, LogCdf};

/// Upper-tail probability `P(χ²_k ≥ x)` of the chi-squared null distribution.
///
/// The asymptotic p-value of every chi-squared-distributed statistic
/// (independence, Kruskal–Wallis, Friedman, Bartlett, Cochran) routes through
/// this single helper so they all share the framework
/// [`ChiSquaredDistribution`] CDF.
///
/// # Arguments
///
/// * `x` — the observed statistic; values `≤ 0` yield `1.0`.
/// * `df` — the degrees of freedom (`> 0`).
///
/// # Returns
///
/// The upper-tail probability in `[0, 1]`.
#[must_use]
pub(crate) fn chi_squared_upper_tail(x: f64, df: i64) -> f64 {
    let dist = ChiSquaredDistribution {
        degrees_of_freedom: df,
        ..Default::default()
    };
    (1.0 - dist.cdf(x)).clamp(0.0, 1.0)
}

/// Natural log of the upper-tail probability `ln P(χ²_k ≥ x)` — the log-space
/// counterpart of [`chi_squared_upper_tail`].
///
/// The chi-squared-routed tests (independence, Kruskal–Wallis, Friedman,
/// Bartlett, Cochran) report a log p-value through this single helper so the
/// extreme tail stays finite where [`chi_squared_upper_tail`] underflows to `0.0`.
///
/// # Arguments
///
/// * `x` — the observed statistic; values `≤ 0` yield `0.0` (log of p = 1).
/// * `df` — the degrees of freedom (`> 0`).
///
/// # Returns
///
/// `ln P(χ²_k ≥ x) ∈ (−∞, 0]`.
#[must_use]
pub(crate) fn chi_squared_upper_log_tail(x: f64, df: i64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    let dist = ChiSquaredDistribution {
        degrees_of_freedom: df,
        ..Default::default()
    };
    dist.logsf(x)
}

/// The sidedness of a hypothesis test, mirroring scipy's `alternative` argument.
///
/// Selects which tail (or both) contributes to the reported p-value. The exact
/// mapping from a statistic to a p-value is per-test, but the convention is
/// uniform: [`Self::TwoSided`] doubles the smaller tail (or integrates both),
/// [`Self::Less`] takes the lower tail, [`Self::Greater`] the upper tail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alternative {
    /// The effect may lie in either direction; both tails contribute.
    TwoSided,
    /// The alternative hypothesis is that the effect is below the null value.
    Less,
    /// The alternative hypothesis is that the effect is above the null value.
    Greater,
}

/// The computation mode for a test that offers both an exact (combinatorial)
/// null distribution and an asymptotic (large-sample) approximation.
///
/// Mirrors scipy's `method` argument for the rank and goodness-of-fit tests.
/// [`Self::Exact`] enumerates the exact null distribution (correct for small
/// samples but combinatorially expensive); [`Self::Asymptotic`] uses the normal
/// or Kolmogorov approximation (cheap, accurate for large samples);
/// [`Self::Auto`] resolves to exact below a documented per-test sample-size
/// threshold and to asymptotic at or above it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Enumerate the exact combinatorial null distribution.
    Exact,
    /// Use the large-sample (normal / Kolmogorov) approximation.
    Asymptotic,
    /// Pick exact below the threshold, asymptotic at or above it.
    Auto,
}

impl Mode {
    /// Resolves [`Self::Auto`] for sample size `n` against the per-test
    /// `threshold`, returning the effective mode actually used.
    ///
    /// [`Self::Exact`] and [`Self::Asymptotic`] pass through unchanged;
    /// [`Self::Auto`] becomes [`Self::Exact`] when `n < threshold` and
    /// [`Self::Asymptotic`] otherwise.
    ///
    /// # Arguments
    ///
    /// * `n` — the governing sample size for the test (per-test definition).
    /// * `threshold` — the size at or above which `Auto` chooses asymptotic.
    ///
    /// # Returns
    ///
    /// [`Self::Exact`] or [`Self::Asymptotic`] (never [`Self::Auto`]).
    #[must_use]
    pub const fn resolve(self, n: usize, threshold: usize) -> Self {
        match self {
            Self::Auto if n < threshold => Self::Exact,
            Self::Auto => Self::Asymptotic,
            other => other,
        }
    }
}

/// The outcome of a hypothesis test: its statistic, p-value, and — where the test
/// defines them — degrees of freedom and an effect size.
///
/// `df` and `effect_size` are `None` for tests that define no such quantity (e.g.
/// Shapiro–Wilk, Fisher exact), matching the reference's contract rather than
/// emitting a misleading numeric placeholder.
///
/// `log_p_value` is the natural log of the p-value (`ln(p_value)`, the
/// `scipy.stats` `logsf`/`logcdf` convention), populated for the tests whose null
/// is a continuous distribution with a numerically-stable log tail (the t-tests,
/// ANOVA/F, and the chi-squared-routed tests). It stays finite in the extreme tail
/// where the linear [`Self::p_value`] underflows to `0.0`. It is
/// `None` for tests that report no continuous-tail p-value in log space (e.g. the
/// exact rank tests and Shapiro–Wilk); [`Self::p_value`] is unchanged regardless.
#[derive(Debug, Clone, PartialEq)]
pub struct TestResult {
    /// The test statistic (χ², t, F, U, …) — its meaning is per-test.
    pub statistic: f64,
    /// The p-value in `[0, 1]` for the test's selected alternative.
    pub p_value: f64,
    /// `ln(p_value)`, finite in the extreme tail where `p_value` underflows, or
    /// `None` for tests with no log-space p-value path.
    pub log_p_value: Option<f64>,
    /// Degrees of freedom, or `None` when the test defines none.
    pub df: Option<f64>,
    /// The effect size (Cramér's V, η², rank-biserial, …), or `None`.
    pub effect_size: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::Mode;

    /// `Auto` picks the exact path strictly below the threshold and the
    /// asymptotic path at or above it; explicit modes pass through.
    #[test]
    fn auto_resolves_against_threshold() {
        assert_eq!(Mode::Auto.resolve(7, 8), Mode::Exact, "below threshold");
        assert_eq!(
            Mode::Auto.resolve(8, 8),
            Mode::Asymptotic,
            "at threshold is asymptotic"
        );
        assert_eq!(
            Mode::Auto.resolve(20, 8),
            Mode::Asymptotic,
            "above threshold"
        );
        assert_eq!(Mode::Exact.resolve(99, 8), Mode::Exact, "explicit exact");
        assert_eq!(
            Mode::Asymptotic.resolve(1, 8),
            Mode::Asymptotic,
            "explicit asymptotic"
        );
    }
}
