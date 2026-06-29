//! Parametric hypothesis tests (t-tests, ANOVA, variance tests).
//!
//! Houses the four t-tests (one-sample, independent pooled, paired, Welch),
//! one-way ANOVA, and the Levene/Bartlett variance tests. The t and F p-values
//! route through the framework null distributions
//! ([`crate::distributions::TDistribution`], [`crate::distributions::FDistribution`]).

pub mod anova;
pub mod t_test;
pub mod variance_tests;

pub use anova::one_way_anova;
pub use t_test::{t_test_1samp, t_test_ind, t_test_paired, t_test_welch};
pub use variance_tests::{bartlett, levene};

use crate::distributions::{Cdf, LogCdf};
use crate::distributions::{FDistribution, TDistribution};
use crate::tests_stat::Alternative;

/// Sample mean of `xs`, or `0.0` for an empty slice (callers validate emptiness).
pub(crate) fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / len_f64(xs.len())
}

/// Unbiased (Bessel-corrected, `n−1`) sample variance of `xs`.
///
/// Returns `0.0` for fewer than two observations; callers that require positive
/// variance check the result and raise a typed error.
pub(crate) fn variance(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let ss: f64 = xs.iter().map(|&x| (x - m) * (x - m)).sum();
    ss / len_f64(n - 1)
}

/// Widens a length to `f64` without an `as` cast (lengths are far below `2^53`).
pub(crate) fn len_f64(n: usize) -> f64 {
    let wide = u64::try_from(n).unwrap_or(u64::MAX);
    let hi = u32::try_from(wide >> 32).unwrap_or(0);
    let lo = u32::try_from(wide & 0xFFFF_FFFF).unwrap_or(0);
    f64::from(hi).mul_add(4_294_967_296.0, f64::from(lo))
}

/// Floors a non-negative `f64` to an `i64` without an `as` cast.
///
/// Binary-searches the integer ranks in `0..=cap`, comparing each candidate's
/// exact `f64` widening against `x`, so no `f64 → integer` conversion is needed
/// (the `style.rs` guard bans `as`). Values above `cap` saturate at `cap`.
pub(crate) fn floor_to_i64(x: f64) -> i64 {
    if x <= 0.0 {
        return 0;
    }
    let cap: i64 = 1_000_000_000;
    let (mut lo, mut hi) = (0i64, cap);
    while lo < hi {
        let mid = lo + (hi - lo + 1) / 2;
        if i64_to_f64(mid) <= x {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Widens a small non-negative `i64` to `f64` without an `as` cast.
fn i64_to_f64(n: i64) -> f64 {
    let wide = u64::try_from(n).unwrap_or(0);
    let hi = u32::try_from(wide >> 32).unwrap_or(0);
    let lo = u32::try_from(wide & 0xFFFF_FFFF).unwrap_or(0);
    f64::from(hi).mul_add(4_294_967_296.0, f64::from(lo))
}

/// Maps a t-statistic and (possibly fractional) degrees of freedom to a p-value
/// for the chosen `alternative`, via the framework Student's t CDF.
///
/// scipy uses the continuous t distribution with fractional df (Welch); the
/// [`TDistribution`] carries integral df, so for fractional df the
/// p-value is linearly interpolated between the two bracketing integer-df CDFs —
/// matching scipy to the asymptotic tolerance.
///
/// # Arguments
///
/// * `t` — the observed t-statistic.
/// * `df` — degrees of freedom (`> 0`, possibly fractional for Welch).
/// * `alternative` — which tail(s) contribute.
///
/// # Returns
///
/// The p-value in `[0, 1]`.
pub(crate) fn p_from_t(t: f64, df: f64, alternative: Alternative) -> f64 {
    let cdf = |x: f64| t_cdf_fractional(x, df);
    let p = match alternative {
        Alternative::Less => cdf(t),
        Alternative::Greater => 1.0 - cdf(t),
        Alternative::TwoSided => 2.0 * (1.0 - cdf(t.abs())),
    };
    p.clamp(0.0, 1.0)
}

/// Maps a t-statistic and degrees of freedom to a **log** p-value for the chosen
/// `alternative` — the log-space counterpart of [`p_from_t`].
///
/// Routes through [`TDistribution`]'s `logsf`/`logcdf` so the extreme tail stays
/// finite where [`p_from_t`] underflows to `0.0`. Fractional (Welch) df is
/// handled by linearly interpolating the log p-value between the two bracketing
/// integer-df evaluations, mirroring the linear-space interpolation in
/// [`t_cdf_fractional`].
///
/// # Arguments
///
/// * `t` — the observed t-statistic.
/// * `df` — degrees of freedom (`> 0`, possibly fractional for Welch).
/// * `alternative` — which tail(s) contribute.
///
/// # Returns
///
/// `ln(p)`, the natural log of the p-value (`≤ 0`), finite in the extreme tail.
pub(crate) fn log_p_from_t(t: f64, df: f64, alternative: Alternative) -> f64 {
    let lo_df = df.floor();
    let lo = log_p_from_t_integer(t, floor_to_i64(lo_df), alternative);
    if (df - lo_df).abs() < 1e-12 {
        return lo.min(0.0);
    }
    let hi = log_p_from_t_integer(t, floor_to_i64(lo_df) + 1, alternative);
    let frac = df - lo_df;
    frac.mul_add(hi - lo, lo).min(0.0)
}

/// Log p-value of a t-statistic at integral `df` for the chosen `alternative`.
fn log_p_from_t_integer(t: f64, df: i64, alternative: Alternative) -> f64 {
    let dist = TDistribution {
        degrees_of_freedom: df.max(1),
        ..Default::default()
    };
    match alternative {
        Alternative::Less => dist.logcdf(t),
        Alternative::Greater => dist.logsf(t),
        // Two-sided p = 2·sf(|t|); ln p = ln 2 + ln sf(|t|).
        Alternative::TwoSided => std::f64::consts::LN_2 + dist.logsf(t.abs()),
    }
}

/// Student's t CDF at `x` for fractional `df`, interpolating between integer-df
/// CDFs of the [`TDistribution`].
fn t_cdf_fractional(x: f64, df: f64) -> f64 {
    let lo_df = df.floor();
    let lo = t_cdf_integer(x, floor_to_i64(lo_df));
    if (df - lo_df).abs() < 1e-12 {
        return lo;
    }
    let hi = t_cdf_integer(x, floor_to_i64(lo_df) + 1);
    let frac = df - lo_df;
    frac.mul_add(hi - lo, lo)
}

/// Student's t CDF at `x` for an integral degrees-of-freedom value.
fn t_cdf_integer(x: f64, df: i64) -> f64 {
    let dist = TDistribution {
        degrees_of_freedom: df.max(1),
        ..Default::default()
    };
    dist.cdf(x)
}

/// Upper-tail probability `P(F ≥ x)` of the F null distribution with the given
/// numerator and denominator degrees of freedom.
///
/// The p-value of every F-distributed statistic (one-way ANOVA, Levene) routes
/// through this helper so they share the framework [`FDistribution`] CDF.
///
/// # Arguments
///
/// * `f` — the observed F-statistic; values `≤ 0` yield `1.0`.
/// * `df_num` — numerator degrees of freedom (`> 0`).
/// * `df_den` — denominator degrees of freedom (`> 0`).
///
/// # Returns
///
/// The upper-tail probability in `[0, 1]`.
pub(crate) fn f_upper_tail(f: f64, df_num: i64, df_den: i64) -> f64 {
    let dist = FDistribution {
        numerator_df: df_num,
        denominator_df: df_den,
        ..Default::default()
    };
    (1.0 - dist.cdf(f)).clamp(0.0, 1.0)
}

/// Natural log of the F upper-tail probability `ln P(F ≥ x)` — the log-space
/// counterpart of [`f_upper_tail`], finite where the linear tail underflows.
///
/// # Arguments
///
/// * `f` — the observed F-statistic; values `≤ 0` yield `0.0` (log of p = 1).
/// * `df_num` — numerator degrees of freedom (`> 0`).
/// * `df_den` — denominator degrees of freedom (`> 0`).
///
/// # Returns
///
/// `ln P(F ≥ x) ∈ (−∞, 0]`.
pub(crate) fn f_upper_log_tail(f: f64, df_num: i64, df_den: i64) -> f64 {
    if f <= 0.0 {
        return 0.0;
    }
    let dist = FDistribution {
        numerator_df: df_num,
        denominator_df: df_den,
        ..Default::default()
    };
    dist.logsf(f).min(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `floor_to_i64` matches integer floors over a range of small inputs.
    #[test]
    fn floor_to_i64_floors_correctly() {
        assert_eq!(floor_to_i64(0.0), 0);
        assert_eq!(floor_to_i64(8.999), 8);
        assert_eq!(floor_to_i64(17.964), 17);
        assert_eq!(floor_to_i64(-3.0), 0);
    }

    /// The unbiased variance of a known sample matches the hand computation.
    #[test]
    fn variance_is_bessel_corrected() {
        let v = variance(&[2.0, 4.0, 6.0]);
        assert!((v - 4.0).abs() < 1e-12, "variance was {v}");
    }
}
