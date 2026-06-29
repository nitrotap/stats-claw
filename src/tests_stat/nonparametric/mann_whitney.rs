//! Mann–Whitney U test (normal approximation with tie and continuity handling).

use super::normal_cdf;
use crate::error::{Error, Result};
use crate::tests_stat::exact::mann_whitney as exact;
use crate::tests_stat::parametric::len_f64;
use crate::tests_stat::ranks::{mid_ranks, tie_correction};
use crate::tests_stat::{Alternative, Mode, TestResult};

/// Default `Auto` threshold: the exact U distribution is used when the larger
/// sample size is below this, otherwise the normal approximation. With the larger
/// sample at `7`, the exact distribution has at most `8·7 + 1 = 57` cells.
const EXACT_THRESHOLD: usize = 8;

/// Mann–Whitney U test of `a` versus `b` via the normal approximation, matching
/// `scipy.stats.mannwhitneyu(a, b, use_continuity, alternative)`.
///
/// Pools both samples, assigns mid-ranks, and forms `U₁ = R_a − nₐ(nₐ+1)/2`
/// (the reported statistic). The p-value is a normal approximation with the
/// tie-corrected variance and, when `use_continuity` is set, a half-unit
/// continuity correction; the tail is chosen by `alternative`. The reported
/// effect size is the rank-biserial correlation.
///
/// # Arguments
///
/// * `a`, `b` — the two independent samples (each non-empty).
/// * `alternative` — `Less`/`Greater` test the location of `a` relative to `b`.
/// * `use_continuity` — apply the half-unit continuity correction (scipy default).
///
/// # Returns
///
/// A [`TestResult`] with `statistic = U₁`, the approximate p-value, no df, and
/// `effect_size = Some(rank-biserial)`.
///
/// # Errors
///
/// Returns [`Error::EmptyInput`] if either sample is empty and
/// [`Error::DegenerateInput`] when every value ties (zero variance).
///
/// # Examples
///
/// ```
/// use stats_claw::tests_stat::{nonparametric::mann_whitney_u, Alternative};
///
/// // Well-separated samples yield a significant two-sided p-value.
/// let a = [1.0_f64, 2.0, 3.0, 4.0];
/// let b = [10.0_f64, 11.0, 12.0, 13.0];
/// let r = mann_whitney_u(&a, &b, Alternative::TwoSided, true)?;
/// assert!(r.p_value < 0.05, "p was {}", r.p_value);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn mann_whitney_u(
    a: &[f64],
    b: &[f64],
    alternative: Alternative,
    use_continuity: bool,
) -> Result<TestResult> {
    if a.is_empty() || b.is_empty() {
        return Err(Error::EmptyInput);
    }
    let n1 = a.len();
    let n2 = b.len();
    let mut pooled = Vec::with_capacity(n1 + n2);
    pooled.extend_from_slice(a);
    pooled.extend_from_slice(b);
    let ranks = mid_ranks(&pooled);
    let rank_sum_a: f64 = ranks.iter().take(n1).sum();

    let n1_f = len_f64(n1);
    let n2_f = len_f64(n2);
    let u1 = rank_sum_a - n1_f * (n1_f + 1.0) / 2.0;
    let u2 = n1_f.mul_add(n2_f, -u1);

    let n = n1_f + n2_f;
    let tie = tie_correction(&pooled);
    let variance = n1_f * n2_f / 12.0 * ((n + 1.0) - tie / (n * (n - 1.0)));
    if variance <= 0.0 {
        return Err(Error::DegenerateInput("all values tied".to_owned()));
    }
    let sigma = variance.sqrt();
    let mu = n1_f * n2_f / 2.0;

    let p_value = p_value(u1, u2, mu, sigma, alternative, use_continuity);
    let rank_biserial = 1.0 - 2.0 * u2 / (n1_f * n2_f);

    Ok(TestResult {
        statistic: u1,
        p_value,
        log_p_value: None,
        df: None,
        effect_size: Some(rank_biserial),
    })
}

/// Mann–Whitney U test with an explicit exact / asymptotic / auto mode selector,
/// matching `scipy.stats.mannwhitneyu(a, b, alternative, method)`.
///
/// In [`Mode::Exact`] the p-value comes from the exact U null distribution
/// ([`crate::tests_stat::exact::mann_whitney`]); in [`Mode::Asymptotic`] it comes
/// from the same normal approximation as [`mann_whitney_u`]; [`Mode::Auto`] uses
/// exact when the larger sample size is below `EXACT_THRESHOLD` **and** there
/// are no ties, otherwise asymptotic. The reported statistic (`U₁`) and effect
/// size (rank-biserial) are identical across modes.
///
/// # Arguments
///
/// * `a`, `b` — the two independent samples (each non-empty).
/// * `alternative` — the tested direction.
/// * `use_continuity` — continuity correction for the asymptotic p-value (ignored
///   by the exact path, which needs none).
/// * `mode` — the p-value computation mode.
///
/// # Returns
///
/// A [`TestResult`] with `statistic = U₁`, the mode's p-value, no df, and
/// `effect_size = Some(rank-biserial)`.
///
/// # Errors
///
/// Returns [`Error::EmptyInput`] if either sample is empty,
/// [`Error::DegenerateInput`] when every value ties, and
/// [`Error::InvalidInput`] if [`Mode::Exact`] is requested on tied data (the
/// exact U distribution is defined only for untied samples).
pub fn mann_whitney_u_mode(
    a: &[f64],
    b: &[f64],
    alternative: Alternative,
    use_continuity: bool,
    mode: Mode,
) -> Result<TestResult> {
    if a.is_empty() || b.is_empty() {
        return Err(Error::EmptyInput);
    }
    let n1 = a.len();
    let n2 = b.len();
    let mut pooled = Vec::with_capacity(n1 + n2);
    pooled.extend_from_slice(a);
    pooled.extend_from_slice(b);
    let has_ties = tie_correction(&pooled) > 0.0;

    let resolved = match mode.resolve(n1.max(n2), EXACT_THRESHOLD) {
        // Auto never picks exact when ties are present (exact is undefined then).
        Mode::Exact if has_ties && matches!(mode, Mode::Auto) => Mode::Asymptotic,
        other => other,
    };
    if matches!(resolved, Mode::Exact) && has_ties {
        return Err(Error::InvalidInput(
            "exact Mann–Whitney requires untied data".to_owned(),
        ));
    }
    if matches!(resolved, Mode::Asymptotic) {
        return mann_whitney_u(a, b, alternative, use_continuity);
    }

    // Exact path: reuse the rank-based statistic, replace only the p-value.
    let ranks = mid_ranks(&pooled);
    let rank_sum_a: f64 = ranks.iter().take(n1).sum();
    let n1_f = len_f64(n1);
    let n2_f = len_f64(n2);
    let u1 = rank_sum_a - n1_f * (n1_f + 1.0) / 2.0;
    let u2 = n1_f.mul_add(n2_f, -u1);
    let p_value = exact::p_value(u1, n1, n2, alternative);
    let rank_biserial = 1.0 - 2.0 * u2 / (n1_f * n2_f);
    Ok(TestResult {
        statistic: u1,
        p_value,
        log_p_value: None,
        df: None,
        effect_size: Some(rank_biserial),
    })
}

/// Computes the normal-approximation p-value for the chosen alternative, matching
/// scipy's continuity-corrected convention.
fn p_value(
    u1: f64,
    u2: f64,
    mu: f64,
    sigma: f64,
    alternative: Alternative,
    use_continuity: bool,
) -> f64 {
    let cc = if use_continuity { 0.5 } else { 0.0 };
    match alternative {
        // P(U_a large) = upper tail on U1.
        Alternative::Greater => {
            let z = (u1 - mu - cc) / sigma;
            (1.0 - normal_cdf(z)).clamp(0.0, 1.0)
        }
        // P(U_a small) = upper tail on U2 (= lower tail on U1).
        Alternative::Less => {
            let z = (u2 - mu - cc) / sigma;
            (1.0 - normal_cdf(z)).clamp(0.0, 1.0)
        }
        // Two-sided doubles the larger-tail probability.
        Alternative::TwoSided => {
            let u = u1.max(u2);
            let z = (u - mu - cc) / sigma;
            (2.0 * (1.0 - normal_cdf(z))).clamp(0.0, 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Well-separated samples give a small two-sided p-value.
    #[test]
    fn separated_samples_are_significant() -> Result<()> {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [10.0, 11.0, 12.0, 13.0];
        let r = mann_whitney_u(&a, &b, Alternative::TwoSided, true)?;
        assert!(r.p_value < 0.05, "p was {}", r.p_value);
        Ok(())
    }

    /// An empty sample is a typed error.
    #[test]
    fn empty_sample_is_error() {
        assert!(matches!(
            mann_whitney_u(&[], &[1.0], Alternative::TwoSided, true),
            Err(Error::EmptyInput)
        ));
    }
}
