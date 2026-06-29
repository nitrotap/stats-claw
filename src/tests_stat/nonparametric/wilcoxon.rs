//! Wilcoxon signed-rank test (normal approximation, zero/tie handling).

use super::normal_cdf;
use crate::error::{Error, Result};
use crate::tests_stat::exact::wilcoxon as exact;
use crate::tests_stat::parametric::len_f64;
use crate::tests_stat::ranks::{mid_ranks, tie_correction};
use crate::tests_stat::{Alternative, Mode, TestResult};

/// Default `Auto` threshold: the exact signed-rank distribution is used when the
/// number of nonzero differences is below this, otherwise the normal
/// approximation. At `n = 24` the exact convolution spans `24·25/2 + 1 = 301`
/// cells over `2^24` patterns — still cheap, and below where the normal
/// approximation is already excellent.
const EXACT_THRESHOLD: usize = 25;

/// Wilcoxon signed-rank test on the paired differences `a − b`, matching
/// `scipy.stats.wilcoxon(a, b, correction, mode="approx", alternative)`.
///
/// Drops zero differences, ranks the absolute differences with mid-ranks, and
/// sums the ranks of the positive differences (`W₊`). The reported statistic is
/// `min(W₊, W₋)` for a two-sided test and `W₊` for a one-sided test, matching
/// scipy. The p-value is the normal approximation with the tie-corrected variance
/// and an optional half-unit continuity correction.
///
/// # Arguments
///
/// * `a`, `b` — paired samples of equal length.
/// * `alternative` — tested direction of the median difference.
/// * `use_continuity` — apply the half-unit continuity correction (scipy default).
///
/// # Returns
///
/// A [`TestResult`] with the Wilcoxon statistic, approximate p-value, no df, and
/// no effect size.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] for length-mismatched inputs and
/// [`Error::InsufficientData`] when every difference is zero (no signed ranks).
///
/// # Examples
///
/// ```
/// use stats_claw::tests_stat::{nonparametric::wilcoxon_signed_rank, Alternative};
///
/// // Clearly different pairs should be significant.
/// let a = [10.0_f64, 11.0, 12.0, 13.0];
/// let b = [1.0_f64, 2.0, 3.0, 4.0];
/// let r = wilcoxon_signed_rank(&a, &b, Alternative::Greater, true)?;
/// assert!(r.p_value < 0.05, "p was {}", r.p_value);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn wilcoxon_signed_rank(
    a: &[f64],
    b: &[f64],
    alternative: Alternative,
    use_continuity: bool,
) -> Result<TestResult> {
    if a.len() != b.len() {
        return Err(Error::InvalidInput(
            "paired samples differ in length".to_owned(),
        ));
    }
    let diffs: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| x - y)
        .filter(|d| d.abs() > 0.0)
        .collect();
    if diffs.is_empty() {
        return Err(Error::InsufficientData);
    }
    let abs: Vec<f64> = diffs.iter().map(|d| d.abs()).collect();
    let ranks = mid_ranks(&abs);
    let mut r_plus = 0.0;
    let mut r_minus = 0.0;
    for (d, r) in diffs.iter().zip(&ranks) {
        if *d > 0.0 {
            r_plus += r;
        } else {
            r_minus += r;
        }
    }

    let count = len_f64(diffs.len());
    let mean = count * (count + 1.0) * 0.25;
    let tie = tie_correction(&abs);
    let var = count * (count + 1.0) * (2.0f64.mul_add(count, 1.0)) / 24.0 - tie / 48.0;
    if var <= 0.0 {
        return Err(Error::InsufficientData);
    }
    let se = var.sqrt();

    let statistic = match alternative {
        Alternative::TwoSided => r_plus.min(r_minus),
        _ => r_plus,
    };
    let p_value = p_value(r_plus, mean, se, alternative, use_continuity);
    Ok(TestResult {
        statistic,
        p_value,
        log_p_value: None,
        df: None,
        effect_size: None,
    })
}

/// Wilcoxon signed-rank test with an explicit exact / asymptotic / auto mode
/// selector, matching `scipy.stats.wilcoxon(a, b, alternative, method)`.
///
/// In [`Mode::Exact`] the p-value comes from the exact signed-rank-sum null
/// distribution ([`crate::tests_stat::exact::wilcoxon`]); in [`Mode::Asymptotic`]
/// it comes from the same normal approximation as [`wilcoxon_signed_rank`];
/// [`Mode::Auto`] uses exact when the number of nonzero differences is below
/// [`EXACT_THRESHOLD`] **and** the absolute differences are untied, otherwise
/// asymptotic. The reported statistic matches [`wilcoxon_signed_rank`].
///
/// # Arguments
///
/// * `a`, `b` — paired samples of equal length.
/// * `alternative` — tested direction of the median difference.
/// * `use_continuity` — continuity correction for the asymptotic p-value (ignored
///   by the exact path).
/// * `mode` — the p-value computation mode.
///
/// # Returns
///
/// A [`TestResult`] with the Wilcoxon statistic, the mode's p-value, no df, and
/// no effect size.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] for length-mismatched inputs or when
/// [`Mode::Exact`] is requested on data with tied absolute differences (the exact
/// distribution is defined only for untied ranks), and [`Error::InsufficientData`]
/// when every difference is zero.
pub fn wilcoxon_signed_rank_mode(
    a: &[f64],
    b: &[f64],
    alternative: Alternative,
    use_continuity: bool,
    mode: Mode,
) -> Result<TestResult> {
    if a.len() != b.len() {
        return Err(Error::InvalidInput(
            "paired samples differ in length".to_owned(),
        ));
    }
    let diffs: Vec<f64> = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| x - y)
        .filter(|d| d.abs() > 0.0)
        .collect();
    if diffs.is_empty() {
        return Err(Error::InsufficientData);
    }
    let abs: Vec<f64> = diffs.iter().map(|d| d.abs()).collect();
    let nonzero = diffs.len();
    let has_ties = tie_correction(&abs) > 0.0;

    let resolved = match mode.resolve(nonzero, EXACT_THRESHOLD) {
        Mode::Exact if has_ties && matches!(mode, Mode::Auto) => Mode::Asymptotic,
        other => other,
    };
    if matches!(resolved, Mode::Exact) && has_ties {
        return Err(Error::InvalidInput(
            "exact Wilcoxon requires untied absolute differences".to_owned(),
        ));
    }
    if matches!(resolved, Mode::Asymptotic) {
        return wilcoxon_signed_rank(a, b, alternative, use_continuity);
    }

    // Exact path: signed-rank sums then the exact null tail.
    let ranks = mid_ranks(&abs);
    let mut r_plus = 0.0;
    let mut r_minus = 0.0;
    for (d, r) in diffs.iter().zip(&ranks) {
        if *d > 0.0 {
            r_plus += r;
        } else {
            r_minus += r;
        }
    }
    let statistic = match alternative {
        Alternative::TwoSided => r_plus.min(r_minus),
        _ => r_plus,
    };
    let p_value = exact::p_value(r_plus, nonzero, alternative);
    Ok(TestResult {
        statistic,
        p_value,
        log_p_value: None,
        df: None,
        effect_size: None,
    })
}

/// Normal-approximation p-value of the signed-rank sum `r_plus`.
fn p_value(r_plus: f64, mean: f64, se: f64, alternative: Alternative, use_continuity: bool) -> f64 {
    let diff = r_plus - mean;
    let cc = if use_continuity { 0.5 } else { 0.0 };
    match alternative {
        // Larger r_plus → median difference > 0.
        Alternative::Greater => {
            let z = (diff - cc) / se;
            (1.0 - normal_cdf(z)).clamp(0.0, 1.0)
        }
        Alternative::Less => {
            let z = (diff + cc) / se;
            normal_cdf(z).clamp(0.0, 1.0)
        }
        Alternative::TwoSided => {
            let z = (diff.abs() - cc) / se;
            (2.0 * (1.0 - normal_cdf(z))).clamp(0.0, 1.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Length-mismatched pairs are rejected.
    #[test]
    fn length_mismatch_is_invalid() {
        assert!(matches!(
            wilcoxon_signed_rank(&[1.0, 2.0], &[1.0], Alternative::TwoSided, true),
            Err(Error::InvalidInput(_))
        ));
    }

    /// All-zero differences leave no signed ranks.
    #[test]
    fn all_zero_differences_are_insufficient() {
        let a = [3.0, 4.0];
        assert!(matches!(
            wilcoxon_signed_rank(&a, &a, Alternative::TwoSided, true),
            Err(Error::InsufficientData)
        ));
    }
}
