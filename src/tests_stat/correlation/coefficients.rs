//! Pearson, Spearman, and Kendall correlation coefficients and their p-values.

use super::kendall_exact::kendall_two_sided_exact;
use crate::error::{Error, Result};
use crate::tests_stat::parametric::{len_f64, log_p_from_t, mean, p_from_t};
use crate::tests_stat::ranks::mid_ranks;
use crate::tests_stat::{Alternative, TestResult};

/// Pearson product-moment correlation of `x` and `y`, matching
/// `scipy.stats.pearsonr(x, y, alternative=…)`.
///
/// The statistic is the linear correlation `r ∈ [−1, 1]`; the p-value comes from
/// the framework Student's t distribution via `t = r·√((n−2)/(1−r²))` on `n − 2`
/// degrees of freedom for the chosen alternative.
///
/// # Arguments
///
/// * `x`, `y` — paired samples of equal length (≥ 3 observations).
/// * `alternative` — tested direction of the correlation.
///
/// # Returns
///
/// A [`TestResult`] with `statistic = r`, the p-value, `df = n − 2`, no effect
/// size.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] for mismatched lengths,
/// [`Error::InsufficientData`] for fewer than three pairs, and
/// [`Error::DegenerateInput`] when either input has zero variance.
///
/// # Examples
///
/// ```
/// use stats_claw::tests_stat::{correlation::pearson, Alternative};
///
/// // Perfectly linear data has a Pearson correlation of 1.
/// let x = [1.0_f64, 2.0, 3.0, 4.0];
/// let y = [2.0_f64, 4.0, 6.0, 8.0];
/// let r = pearson(&x, &y, Alternative::TwoSided)?;
/// assert!((r.statistic - 1.0).abs() < 1e-12, "r was {}", r.statistic);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn pearson(x: &[f64], y: &[f64], alternative: Alternative) -> Result<TestResult> {
    let r = pearson_r(x, y)?;
    Ok(correlation_t_result(r, x.len(), alternative))
}

/// Spearman rank correlation of `x` and `y`, matching
/// `scipy.stats.spearmanr(x, y, alternative=…)`.
///
/// Computes the Pearson correlation of the mid-ranks; the p-value uses the same
/// Student's t approximation as [`pearson`] on `n − 2` degrees of freedom.
///
/// # Arguments
///
/// * `x`, `y` — paired samples of equal length (≥ 3 observations).
/// * `alternative` — tested direction of the monotone association.
///
/// # Returns
///
/// A [`TestResult`] with `statistic = ρ`, the p-value, `df = n − 2`.
///
/// # Errors
///
/// As [`pearson`], after ranking.
pub fn spearman(x: &[f64], y: &[f64], alternative: Alternative) -> Result<TestResult> {
    if x.len() != y.len() {
        return Err(Error::InvalidInput("samples differ in length".to_owned()));
    }
    let rx = mid_ranks(x);
    let ry = mid_ranks(y);
    let rho = pearson_r(&rx, &ry)?;
    Ok(correlation_t_result(rho, x.len(), alternative))
}

/// Kendall's tau-b correlation of `x` and `y`, matching
/// `scipy.stats.kendalltau(x, y, alternative=…)`.
///
/// The statistic is tau-b (tie-adjusted). With no ties the p-value uses the exact
/// permutation null distribution (scipy's default for small, untied data);
/// otherwise it falls back to the tie-corrected normal approximation.
///
/// # Arguments
///
/// * `x`, `y` — paired samples of equal length (≥ 2 observations).
/// * `alternative` — tested direction of the association.
///
/// # Returns
///
/// A [`TestResult`] with `statistic = τ`, the p-value, no df, no effect size.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] for mismatched lengths and
/// [`Error::InsufficientData`] for fewer than two pairs.
pub fn kendall(x: &[f64], y: &[f64], alternative: Alternative) -> Result<TestResult> {
    if x.len() != y.len() {
        return Err(Error::InvalidInput("samples differ in length".to_owned()));
    }
    let n = x.len();
    if n < 2 {
        return Err(Error::InsufficientData);
    }
    let (concordant, discordant, tie_x, tie_y) = pair_counts(x, y);
    let n0 = concordant + discordant + tie_x + tie_y;
    let denom = ((n0 - tie_x) * (n0 - tie_y)).sqrt();
    let tau = if denom > 0.0 {
        (concordant - discordant) / denom
    } else {
        0.0
    };
    let has_ties = tie_x > 0.0 || tie_y > 0.0;
    let p_value = if has_ties {
        kendall_normal_p(tau, n, alternative)
    } else {
        kendall_two_sided_exact(n, discordant, tau, alternative)
    };
    Ok(TestResult {
        statistic: tau,
        p_value,
        // Kendall's tau uses an exact-permutation / normal-approx null, not a
        // continuous-tail distribution with a cheap log-sf; no log p-value here.
        log_p_value: None,
        df: None,
        effect_size: None,
    })
}

/// Pearson correlation coefficient of two equal-length samples.
fn pearson_r(x: &[f64], y: &[f64]) -> Result<f64> {
    if x.len() != y.len() {
        return Err(Error::InvalidInput("samples differ in length".to_owned()));
    }
    if x.len() < 3 {
        return Err(Error::InsufficientData);
    }
    let (mx, my) = (mean(x), mean(y));
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for (&xi, &yi) in x.iter().zip(y) {
        let dx = xi - mx;
        let dy = yi - my;
        sxy = dx.mul_add(dy, sxy);
        sxx = dx.mul_add(dx, sxx);
        syy = dy.mul_add(dy, syy);
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return Err(Error::DegenerateInput("zero-variance input".to_owned()));
    }
    Ok((sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0))
}

/// Builds a t-based [`TestResult`] from a correlation coefficient on `n` pairs.
fn correlation_t_result(r: f64, n: usize, alternative: Alternative) -> TestResult {
    let df = len_f64(n - 2);
    let one_minus = r.mul_add(-r, 1.0).max(0.0);
    let t = if one_minus <= 0.0 {
        f64::INFINITY.copysign(r)
    } else {
        r * (df / one_minus).sqrt()
    };
    TestResult {
        statistic: r,
        p_value: p_from_t(t, df, alternative),
        log_p_value: Some(log_p_from_t(t, df, alternative)),
        df: Some(df),
        effect_size: None,
    }
}

/// Counts concordant, discordant, and per-variable tied pairs of `x`, `y`.
fn pair_counts(x: &[f64], y: &[f64]) -> (f64, f64, f64, f64) {
    let n = x.len();
    let (mut con, mut dis, mut tx, mut ty) = (0.0, 0.0, 0.0, 0.0);
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = sign(x.get(j), x.get(i));
            let dy = sign(y.get(j), y.get(i));
            if dx == 0.0 && dy == 0.0 {
                tx += 1.0;
                ty += 1.0;
            } else if dx == 0.0 {
                tx += 1.0;
            } else if dy == 0.0 {
                ty += 1.0;
            } else if (dx > 0.0) == (dy > 0.0) {
                con += 1.0;
            } else {
                dis += 1.0;
            }
        }
    }
    (con, dis, tx, ty)
}

/// Sign of `a − b` for two optional slice elements (`0.0` when equal/absent).
fn sign(a: Option<&f64>, b: Option<&f64>) -> f64 {
    match (a, b) {
        (Some(&a), Some(&b)) if a > b => 1.0,
        (Some(&a), Some(&b)) if a < b => -1.0,
        _ => 0.0,
    }
}

/// Normal-approximation Kendall p-value (used when ties are present).
fn kendall_normal_p(tau: f64, n: usize, alternative: Alternative) -> f64 {
    use crate::tests_stat::nonparametric::normal_cdf as cdf;
    let n_f = len_f64(n);
    let var = 2.0 * (2.0f64.mul_add(n_f, 5.0)) / (9.0 * n_f * (n_f - 1.0));
    let z = tau / var.sqrt();
    match alternative {
        Alternative::Greater => (1.0 - cdf(z)).clamp(0.0, 1.0),
        Alternative::Less => cdf(z).clamp(0.0, 1.0),
        Alternative::TwoSided => (2.0 * (1.0 - cdf(z.abs()))).clamp(0.0, 1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Perfectly increasing data has a Pearson correlation of 1.
    #[test]
    fn perfect_linear_is_one() -> Result<()> {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [2.0, 4.0, 6.0, 8.0];
        let r = pearson(&x, &y, Alternative::TwoSided)?;
        assert!((r.statistic - 1.0).abs() < 1e-12, "r was {}", r.statistic);
        Ok(())
    }

    /// Constant input is a typed error for Pearson.
    #[test]
    fn constant_input_is_degenerate() {
        let x = [1.0, 1.0, 1.0, 1.0];
        let y = [1.0, 2.0, 3.0, 4.0];
        assert!(matches!(
            pearson(&x, &y, Alternative::TwoSided),
            Err(Error::DegenerateInput(_))
        ));
    }
}
