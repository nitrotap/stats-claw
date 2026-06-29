//! Homogeneity-of-variance tests: Levene (mean-centered) and Bartlett.

use super::{len_f64, mean, one_way_anova, variance};
use crate::error::{Error, Result};
use crate::tests_stat::{TestResult, chi_squared_upper_log_tail, chi_squared_upper_tail};

/// Levene's test for equal variances using the mean-centered transform, matching
/// `scipy.stats.levene(*groups, center="mean")`.
///
/// Replaces each observation by its absolute deviation from its group mean, then
/// runs a one-way ANOVA on those deviations; the F-statistic and its framework
/// F-distribution p-value are the result. Levene is robust to non-normality.
///
/// # Arguments
///
/// * `groups` — two or more samples, each with at least one observation.
///
/// # Returns
///
/// A [`TestResult`] with the Levene F-statistic, p-value, and `df = k − 1`
/// (between-group); the within-group df is `N − k`.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] for fewer than two groups and
/// [`Error::DegenerateInput`] when every deviation is zero (all-constant data).
///
/// # Examples
///
/// ```
/// use stats_claw::tests_stat::parametric::levene;
///
/// // Equal-spread groups produce a near-zero Levene F.
/// let a = [1.0_f64, 2.0, 3.0, 4.0];
/// let b = [5.0_f64, 6.0, 7.0, 8.0];
/// let r = levene(&[&a[..], &b[..]])?;
/// assert!(r.statistic.abs() < 1e-9, "Levene F was {}", r.statistic);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn levene(groups: &[&[f64]]) -> Result<TestResult> {
    if groups.len() < 2 {
        return Err(Error::InsufficientData);
    }
    let deviations: Vec<Vec<f64>> = groups
        .iter()
        .map(|g| {
            let gm = mean(g);
            g.iter().map(|&x| (x - gm).abs()).collect()
        })
        .collect();
    let views: Vec<&[f64]> = deviations.iter().map(Vec::as_slice).collect();
    let anova = one_way_anova(&views)?;
    // Levene reports no effect size; keep the F statistic, p-value, and df.
    Ok(TestResult {
        statistic: anova.statistic,
        p_value: anova.p_value,
        log_p_value: anova.log_p_value,
        df: anova.df,
        effect_size: None,
    })
}

/// Bartlett's test for equal variances, matching `scipy.stats.bartlett`.
///
/// Forms the corrected log-likelihood-ratio statistic, which is chi-squared with
/// `k − 1` degrees of freedom under normality; the p-value is the framework
/// chi-squared upper tail. Bartlett is sensitive to departures from normality.
///
/// # Arguments
///
/// * `groups` — two or more samples, each with at least two observations.
///
/// # Returns
///
/// A [`TestResult`] with the Bartlett statistic, p-value, and `df = k − 1`.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] for fewer than two groups or any group
/// under two observations, and [`Error::DegenerateInput`] when a group has zero
/// variance (the log is undefined).
pub fn bartlett(groups: &[&[f64]]) -> Result<TestResult> {
    let k = groups.len();
    if k < 2 || groups.iter().any(|g| g.len() < 2) {
        return Err(Error::InsufficientData);
    }
    let mut total_df = 0.0;
    let mut pooled_num = 0.0;
    let mut sum_log = 0.0;
    let mut sum_inv_df = 0.0;
    for group in groups {
        let var = variance(group);
        if var <= 0.0 {
            return Err(Error::DegenerateInput("zero-variance group".to_owned()));
        }
        let df = len_f64(group.len() - 1);
        total_df += df;
        pooled_num += df * var;
        sum_log += df * var.ln();
        sum_inv_df += 1.0 / df;
    }
    let pooled_var = pooled_num / total_df;
    let numerator = total_df * pooled_var.ln() - sum_log;
    let k_f = len_f64(k);
    let correction = 1.0 + (sum_inv_df - 1.0 / total_df) / (3.0 * (k_f - 1.0));
    let statistic = numerator / correction;
    let df_int = i64::try_from(k - 1).unwrap_or(i64::MAX);
    let p_value = chi_squared_upper_tail(statistic, df_int);
    Ok(TestResult {
        statistic,
        p_value,
        log_p_value: Some(chi_squared_upper_log_tail(statistic, df_int)),
        df: Some(k_f - 1.0),
        effect_size: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Levene on two identical-spread groups yields a near-zero statistic.
    #[test]
    fn levene_zero_for_equal_spread() -> Result<()> {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let r = levene(&[&a[..], &b[..]])?;
        assert!(r.statistic.abs() < 1e-9, "Levene F was {}", r.statistic);
        Ok(())
    }

    /// Bartlett rejects a zero-variance group rather than taking ln(0).
    #[test]
    fn bartlett_rejects_zero_variance() {
        let a = [2.0, 2.0, 2.0];
        let b = [1.0, 2.0, 3.0];
        assert!(matches!(
            bartlett(&[&a[..], &b[..]]),
            Err(Error::DegenerateInput(_))
        ));
    }
}
