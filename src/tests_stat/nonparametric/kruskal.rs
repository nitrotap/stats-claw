//! Kruskal–Wallis H test (tie-corrected rank ANOVA).

use crate::error::{Error, Result};
use crate::tests_stat::parametric::len_f64;
use crate::tests_stat::ranks::{mid_ranks, tie_correction};
use crate::tests_stat::{chi_squared_upper_log_tail, chi_squared_upper_tail, TestResult};

/// Kruskal–Wallis H test across `groups`, matching `scipy.stats.kruskal`.
///
/// Pools and mid-ranks all observations, sums the ranks within each group, and
/// forms the tie-corrected H statistic, which is chi-squared with `k − 1` degrees
/// of freedom; the p-value is the framework chi-squared upper tail.
///
/// # Arguments
///
/// * `groups` — two or more samples, each with at least one observation.
///
/// # Returns
///
/// A [`TestResult`] with the H statistic, p-value, `df = k − 1`, no effect size.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] for fewer than two non-empty groups and
/// [`Error::DegenerateInput`] when every value ties (the tie correction divides by
/// zero).
///
/// # Examples
///
/// ```
/// use stats_claw::tests_stat::nonparametric::kruskal_wallis;
///
/// // Well-separated groups produce H > 0 and df = k - 1.
/// let a = [1.0_f64, 2.0, 3.0];
/// let b = [10.0_f64, 11.0, 12.0];
/// let c = [20.0_f64, 21.0, 22.0];
/// let r = kruskal_wallis(&[&a[..], &b[..], &c[..]])?;
/// assert_eq!(r.df, Some(2.0));
/// assert!(r.p_value < 0.05, "p was {}", r.p_value);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn kruskal_wallis(groups: &[&[f64]]) -> Result<TestResult> {
    let k = groups.len();
    if k < 2 || groups.iter().any(|g| g.is_empty()) {
        return Err(Error::InsufficientData);
    }
    let mut pooled = Vec::new();
    for group in groups {
        pooled.extend_from_slice(group);
    }
    let ranks = mid_ranks(&pooled);
    let total = pooled.len();
    let n = len_f64(total);

    let mut sum_term = 0.0;
    let mut offset = 0;
    for group in groups {
        let size = group.len();
        let rank_sum: f64 = ranks.iter().skip(offset).take(size).sum();
        sum_term += rank_sum * rank_sum / len_f64(size);
        offset += size;
    }

    let h_uncorrected = (12.0 / (n * (n + 1.0))).mul_add(sum_term, -(3.0 * (n + 1.0)));
    let tie = tie_correction(&pooled);
    let correction = 1.0 - tie / (n * n).mul_add(n, -n);
    if correction <= 0.0 {
        return Err(Error::DegenerateInput("all values tied".to_owned()));
    }
    let h = h_uncorrected / correction;

    let df_int = i64::try_from(k - 1).unwrap_or(i64::MAX);
    let p_value = chi_squared_upper_tail(h, df_int);
    Ok(TestResult {
        statistic: h,
        p_value,
        log_p_value: Some(chi_squared_upper_log_tail(h, df_int)),
        df: Some(len_f64(k - 1)),
        effect_size: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Well-separated groups produce a large H and small p-value.
    #[test]
    fn separated_groups_are_significant() -> Result<()> {
        let a = [1.0, 2.0, 3.0];
        let b = [10.0, 11.0, 12.0];
        let c = [20.0, 21.0, 22.0];
        let r = kruskal_wallis(&[&a[..], &b[..], &c[..]])?;
        assert_eq!(r.df, Some(2.0), "df was {:?}", r.df);
        assert!(r.p_value < 0.05, "p was {}", r.p_value);
        Ok(())
    }
}
