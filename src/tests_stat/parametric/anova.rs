//! One-way analysis of variance (ANOVA).

use super::{f_upper_log_tail, f_upper_tail, len_f64, mean};
use crate::error::{Error, Result};
use crate::tests_stat::TestResult;

/// One-way ANOVA across `groups`, matching `scipy.stats.f_oneway`.
///
/// Partitions the total sum of squares into between-group and within-group
/// components, forms `F = MS_between / MS_within`, and takes the upper-tail
/// p-value from the framework F distribution. The reported effect size is
/// `η² = SS_between / SS_total`.
///
/// # Arguments
///
/// * `groups` — two or more samples; each must hold at least one observation and
///   the pooled data must have positive within-group variance.
///
/// # Returns
///
/// A [`TestResult`] with the F-statistic, p-value, `df = k − 1` (between-group),
/// and `effect_size = Some(η²)`. The within-group df is `N − k`.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] for fewer than two groups or a total below
/// `k + 1` observations, and [`Error::DegenerateInput`] when the within-group
/// variation is zero (constant data).
///
/// # Examples
///
/// ```
/// use stats_claw::tests_stat::parametric::one_way_anova;
///
/// let a = [1.0, 2.0, 3.0];
/// let b = [4.0, 5.0, 6.0];
/// let r = one_way_anova(&[&a[..], &b[..]])?;
/// assert_eq!(r.df, Some(1.0));
/// assert!(r.effect_size.is_some());
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn one_way_anova(groups: &[&[f64]]) -> Result<TestResult> {
    let k = groups.len();
    if k < 2 {
        return Err(Error::InsufficientData);
    }
    let total_n: usize = groups.iter().map(|g| g.len()).sum();
    if total_n < k + 1 || groups.iter().any(|g| g.is_empty()) {
        return Err(Error::InsufficientData);
    }

    let grand_mean = {
        let sum: f64 = groups.iter().flat_map(|g| g.iter()).sum();
        sum / len_f64(total_n)
    };

    let mut ss_between = 0.0;
    let mut ss_within = 0.0;
    for group in groups {
        let gm = mean(group);
        let diff = gm - grand_mean;
        ss_between = (len_f64(group.len()) * diff).mul_add(diff, ss_between);
        ss_within += group.iter().map(|&x| (x - gm) * (x - gm)).sum::<f64>();
    }
    let ss_total = ss_between + ss_within;

    let df_between = k - 1;
    let df_within = total_n - k;
    if ss_within <= 0.0 {
        return Err(Error::DegenerateInput(
            "zero within-group variation".to_owned(),
        ));
    }
    let ms_between = ss_between / len_f64(df_between);
    let ms_within = ss_within / len_f64(df_within);
    let f = ms_between / ms_within;
    let df_num = i64::try_from(df_between).unwrap_or(i64::MAX);
    let df_den = i64::try_from(df_within).unwrap_or(i64::MAX);
    let p_value = f_upper_tail(f, df_num, df_den);
    let eta_squared = if ss_total > 0.0 {
        ss_between / ss_total
    } else {
        0.0
    };

    Ok(TestResult {
        statistic: f,
        p_value,
        log_p_value: Some(f_upper_log_tail(f, df_num, df_den)),
        df: Some(len_f64(df_between)),
        effect_size: Some(eta_squared),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Identical groups produce a zero F-statistic and zero effect size.
    #[test]
    fn identical_groups_have_zero_f() -> Result<()> {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0, 3.0];
        let r = one_way_anova(&[&a[..], &b[..]])?;
        assert!(r.statistic.abs() < 1e-12, "F was {}", r.statistic);
        assert!(
            r.effect_size.is_some_and(|e| e.abs() < 1e-12),
            "eta^2 was {:?}",
            r.effect_size
        );
        Ok(())
    }

    /// A single group is insufficient for an F-test.
    #[test]
    fn single_group_is_insufficient() {
        let a = [1.0, 2.0];
        assert!(matches!(
            one_way_anova(&[&a[..]]),
            Err(Error::InsufficientData)
        ));
    }
}
