//! Cochran's Q test for K paired binary treatments.

use crate::error::{Error, Result};
use crate::tests_stat::parametric::len_f64;
use crate::tests_stat::{chi_squared_upper_log_tail, chi_squared_upper_tail, TestResult};

/// Cochran's Q test across `rows`, matching
/// `statsmodels.stats.contingency_tables.cochrans_q`.
///
/// `rows` is subject-major: each slice is one subject's binary (`0`/`1`) response
/// across the `k` treatments, and all subjects share the treatment count. Forms
/// `Q = (k − 1)·(k·Σ Gⱼ² − (Σ Gⱼ)²) / (k·Σ Lᵢ − Σ Lᵢ²)`, where `Gⱼ` are the
/// treatment column sums and `Lᵢ` the subject row sums; `Q` is chi-squared with
/// `k − 1` degrees of freedom.
///
/// # Arguments
///
/// * `rows` — at least two subjects, each a slice of `k ≥ 2` binary responses.
///
/// # Returns
///
/// A [`TestResult`] with the Q statistic, p-value, `df = k − 1`, no effect size.
/// When every subject responds identically across treatments the denominator is
/// zero, so `Q = 0` and the p-value is 1.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] for fewer than two subjects or fewer than
/// two treatments, and [`Error::InvalidInput`] when subjects differ in treatment
/// count.
///
/// # Examples
///
/// ```
/// use stats_claw::tests_stat::categorical::cochrans_q;
///
/// // All-identical responses give Q = 0 and p = 1.
/// let r1 = [1.0_f64, 1.0, 1.0];
/// let r2 = [0.0_f64, 0.0, 0.0];
/// let r = cochrans_q(&[&r1[..], &r2[..]])?;
/// assert!(r.statistic.abs() < 1e-12, "Q was {}", r.statistic);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn cochrans_q(rows: &[&[f64]]) -> Result<TestResult> {
    let n = rows.len();
    if n < 2 {
        return Err(Error::InsufficientData);
    }
    let k = rows.first().map_or(0, |r| r.len());
    if k < 2 {
        return Err(Error::InsufficientData);
    }
    if rows.iter().any(|r| r.len() != k) {
        return Err(Error::InvalidInput(
            "subjects differ in treatment count".to_owned(),
        ));
    }

    let mut col_sums = vec![0.0; k];
    let mut sum_row_sq = 0.0;
    let mut grand = 0.0;
    for row in rows {
        let row_sum: f64 = row.iter().sum();
        sum_row_sq = row_sum.mul_add(row_sum, sum_row_sq);
        grand += row_sum;
        for (j, &v) in row.iter().enumerate() {
            if let Some(slot) = col_sums.get_mut(j) {
                *slot += v;
            }
        }
    }
    let sum_col_sq: f64 = col_sums.iter().map(|&g| g * g).sum();
    let k_f = len_f64(k);

    let denom = k_f.mul_add(grand, -sum_row_sq);
    let statistic = if denom > 0.0 {
        (k_f - 1.0) * k_f.mul_add(sum_col_sq, -(grand * grand)) / denom
    } else {
        0.0
    };

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

    /// Fewer than two subjects is insufficient.
    #[test]
    fn single_subject_is_insufficient() {
        let row = [1.0, 0.0, 1.0];
        assert!(matches!(
            cochrans_q(&[&row[..]]),
            Err(Error::InsufficientData)
        ));
    }

    /// All-identical responses give Q = 0 and p = 1.
    #[test]
    fn identical_responses_are_zero_q() -> Result<()> {
        let r1 = [1.0, 1.0, 1.0];
        let r2 = [0.0, 0.0, 0.0];
        let r = cochrans_q(&[&r1[..], &r2[..]])?;
        assert!(r.statistic.abs() < 1e-12, "Q was {}", r.statistic);
        Ok(())
    }
}
