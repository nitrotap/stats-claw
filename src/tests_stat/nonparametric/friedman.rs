//! Friedman test for repeated measures (rank-based, with Kendall's W effect size).

use crate::error::{Error, Result};
use crate::tests_stat::parametric::len_f64;
use crate::tests_stat::ranks::{mid_ranks, tie_correction};
use crate::tests_stat::{chi_squared_upper_log_tail, chi_squared_upper_tail, TestResult};

/// Friedman test across `treatments`, matching `scipy.stats.friedmanchisquare`.
///
/// `treatments` is treatment-major: each slice holds one treatment's measurement
/// for every subject, and all slices share the subject count `n`. Within each
/// subject the `k` treatment values are mid-ranked, the ranks are summed per
/// treatment, and the chi-squared-distributed statistic on `k − 1` degrees of
/// freedom is formed; the p-value is the framework chi-squared upper tail. The
/// effect size is Kendall's W (`= statistic / (n·(k−1))`).
///
/// # Arguments
///
/// * `treatments` — at least three treatment slices of equal, non-zero length.
///
/// # Returns
///
/// A [`TestResult`] with the Friedman statistic, p-value, `df = k − 1`, and
/// `effect_size = Some(W)`.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] for fewer than three treatments or empty
/// subjects, and [`Error::InvalidInput`] when the treatment slices differ in
/// length.
///
/// # Examples
///
/// ```
/// use stats_claw::tests_stat::nonparametric::friedman;
///
/// // Consistent ranking across subjects yields Kendall's W ≈ 1.
/// let t1 = [1.0_f64, 1.0, 1.0];
/// let t2 = [2.0_f64, 2.0, 2.0];
/// let t3 = [3.0_f64, 3.0, 3.0];
/// let r = friedman(&[&t1[..], &t2[..], &t3[..]])?;
/// assert!(r.effect_size.is_some_and(|w| (w - 1.0).abs() < 1e-9),
///     "W was {:?}", r.effect_size);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn friedman(treatments: &[&[f64]]) -> Result<TestResult> {
    let k = treatments.len();
    if k < 3 {
        return Err(Error::InsufficientData);
    }
    let n = treatments.first().map_or(0, |t| t.len());
    if n == 0 {
        return Err(Error::InsufficientData);
    }
    if treatments.iter().any(|t| t.len() != n) {
        return Err(Error::InvalidInput(
            "treatments differ in subject count".to_owned(),
        ));
    }

    let mut rank_sums = vec![0.0; k];
    let mut tie_sum = 0.0;
    for subject in 0..n {
        let row: Vec<f64> = treatments
            .iter()
            .map(|t| t.get(subject).copied().unwrap_or(f64::NAN))
            .collect();
        let ranks = mid_ranks(&row);
        tie_sum += tie_correction(&row);
        for (j, r) in ranks.iter().enumerate() {
            if let Some(slot) = rank_sums.get_mut(j) {
                *slot += r;
            }
        }
    }

    let n_f = len_f64(n);
    let k_f = len_f64(k);
    let sum_sq: f64 = rank_sums.iter().map(|&r| r * r).sum();
    let uncorrected =
        (12.0 / (n_f * k_f * (k_f + 1.0))).mul_add(sum_sq, -(3.0 * n_f * (k_f + 1.0)));
    // scipy's within-row tie correction: divide by 1 − Σ(t³−t) / (n·(k³−k)).
    let correction = 1.0 - tie_sum / (n_f * (k_f * k_f).mul_add(k_f, -k_f));
    let statistic = if correction > 0.0 {
        uncorrected / correction
    } else {
        uncorrected
    };

    let df_int = i64::try_from(k - 1).unwrap_or(i64::MAX);
    let p_value = chi_squared_upper_tail(statistic, df_int);
    let kendalls_w = statistic / (n_f * (k_f - 1.0));

    Ok(TestResult {
        statistic,
        p_value,
        log_p_value: Some(chi_squared_upper_log_tail(statistic, df_int)),
        df: Some(k_f - 1.0),
        effect_size: Some(kendalls_w),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fewer than three treatments is insufficient for Friedman.
    #[test]
    fn two_treatments_are_insufficient() {
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        assert!(matches!(
            friedman(&[&a[..], &b[..]]),
            Err(Error::InsufficientData)
        ));
    }

    /// Consistent ranking across subjects yields W ≈ 1.
    #[test]
    fn consistent_ranking_has_high_w() -> Result<()> {
        let t1 = [1.0, 1.0, 1.0];
        let t2 = [2.0, 2.0, 2.0];
        let t3 = [3.0, 3.0, 3.0];
        let r = friedman(&[&t1[..], &t2[..], &t3[..]])?;
        assert!(
            r.effect_size.is_some_and(|w| (w - 1.0).abs() < 1e-9),
            "W was {:?}",
            r.effect_size
        );
        Ok(())
    }
}
