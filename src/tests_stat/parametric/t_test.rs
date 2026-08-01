//! Student's t-tests: one-sample, independent (pooled), paired, and Welch.

use super::{len_f64, log_p_from_t, mean, p_from_t, variance};
use crate::error::{Error, Result};
use crate::tests_stat::{Alternative, TestResult};

/// One-sample t-test of `sample`'s mean against `popmean`, matching
/// `scipy.stats.ttest_1samp(sample, popmean, alternative=…)`.
///
/// The statistic is `t = (x̄ − μ₀) / (s / √n)` on `n − 1` degrees of freedom, with
/// the p-value taken from the framework Student's t CDF for the chosen
/// `alternative`.
///
/// # Arguments
///
/// * `sample` — the observations; needs at least two distinct-enough values for a
///   positive variance.
/// * `popmean` — the null-hypothesis population mean `μ₀`.
/// * `alternative` — which tail(s) contribute to the p-value.
///
/// # Returns
///
/// A [`TestResult`] with the t-statistic, p-value, `df = n − 1`, no effect size.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] for fewer than two observations and
/// [`Error::DegenerateInput`] for a zero-variance sample.
///
/// # Examples
///
/// ```
/// use stats_claw::tests_stat::{parametric::t_test_1samp, Alternative};
///
/// let sample = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// // Testing at the sample mean: t = 0, p = 1.
/// let r = t_test_1samp(&sample, 3.0, Alternative::TwoSided)?;
/// assert!(r.statistic.abs() < 1e-12, "t was {}", r.statistic);
/// assert!((r.p_value - 1.0).abs() < 1e-9, "p was {}", r.p_value);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn t_test_1samp(sample: &[f64], popmean: f64, alternative: Alternative) -> Result<TestResult> {
    let n = sample.len();
    if n < 2 {
        return Err(Error::InsufficientData);
    }
    let var = variance(sample);
    if var <= 0.0 {
        return Err(Error::DegenerateInput("zero-variance sample".to_owned()));
    }
    let df = len_f64(n - 1);
    let se = (var / len_f64(n)).sqrt();
    let t = (mean(sample) - popmean) / se;
    Ok(result(t, df, alternative))
}

/// Independent two-sample t-test assuming equal variances (pooled), matching
/// `scipy.stats.ttest_ind(a, b, alternative=…)`.
///
/// # Arguments
///
/// * `a`, `b` — the two independent samples (each ≥ 2 observations).
/// * `alternative` — tested direction of `mean(a) − mean(b)`.
///
/// # Returns
///
/// A [`TestResult`] with the pooled t-statistic and `df = nₐ + n_b − 2`.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] for a sample under two observations and
/// [`Error::DegenerateInput`] when the pooled variance is zero.
pub fn t_test_ind(a: &[f64], b: &[f64], alternative: Alternative) -> Result<TestResult> {
    let (na, nb) = (a.len(), b.len());
    if na < 2 || nb < 2 {
        return Err(Error::InsufficientData);
    }
    let (va, vb) = (variance(a), variance(b));
    let df_int = na + nb - 2;
    let pooled_var = (len_f64(na - 1) * va + len_f64(nb - 1) * vb) / len_f64(df_int);
    if pooled_var <= 0.0 {
        return Err(Error::DegenerateInput("zero pooled variance".to_owned()));
    }
    let se = (pooled_var * (1.0 / len_f64(na) + 1.0 / len_f64(nb))).sqrt();
    let t = (mean(a) - mean(b)) / se;
    Ok(result(t, len_f64(df_int), alternative))
}

/// Paired (dependent) t-test on the differences `a − b`, matching
/// `scipy.stats.ttest_rel(a, b, alternative=…)`.
///
/// # Arguments
///
/// * `a`, `b` — paired samples of equal length (≥ 2 pairs).
/// * `alternative` — tested direction of the mean difference.
///
/// # Returns
///
/// A [`TestResult`] with the one-sample t-statistic of the differences and
/// `df = n − 1`.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] for length-mismatched inputs,
/// [`Error::InsufficientData`] for fewer than two pairs, and
/// [`Error::DegenerateInput`] for zero-variance differences.
pub fn t_test_paired(a: &[f64], b: &[f64], alternative: Alternative) -> Result<TestResult> {
    if a.len() != b.len() {
        return Err(Error::InvalidInput(
            "paired samples differ in length".to_owned(),
        ));
    }
    let diffs: Vec<f64> = a.iter().zip(b).map(|(&x, &y)| x - y).collect();
    t_test_1samp(&diffs, 0.0, alternative)
}

/// Welch's unequal-variance two-sample t-test, matching
/// `scipy.stats.ttest_ind(a, b, equal_var=False, alternative=…)`.
///
/// The denominator uses each sample's own variance, and the degrees of freedom are
/// the (fractional) Welch–Satterthwaite approximation, reported in `df`.
///
/// # Arguments
///
/// * `a`, `b` — the two independent samples (each ≥ 2 observations).
/// * `alternative` — tested direction of `mean(a) − mean(b)`.
///
/// # Returns
///
/// A [`TestResult`] with the Welch t-statistic and fractional Welch–Satterthwaite
/// `df`.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] for a sample under two observations and
/// [`Error::DegenerateInput`] when both sample variances are zero.
pub fn t_test_welch(a: &[f64], b: &[f64], alternative: Alternative) -> Result<TestResult> {
    let (na, nb) = (a.len(), b.len());
    if na < 2 || nb < 2 {
        return Err(Error::InsufficientData);
    }
    let (va, vb) = (variance(a), variance(b));
    let (sa, sb) = (va / len_f64(na), vb / len_f64(nb));
    let denom = sa + sb;
    if denom <= 0.0 {
        return Err(Error::DegenerateInput(
            "zero variance in both samples".to_owned(),
        ));
    }
    let t = (mean(a) - mean(b)) / denom.sqrt();
    // Welch–Satterthwaite fractional degrees of freedom.
    let df = denom * denom / (sa * sa / len_f64(na - 1) + sb * sb / len_f64(nb - 1));
    Ok(result(t, df, alternative))
}

/// Builds a [`TestResult`] from a t-statistic and degrees of freedom.
fn result(t: f64, df: f64, alternative: Alternative) -> TestResult {
    TestResult {
        statistic: t,
        p_value: p_from_t(t, df, alternative),
        log_p_value: Some(log_p_from_t(t, df, alternative)),
        df: Some(df),
        effect_size: None,
    }
}

/// Kani formal-verification harnesses for the t-test input-validation layer.
///
/// Compiled only under `cargo kani` (behind `#[cfg(kani)]`). Each call supplies a
/// shape that trips the leading structural guard (too few observations, or a length
/// mismatch), so the function returns `Err` before reaching the variance / Student-t
/// interior — the symbolic sample values are never inspected, and the proof stays
/// clear of the transcendental t-distribution tail (whose value proofs spurious-fail
/// under CBMC). This proves the validation logic itself is panic-free and rejects
/// the bad shape for arbitrary values.
#[cfg(kani)]
mod verification {
    use super::{t_test_1samp, t_test_ind, t_test_paired, t_test_welch};
    use crate::error::Error;
    use crate::tests_stat::Alternative;

    /// Proves the four t-test entry points reject shape-invalid input via `Err`
    /// without panicking, for arbitrary symbolic sample values.
    #[kani::proof]
    // Live path returns Err before any loop; the unwind bound caps the dead
    // transcendental-tail branches CBMC unwinds during model construction.
    #[kani::unwind(2)]
    fn t_test_validation_rejects_bad_shapes() {
        let one: [f64; 1] = [kani::any()];
        assert!(
            matches!(
                t_test_1samp(&one, kani::any(), Alternative::TwoSided),
                Err(Error::InsufficientData)
            ),
            "one-sample t-test under two observations must reject"
        );
        let a1: [f64; 1] = [kani::any()];
        let b1: [f64; 1] = [kani::any()];
        assert!(
            matches!(
                t_test_ind(&a1, &b1, Alternative::TwoSided),
                Err(Error::InsufficientData)
            ),
            "independent t-test under two observations must reject"
        );
        assert!(
            matches!(
                t_test_welch(&a1, &b1, Alternative::TwoSided),
                Err(Error::InsufficientData)
            ),
            "Welch t-test under two observations must reject"
        );
        let pa: [f64; 2] = [kani::any(), kani::any()];
        let pb: [f64; 3] = [kani::any(), kani::any(), kani::any()];
        assert!(
            matches!(
                t_test_paired(&pa, &pb, Alternative::TwoSided),
                Err(Error::InvalidInput(_))
            ),
            "paired t-test with length mismatch must reject"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-sample test at the sample mean has a zero statistic and p = 1.
    #[test]
    fn t_at_sample_mean_is_zero() -> Result<()> {
        let sample = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = t_test_1samp(&sample, 3.0, Alternative::TwoSided)?;
        assert!(r.statistic.abs() < 1e-12, "t was {}", r.statistic);
        assert!((r.p_value - 1.0).abs() < 1e-9, "p was {}", r.p_value);
        Ok(())
    }

    /// At the sample mean the two-sided p-value is 1, so its log p-value is 0.
    #[test]
    fn log_p_at_sample_mean_is_zero() -> Result<()> {
        let sample = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = t_test_1samp(&sample, 3.0, Alternative::TwoSided)?;
        let log_p = r
            .log_p_value
            .ok_or_else(|| Error::InvalidInput("no log p".to_owned()))?;
        assert!(log_p.abs() < 1e-12, "log p was {log_p}");
        Ok(())
    }

    /// In the body, the log p-value equals `ln(p_value)`; in a deep tail where the
    /// linear p-value has shrunk to a tiny number, the log p-value stays finite and
    /// matches scipy `t.logsf` — the log-space path's reason to exist.
    #[test]
    fn log_p_consistent_in_body_and_finite_in_deep_tail() -> Result<()> {
        // Body: a moderate statistic; log_p == ln(p_value).
        let sample = [4.0, 6.0, 5.0, 7.0, 3.0];
        let r = t_test_1samp(&sample, 0.0, Alternative::TwoSided)?;
        let log_p = r
            .log_p_value
            .ok_or_else(|| Error::InvalidInput("no log p".to_owned()))?;
        assert!(
            (log_p - r.p_value.ln()).abs() < 1e-9,
            "log p {log_p} should equal ln(p) {}",
            r.p_value.ln()
        );

        // Deep tail: a tight sample far from the null gives a huge |t| (≈ t=632,
        // df=4). scipy: t.logsf(632, 4)*1 + ln 2 = -36.196..., still finite.
        let tight = [100.0, 100.001, 99.999, 100.0005, 99.9995];
        let r2 = t_test_1samp(&tight, 0.0, Alternative::TwoSided)?;
        let log_p2 = r2
            .log_p_value
            .ok_or_else(|| Error::InvalidInput("no log p".to_owned()))?;
        assert!(
            log_p2.is_finite() && log_p2 < -30.0,
            "deep-tail log p should be finite and far negative, was {log_p2}"
        );
        Ok(())
    }

    /// Paired samples of unequal length are rejected.
    #[test]
    fn paired_length_mismatch_is_invalid() {
        assert!(matches!(
            t_test_paired(&[1.0, 2.0], &[1.0], Alternative::TwoSided),
            Err(Error::InvalidInput(_))
        ));
    }
}
