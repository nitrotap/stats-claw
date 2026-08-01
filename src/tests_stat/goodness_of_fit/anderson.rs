//! Anderson–Darling normality test (A² statistic and critical values).

use crate::distributions::Cdf;
use crate::distributions::NormalDistribution;
use crate::error::{Error, Result};
use crate::tests_stat::TestResult;
use crate::tests_stat::parametric::{len_f64, mean, variance};

/// Base critical values for the normal case (scipy's `_Avals_norm`), before the
/// finite-sample adjustment.
const AVALS_NORM: [f64; 5] = [0.561, 0.631, 0.752, 0.873, 1.035];

/// Anderson–Darling test for normality, matching `scipy.stats.anderson(x,
/// dist="norm")`.
///
/// Standardizes the sample by its mean and (Bessel-corrected) standard deviation,
/// then forms the A² statistic from the standard-normal CDF. The statistic is
/// reported as-is (scipy does not adjust it); the critical-value table is exposed
/// separately by [`anderson_normal_critical_values`].
///
/// # Arguments
///
/// * `sample` — the observations (at least two, with positive variance).
///
/// # Returns
///
/// A [`TestResult`] with `statistic = A²`, no p-value contract (set to `NaN` —
/// Anderson–Darling reports critical values, not a p-value), no df, no effect
/// size.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] for fewer than two observations and
/// [`Error::DegenerateInput`] for a zero-variance sample.
///
/// # Examples
///
/// ```
/// use stats_claw::tests_stat::goodness_of_fit::anderson_darling;
///
/// // The A² statistic is positive for a spread sample.
/// let sample = [-1.0_f64, -0.5, 0.0, 0.5, 1.0, 1.5, -0.3, 0.8, -1.2, 0.4, 0.2, -0.7];
/// let r = anderson_darling(&sample)?;
/// assert!(r.statistic > 0.0, "A² was {}", r.statistic);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn anderson_darling(sample: &[f64]) -> Result<TestResult> {
    let n = sample.len();
    if n < 2 {
        return Err(Error::InsufficientData);
    }
    let var = variance(sample);
    if var <= 0.0 {
        return Err(Error::DegenerateInput("zero-variance sample".to_owned()));
    }
    let mu = mean(sample);
    let sd = var.sqrt();
    let standard = NormalDistribution {
        mean: 0.0,
        standard_deviation: 1.0,
        ..Default::default()
    };

    let mut sorted = sample.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n_f = len_f64(n);
    let mut acc = 0.0;
    for i in 0..n {
        let zi = (sorted.get(i).copied().unwrap_or(f64::NAN) - mu) / sd;
        let z_tail = (sorted.get(n - 1 - i).copied().unwrap_or(f64::NAN) - mu) / sd;
        let ln_cdf = standard.cdf(zi).ln();
        let ln_sf = (1.0 - standard.cdf(z_tail)).ln();
        let weight = len_f64(2 * i + 1) / n_f;
        acc += weight * (ln_cdf + ln_sf);
    }
    let statistic = -n_f - acc;

    Ok(TestResult {
        statistic,
        p_value: f64::NAN,
        log_p_value: None,
        df: None,
        effect_size: None,
    })
}

/// Returns the five normal-case critical values for sample size `n`, matching
/// `scipy.stats.anderson(...).critical_values` (the 15/10/5/2.5/1% levels).
///
/// Each base value is divided by the finite-sample factor
/// `1 + 0.75/n + 2.25/n²` and rounded to three decimals, exactly as scipy does.
///
/// # Arguments
///
/// * `n` — the sample size (`≥ 1`).
///
/// # Returns
///
/// The five critical values in ascending significance (15% first, 1% last).
#[must_use]
pub fn anderson_normal_critical_values(n: usize) -> Vec<f64> {
    let n_f = len_f64(n.max(1));
    let factor = 2.25f64.mul_add(1.0 / (n_f * n_f), 0.75f64.mul_add(1.0 / n_f, 1.0));
    AVALS_NORM
        .iter()
        .map(|&base| round3(base / factor))
        .collect()
}

/// Rounds a value to three decimal places (scipy's `around(x, 3)`).
fn round3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// Kani formal-verification harness for the Anderson–Darling input-validation layer.
///
/// Compiled only under `cargo kani`. A single observation trips the leading `n < 2`
/// guard, so [`anderson_darling`] returns `Err` before the sorted-CDF and
/// significance interior; the symbolic value is never inspected.
#[cfg(kani)]
mod verification {
    use super::anderson_darling;
    use crate::error::Error;

    /// Proves the Anderson–Darling test rejects fewer than two observations via
    /// `Err` without panicking, for an arbitrary symbolic value.
    #[kani::proof]
    // Live path returns Err before any loop; the unwind bound caps the dead
    // transcendental-tail branches CBMC unwinds during model construction.
    #[kani::unwind(2)]
    fn anderson_darling_rejects_insufficient() {
        let one: [f64; 1] = [kani::any()];
        assert!(
            matches!(anderson_darling(&one), Err(Error::InsufficientData)),
            "under two observations must reject with InsufficientData"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A zero-variance sample is rejected before taking its log.
    #[test]
    fn zero_variance_is_degenerate() {
        let sample = [3.0, 3.0, 3.0];
        assert!(matches!(
            anderson_darling(&sample),
            Err(Error::DegenerateInput(_))
        ));
    }

    /// The critical values are positive and ascending.
    #[test]
    fn critical_values_are_ascending() {
        let cv = anderson_normal_critical_values(30);
        assert_eq!(
            cv.len(),
            5,
            "expected five critical values, got {}",
            cv.len()
        );
        for pair in cv.windows(2) {
            if let [lo, hi] = pair {
                assert!(lo < hi, "critical values must ascend: {lo} !< {hi}");
            }
        }
    }
}
