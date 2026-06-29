//! Shapiro–Wilk normality test via Royston's AS R94 algorithm.

use crate::distributions::NormalDistribution;
use crate::distributions::{Cdf, Quantile};
use crate::error::{Error, Result};
use crate::tests_stat::parametric::{len_f64, mean};
use crate::tests_stat::TestResult;

/// Shapiro–Wilk test for normality, matching `scipy.stats.shapiro`.
///
/// Computes the `W` statistic from Royston's (1992) approximation to the
/// expected-order-statistic weights, then maps `W` to a p-value through Royston's
/// normalizing transform (valid for `12 ≤ n ≤ 5000`).
///
/// # Arguments
///
/// * `sample` — the observations (at least three, with positive spread).
///
/// # Returns
///
/// A [`TestResult`] with `statistic = W ∈ (0, 1]`, the p-value, no df, no effect
/// size.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] for fewer than three observations and
/// [`Error::DegenerateInput`] for a constant sample.
///
/// # Examples
///
/// ```
/// use stats_claw::tests_stat::goodness_of_fit::shapiro_wilk;
///
/// // W lies in (0, 1] for a normal-looking sample.
/// let sample = [-1.0_f64, -0.5, -0.2, 0.0, 0.1, 0.3, 0.6, 1.0, 1.2, -0.8, 0.4, 0.9];
/// let r = shapiro_wilk(&sample)?;
/// assert!(r.statistic > 0.0 && r.statistic <= 1.0, "W was {}", r.statistic);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn shapiro_wilk(sample: &[f64]) -> Result<TestResult> {
    let n = sample.len();
    if n < 3 {
        return Err(Error::InsufficientData);
    }
    let weights = royston_weights(n);
    let mut sorted = sample.to_vec();
    sorted.sort_by(f64::total_cmp);

    let mu = mean(&sorted);
    let ss: f64 = sorted.iter().map(|&x| (x - mu) * (x - mu)).sum();
    if ss <= 0.0 {
        return Err(Error::DegenerateInput("constant sample".to_owned()));
    }
    let numerator: f64 = weights.iter().zip(&sorted).map(|(&a, &x)| a * x).sum();
    let w = (numerator * numerator) / ss;

    let p_value = royston_p_value(w, n);
    Ok(TestResult {
        statistic: w,
        p_value,
        log_p_value: None,
        df: None,
        effect_size: None,
    })
}

/// Royston's approximate Shapiro–Wilk weights `aᵢ` for sample size `n`.
fn royston_weights(n: usize) -> Vec<f64> {
    let standard = NormalDistribution {
        mean: 0.0,
        standard_deviation: 1.0,
        ..Default::default()
    };
    let n_f = len_f64(n);
    // Blom's expected normal order statistics m_i.
    let m: Vec<f64> = (1..=n)
        .map(|i| standard.quantile((len_f64(i) - 0.375) / (n_f + 0.25)))
        .collect();
    let m_norm_sq: f64 = m.iter().map(|&v| v * v).sum();
    let m_norm = m_norm_sq.sqrt();
    let rsn = 1.0 / n_f.sqrt();

    let m_n = m.get(n - 1).copied().unwrap_or(0.0);
    // Correction for the largest weight (coefficients low-to-high in rsn).
    let a_n = poly(
        &[
            m_n / m_norm,
            0.221_157,
            -0.147_981,
            -2.071_190,
            4.434_685,
            -2.706_056,
        ],
        rsn,
    );

    let mut weights = vec![0.0; n];
    if n <= 5 {
        let phi = (2.0 * m_n).mul_add(-m_n, m_norm_sq) / (2.0 * a_n).mul_add(-a_n, 1.0);
        fill_weights(&mut weights, &m, phi, a_n, None);
    } else {
        let m_n1 = m.get(n - 2).copied().unwrap_or(0.0);
        let a_n1 = poly(
            &[
                m_n1 / m_norm,
                0.042_981,
                -0.293_762,
                -1.752_461,
                5.682_633,
                -3.582_633,
            ],
            rsn,
        );
        let phi = (2.0f64.mul_add(-(m_n1 * m_n1), 2.0f64.mul_add(-(m_n * m_n), m_norm_sq)))
            / (2.0f64.mul_add(-(a_n1 * a_n1), 2.0f64.mul_add(-(a_n * a_n), 1.0)));
        fill_weights(&mut weights, &m, phi, a_n, Some(a_n1));
    }
    weights
}

/// Fills the weight vector from the corrected extreme weights and the central
/// rescaling `phi`.
fn fill_weights(weights: &mut [f64], m: &[f64], phi: f64, a_n: f64, a_n1: Option<f64>) {
    let n = weights.len();
    let sqrt_phi = phi.sqrt();
    for i in 0..n {
        let value = if i == n - 1 {
            a_n
        } else if i == 0 {
            -a_n
        } else if a_n1.is_some() && i == n - 2 {
            a_n1.unwrap_or(0.0)
        } else if a_n1.is_some() && i == 1 {
            -a_n1.unwrap_or(0.0)
        } else {
            m.get(i).copied().unwrap_or(0.0) / sqrt_phi
        };
        if let Some(slot) = weights.get_mut(i) {
            *slot = value;
        }
    }
}

/// Maps `W` to a p-value via Royston's normalizing transform for `12 ≤ n ≤ 5000`.
fn royston_p_value(w: f64, n: usize) -> f64 {
    let n_f = len_f64(n);
    let ln_n = n_f.ln();
    let (mu, sigma) = if n <= 11 {
        let gamma = poly(&[-2.273, 0.459], n_f);
        let mu = poly(&[0.5440, -0.399_78, 0.025_054, -6.714e-4], n_f);
        let sigma = poly(&[1.3822, -0.776_39, 0.062_767, -0.002_032_2], n_f).exp();
        // Small-n transform: w1 = −ln(γ − ln(1 − W)).
        let w1 = -(gamma - (1.0 - w).ln()).ln();
        let z = (w1 - mu) / sigma;
        return survival(z);
    } else {
        let mu = poly(&[-1.5861, -0.310_82, -0.083_751, 0.003_891_5], ln_n);
        let sigma = poly(&[-0.4803, -0.082_676, 0.003_030_2], ln_n).exp();
        (mu, sigma)
    };
    let z = ((1.0 - w).ln() - mu) / sigma;
    survival(z)
}

/// Standard-normal survival `P(Z ≥ z)` via the framework normal CDF.
fn survival(z: f64) -> f64 {
    let standard = NormalDistribution {
        mean: 0.0,
        standard_deviation: 1.0,
        ..Default::default()
    };
    (1.0 - standard.cdf(z)).clamp(0.0, 1.0)
}

/// Evaluates a polynomial with coefficients given low-degree-first
/// (`c[0] + c[1]·x + c[2]·x² + …`) by Horner's method.
fn poly(coeffs: &[f64], x: f64) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, &c| acc.mul_add(x, c))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A constant sample is rejected.
    #[test]
    fn constant_sample_is_degenerate() {
        let sample = [2.0, 2.0, 2.0, 2.0];
        assert!(matches!(
            shapiro_wilk(&sample),
            Err(Error::DegenerateInput(_))
        ));
    }

    /// W lies in (0, 1] for a normal-looking sample.
    #[test]
    fn statistic_in_unit_interval() -> Result<()> {
        let sample = [
            -1.0, -0.5, -0.2, 0.0, 0.1, 0.3, 0.6, 1.0, 1.2, -0.8, 0.4, 0.9,
        ];
        let r = shapiro_wilk(&sample)?;
        assert!(
            r.statistic > 0.0 && r.statistic <= 1.0,
            "W was {}",
            r.statistic
        );
        Ok(())
    }
}
