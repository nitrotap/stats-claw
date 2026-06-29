//! Bayesian interval estimation: the Beta-posterior credible interval.

use crate::distributions::BetaDistribution;
use crate::distributions::Quantile;
use crate::error::{Error, Result};

/// Equal-tailed credible interval of a Beta-binomial posterior.
///
/// Updates a `Beta(α₀, β₀)` prior with `successes` out of `trials` Bernoulli
/// observations to the posterior `Beta(α₀ + successes, β₀ + trials − successes)`,
/// then returns its `alpha/2` and `1 − alpha/2` quantiles, matching
/// `scipy.stats.beta.ppf`.
///
/// # Arguments
///
/// * `prior_alpha` — prior `α₀ > 0`.
/// * `prior_beta` — prior `β₀ > 0`.
/// * `successes` — observed successes `k ≥ 0`.
/// * `trials` — observed trials `n ≥ k`.
/// * `alpha` — total tail mass (e.g. `0.05` for a 95% interval).
///
/// # Returns
///
/// `(lower, upper)` posterior quantiles, both in `[0, 1]`.
///
/// # Errors
///
/// Returns [`Error::InvalidInput`] for a non-positive prior parameter, a negative
/// success count, or `successes > trials`.
///
/// # Examples
///
/// ```
/// use stats_claw::resampling::beta_credible_interval;
///
/// // Uniform prior, 7 of 10 successes → 95% credible interval inside (0, 1).
/// let (lo, hi) = beta_credible_interval(1.0, 1.0, 7.0, 10.0, 0.05)?;
/// assert!(0.0 < lo && lo < hi && hi < 1.0);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn beta_credible_interval(
    prior_alpha: f64,
    prior_beta: f64,
    successes: f64,
    trials: f64,
    alpha: f64,
) -> Result<(f64, f64)> {
    if prior_alpha <= 0.0 || prior_beta <= 0.0 {
        return Err(Error::InvalidInput(
            "prior parameters must be positive".to_owned(),
        ));
    }
    if successes < 0.0 || trials < successes {
        return Err(Error::InvalidInput(
            "need 0 <= successes <= trials".to_owned(),
        ));
    }
    let posterior = BetaDistribution {
        alpha_parameter: prior_alpha + successes,
        beta_parameter: prior_beta + (trials - successes),
        support_lower_bound: 0.0,
        support_upper_bound: 1.0,
        ..Default::default()
    };
    let lower = posterior.quantile(alpha / 2.0);
    let upper = posterior.quantile(1.0 - alpha / 2.0);
    Ok((lower, upper))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A uniform prior with all-success data pushes the interval toward 1.
    #[test]
    fn all_success_interval_is_high() -> Result<()> {
        let (lo, hi) = beta_credible_interval(1.0, 1.0, 10.0, 10.0, 0.05)?;
        assert!(lo > 0.5 && hi <= 1.0, "interval was ({lo}, {hi})");
        Ok(())
    }

    /// An improper prior is rejected.
    #[test]
    fn non_positive_prior_is_invalid() {
        assert!(matches!(
            beta_credible_interval(0.0, 1.0, 1.0, 2.0, 0.05),
            Err(Error::InvalidInput(_))
        ));
    }
}
