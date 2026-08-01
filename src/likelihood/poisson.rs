//! Poisson likelihood: the one-parameter count model for the
//! [`PoissonLikelihood`](crate::likelihood::PoissonLikelihood).
//!
//! The Poisson model has a single rate parameter `λ > 0` and describes counts
//! `x ∈ {0, 1, 2, …}`. Its log-likelihood of a sample is
//! `ℓ(λ) = Σᵢ (xᵢ·ln λ − λ − ln Γ(xᵢ + 1))`, and the maximum-likelihood estimate
//! is available in closed form as the sample mean `λ̂ = x̄`.
//!
//! [`PoissonLikelihood`](crate::likelihood::PoissonLikelihood) implements the
//! generic [`LogLikelihood`] trait (so it can
//! be fed to [`fit_mle`](crate::likelihood::fit_mle)) and additionally offers a
//! [`fit`](crate::likelihood::PoissonLikelihood::fit) that returns the exact
//! closed-form estimate.

use crate::algorithms::count_to_f64;
use crate::error::{Error, Result};
use crate::likelihood::{LogLikelihood, MleFit};
use crate::special::ln_gamma;

/// Reports whether `x` is a valid Poisson observation: a finite, non-negative
/// integer count expressed as an `f64`.
///
/// The integrality test compares `x` against its rounded value with a relational
/// (`<= 0.0`) operator rather than `==`, which keeps it clear of the crate's
/// `float_cmp` lint while still accepting only exact integers.
///
/// # Arguments
///
/// * `x` — a candidate observation.
///
/// # Returns
///
/// `true` iff `x` is finite, `x ≥ 0`, and `x` has no fractional part.
fn is_count(x: f64) -> bool {
    x.is_finite() && x >= 0.0 && (x - x.round()).abs() <= 0.0
}

impl crate::likelihood::PoissonLikelihood {
    /// Computes the closed-form maximum-likelihood fit `λ̂ = x̄` (the sample
    /// mean) of the Poisson rate.
    ///
    /// The estimate is exact, so the returned [`MleFit`] reports
    /// [`converged`](MleFit::converged) as `true` and zero
    /// [`iterations`](MleFit::iterations); its AIC/BIC follow the shared
    /// information-criteria formulas with `k = 1` parameter.
    ///
    /// # Arguments
    ///
    /// * `data` — the observed counts; each entry must be a finite, non-negative
    ///   integer, and the sample must contain at least one non-zero count.
    ///
    /// # Returns
    ///
    /// An [`MleFit`] whose single parameter is `λ̂` and whose
    /// [`log_likelihood`](MleFit::log_likelihood) is `ℓ(λ̂; data)`.
    ///
    /// # Errors
    ///
    /// * [`Error::InsufficientData`] if `data` is empty.
    /// * [`Error::InvalidInput`] if any observation is negative, non-integer, or
    ///   non-finite.
    /// * [`Error::DegenerateInput`] if every observation is zero, which drives
    ///   `λ̂ = 0` and leaves the log-likelihood undefined (`ln 0`).
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::likelihood::PoissonLikelihood;
    ///
    /// let model = PoissonLikelihood::default();
    /// let fit = model.fit(&[2.0, 3.0, 1.0, 5.0, 0.0, 4.0, 2.0, 3.0])?;
    /// // The MLE of the Poisson rate is the sample mean, here 20 / 8 = 2.5.
    /// assert!((fit.params()[0] - 2.5).abs() < 1e-12, "lambda_hat was {}", fit.params()[0]);
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    pub fn fit(&self, data: &[f64]) -> Result<MleFit> {
        if data.is_empty() {
            return Err(Error::InsufficientData);
        }
        if data.iter().any(|&x| !is_count(x)) {
            return Err(Error::InvalidInput(
                "Poisson data must be finite non-negative integer counts".to_owned(),
            ));
        }
        let sum: f64 = data.iter().sum();
        let lambda_hat = sum / count_to_f64(data.len());
        if lambda_hat <= 0.0 {
            return Err(Error::DegenerateInput(
                "all-zero counts drive lambda_hat to 0, where the log-likelihood is undefined"
                    .to_owned(),
            ));
        }
        let log_likelihood = self.log_likelihood(&[lambda_hat], data);
        Ok(MleFit::from_closed_form(
            vec![lambda_hat],
            log_likelihood,
            data.len(),
        ))
    }
}

impl LogLikelihood for crate::likelihood::PoissonLikelihood {
    /// Returns the single free parameter count of the Poisson model, `1` (the
    /// rate `λ`).
    fn n_params(&self) -> usize {
        1
    }

    /// Evaluates `ℓ([λ]; data) = Σᵢ (xᵢ·ln λ − λ − ln Γ(xᵢ + 1))`.
    ///
    /// Returns [`f64::NEG_INFINITY`] when `params[0] = λ ≤ 0` (or is `NaN`), and
    /// likewise when any observation is negative, non-integer, or non-finite —
    /// all of which lie outside the Poisson support.
    ///
    /// # Arguments
    ///
    /// * `params` — the one-element rate vector `[λ]`; a shorter slice yields
    ///   [`f64::NEG_INFINITY`].
    /// * `data` — the observed counts.
    ///
    /// # Returns
    ///
    /// The total log-likelihood, or [`f64::NEG_INFINITY`] outside the valid
    /// domain.
    fn log_likelihood(&self, params: &[f64], data: &[f64]) -> f64 {
        let Some(&lambda) = params.first() else {
            return f64::NEG_INFINITY;
        };
        if lambda.is_nan() || lambda <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if data.iter().any(|&x| !is_count(x)) {
            return f64::NEG_INFINITY;
        }
        let ln_lambda = lambda.ln();
        data.iter()
            .map(|&x| x.mul_add(ln_lambda, -lambda) - ln_gamma(x + 1.0))
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use crate::error::Error;
    use crate::likelihood::LogLikelihood;
    use crate::likelihood::PoissonLikelihood;

    #[test]
    fn fit_empty_data_is_insufficient() {
        let model = PoissonLikelihood::default();
        assert!(
            matches!(model.fit(&[]), Err(Error::InsufficientData)),
            "empty data must be rejected"
        );
    }

    #[test]
    fn fit_negative_count_is_invalid() {
        let model = PoissonLikelihood::default();
        assert!(
            matches!(model.fit(&[1.0, -2.0, 3.0]), Err(Error::InvalidInput(_))),
            "a negative count must be rejected"
        );
    }

    #[test]
    fn fit_non_integer_count_is_invalid() {
        let model = PoissonLikelihood::default();
        assert!(
            matches!(model.fit(&[1.0, 2.5, 3.0]), Err(Error::InvalidInput(_))),
            "a fractional count must be rejected"
        );
    }

    #[test]
    fn fit_all_zero_counts_is_degenerate() {
        let model = PoissonLikelihood::default();
        assert!(
            matches!(model.fit(&[0.0, 0.0, 0.0]), Err(Error::DegenerateInput(_))),
            "all-zero counts must be rejected"
        );
    }

    /// Sample fixture shared across the numeric tests (sum 20, n 8, mean 2.5).
    const DATA: [f64; 8] = [2.0, 3.0, 1.0, 5.0, 0.0, 4.0, 2.0, 3.0];

    #[test]
    fn log_likelihood_matches_scipy_at_lambda_3() {
        // python3:
        //   import numpy as np; from scipy.stats import poisson
        //   data=np.array([2,3,1,5,0,4,2,3])
        //   poisson.logpmf(data, 3.0).sum()  ->  -14.963113099343797
        let model = PoissonLikelihood::default();
        let got = model.log_likelihood(&[3.0], &DATA);
        let want = -14.963_113_099_343_797;
        assert!(
            (got - want).abs() <= 1e-10 * want.abs(),
            "logL(3.0) was {got}, want {want}"
        );
    }

    #[test]
    fn log_likelihood_matches_scipy_at_lambda_1() {
        // python3:
        //   poisson.logpmf(np.array([2,3,1,5,0,4,2,3]), 1.0).sum()
        //     ->  -20.93535887270599
        let model = PoissonLikelihood::default();
        let got = model.log_likelihood(&[1.0], &DATA);
        let want = -20.935_358_872_705_99;
        assert!(
            (got - want).abs() <= 1e-10 * want.abs(),
            "logL(1.0) was {got}, want {want}"
        );
    }

    #[test]
    fn log_likelihood_non_positive_lambda_is_neg_inf() {
        let model = PoissonLikelihood::default();
        assert!(
            model.log_likelihood(&[0.0], &DATA) == f64::NEG_INFINITY,
            "lambda = 0 must give -inf"
        );
        assert!(
            model.log_likelihood(&[-1.0], &DATA) == f64::NEG_INFINITY,
            "negative lambda must give -inf"
        );
    }

    #[test]
    fn log_likelihood_negative_observation_is_neg_inf() {
        let model = PoissonLikelihood::default();
        assert!(
            model.log_likelihood(&[2.5], &[1.0, -2.0, 3.0]) == f64::NEG_INFINITY,
            "a negative observation must give -inf"
        );
    }

    #[test]
    fn log_likelihood_non_finite_observation_is_neg_inf() {
        // Pins the D3 contract: a non-finite observation is rejected as −∞. The
        // `is_count` guard already handles this (NaN/±∞ are not finite counts);
        // this test locks the behavior against future refactors.
        let model = PoissonLikelihood::default();
        assert!(
            model.log_likelihood(&[2.5], &[1.0, f64::NAN, 3.0]) == f64::NEG_INFINITY,
            "a NaN observation must give -inf"
        );
        assert!(
            model.log_likelihood(&[2.5], &[1.0, f64::INFINITY, 3.0]) == f64::NEG_INFINITY,
            "a +inf observation must give -inf"
        );
    }

    #[test]
    fn fit_recovers_sample_mean_and_criteria() -> Result<(), Error> {
        // python3:
        //   lh = data.mean()  -> 2.5
        //   ll = poisson.logpmf(data, lh).sum()  -> -14.60954423522289
        //   aic = 2*1 - 2*ll  -> 31.21908847044578
        //   bic = 1*math.log(8) - 2*ll  -> 31.298530012125614
        let model = PoissonLikelihood::default();
        let fit = model.fit(&DATA)?;
        let lambda_hat = *fit.params().first().unwrap_or(&f64::NAN);
        assert!(
            (lambda_hat - 2.5).abs() <= 1e-12,
            "lambda_hat was {lambda_hat}"
        );
        let ll = -14.609_544_235_222_89;
        assert!(
            (fit.log_likelihood() - ll).abs() <= 1e-10 * ll.abs(),
            "logL was {}",
            fit.log_likelihood()
        );
        assert!(fit.converged(), "closed-form fit must report converged");
        assert_eq!(
            fit.iterations(),
            0,
            "closed-form fit performs no iterations"
        );
        assert!(
            (fit.aic() - 31.219_088_470_445_78).abs() <= 1e-10,
            "aic was {}",
            fit.aic()
        );
        assert!(
            (fit.bic() - 31.298_530_012_125_614).abs() <= 1e-10,
            "bic was {}",
            fit.bic()
        );
        Ok(())
    }

    #[test]
    fn fit_mle_from_perturbed_init_matches_closed_form() -> Result<(), Error> {
        let model = PoissonLikelihood::default();
        let closed = model.fit(&DATA)?;
        // Start the free optimizer away from the true rate; it must recover it.
        let numeric = crate::likelihood::fit_mle(&model, &DATA, &[1.0], 1e-10)?;
        let numeric_lambda = *numeric.params().first().unwrap_or(&f64::NAN);
        let closed_lambda = *closed.params().first().unwrap_or(&f64::NAN);
        assert!(
            (numeric_lambda - closed_lambda).abs() <= 1e-5,
            "numeric lambda_hat {numeric_lambda} vs closed {closed_lambda}"
        );
        Ok(())
    }
}
