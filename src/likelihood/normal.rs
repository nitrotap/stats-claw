//! Normal (Gaussian) maximum-likelihood numerics, for the
//! [`NormalLikelihood`].
//!
//! The two-parameter model `θ = [μ, σ]` (σ the standard deviation) has the
//! log-likelihood
//! `ℓ(μ, σ; x) = −n/2·ln(2π) − n·ln σ − Σ(xᵢ − μ)²/(2σ²)`,
//! matching `scipy.stats.norm.logpdf(x, μ, σ).sum()`. Its maximum-likelihood
//! estimate is closed form: `μ̂` is the sample mean and `σ̂` the *biased*
//! (population) standard deviation `√(Σ(xᵢ − μ̂)²/n)`, so [`NormalLikelihood::fit`]
//! returns an exact [`MleFit`] without invoking the numerical optimizer.
//!
//! # Examples
//!
//! ```
//! use stats_claw::likelihood::NormalLikelihood;
//!
//! let model = NormalLikelihood::default();
//! let fit = model.fit(&[2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])?;
//! // Closed-form MLE: sample mean 5, biased std 2.
//! assert!((fit.params()[0] - 5.0).abs() < 1e-12, "mu_hat was {}", fit.params()[0]);
//! assert!((fit.params()[1] - 2.0).abs() < 1e-12, "sigma_hat was {}", fit.params()[1]);
//! # Ok::<(), stats_claw::error::Error>(())
//! ```

use crate::algorithms::count_to_f64;
use crate::error::{Error, Result};
use crate::likelihood::NormalLikelihood;
use crate::likelihood::{LogLikelihood, MleFit};
use std::f64::consts::PI;

impl LogLikelihood for NormalLikelihood {
    /// Returns `2` — the model's free parameters are `μ` (mean) and `σ`
    /// (standard deviation), in that order.
    fn n_params(&self) -> usize {
        2
    }

    /// Evaluates the Gaussian log-likelihood
    /// `ℓ = −n/2·ln(2π) − n·ln σ − Σ(xᵢ − μ)²/(2σ²)` with `params = [μ, σ]`.
    ///
    /// Returns [`f64::NEG_INFINITY`] for any `θ` outside the valid domain: a
    /// non-positive or non-finite `σ`, or a non-finite `μ`. Per the
    /// [`LogLikelihood`] contract, any non-finite observation (`NaN` or `±∞`) also
    /// yields [`f64::NEG_INFINITY`] rather than letting `(xᵢ − μ)²` propagate a
    /// `NaN` (or an incidental `−∞` for `+∞`). An empty `data` yields `0.0` (the
    /// empty product's log-likelihood); callers wanting an error on empty input
    /// use [`NormalLikelihood::fit`].
    fn log_likelihood(&self, params: &[f64], data: &[f64]) -> f64 {
        let mu = *params.first().unwrap_or(&f64::NAN);
        let sigma = *params.get(1).unwrap_or(&f64::NAN);
        if !mu.is_finite() || !sigma.is_finite() || sigma <= 0.0 {
            return f64::NEG_INFINITY;
        }
        // A non-finite observation is outside the support: return −∞ explicitly
        // instead of propagating the NaN/±∞ that the squared-error sum would.
        if !data.iter().all(|x| x.is_finite()) {
            return f64::NEG_INFINITY;
        }
        let n = count_to_f64(data.len());
        let inv_var = 1.0 / (sigma * sigma);
        let sse: f64 = data.iter().map(|x| (x - mu) * (x - mu)).sum();
        // −n/2·ln(2π) − n·ln σ − sse/(2σ²), grouped to fused multiply-adds.
        let neg_half_n = -0.5 * n;
        let two_pi_term = neg_half_n * (2.0 * PI).ln();
        let sigma_term = (-n).mul_add(sigma.ln(), two_pi_term);
        (0.5 * sse).mul_add(-inv_var, sigma_term)
    }
}

impl NormalLikelihood {
    /// Closed-form maximum-likelihood fit of `θ = [μ, σ]` to `data`.
    ///
    /// The Gaussian MLE is analytic: `μ̂` is the sample mean and `σ̂` the
    /// *biased* (population) standard deviation `√(Σ(xᵢ − μ̂)²/n)` — matching
    /// `numpy.std(data, ddof=0)`. The result is exact, so the returned
    /// [`MleFit`] reports `converged() == true` and `iterations() == 0`.
    ///
    /// # Arguments
    ///
    /// * `data` — the observed sample; must contain at least two points with
    ///   non-zero spread.
    ///
    /// # Returns
    ///
    /// An [`MleFit`] whose `params()` are `[μ̂, σ̂]`, carrying the attained
    /// log-likelihood and the AIC/BIC (`k = 2`, `n = data.len()`).
    ///
    /// # Errors
    ///
    /// * [`Error::InsufficientData`] if `data` has fewer than two points (with a
    ///   single point `σ̂ = 0`, which the density cannot represent).
    /// * [`Error::DegenerateInput`] if every observation is identical, so the
    ///   estimated `σ̂` is zero and the log-likelihood is undefined.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::likelihood::NormalLikelihood;
    ///
    /// let fit = NormalLikelihood::default().fit(&[1.0, 2.0, 3.0])?;
    /// assert!((fit.params()[0] - 2.0).abs() < 1e-12, "mu_hat was {}", fit.params()[0]);
    /// assert!(fit.converged());
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    pub fn fit(&self, data: &[f64]) -> Result<MleFit> {
        if data.len() < 2 {
            return Err(Error::InsufficientData);
        }
        let n = count_to_f64(data.len());
        let mu_hat = data.iter().sum::<f64>() / n;
        let sse: f64 = data.iter().map(|x| (x - mu_hat) * (x - mu_hat)).sum();
        let sigma_hat = (sse / n).sqrt();
        if sigma_hat <= 0.0 {
            return Err(Error::DegenerateInput(
                "all observations are identical (zero variance)".to_owned(),
            ));
        }
        let params = vec![mu_hat, sigma_hat];
        let log_likelihood = self.log_likelihood(&params, data);
        Ok(MleFit::from_closed_form(params, log_likelihood, data.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::likelihood::fit_mle;

    /// Textbook sample: sample mean 5, biased MLE std 2 (Σ(x−5)² = 32, /8 = 4,
    /// √4 = 2). Reused by the fit and consistency tests.
    const DATA: [f64; 8] = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];

    /// Reads parameter `i` from a fit, mapping an out-of-range index to an error
    /// so tests use `?` rather than an index that could panic.
    fn param(fit: &MleFit, i: usize) -> Result<f64> {
        fit.params()
            .get(i)
            .copied()
            .ok_or_else(|| Error::InvalidInput(format!("missing parameter {i}")))
    }

    /// Returns whether `x` is exactly `−∞` without a lint-tripping float `==`.
    fn is_neg_inf(x: f64) -> bool {
        x.is_infinite() && x.is_sign_negative()
    }

    #[test]
    fn fit_rejects_bad_input() {
        let model = NormalLikelihood::default();
        assert!(
            matches!(model.fit(&[]), Err(Error::InsufficientData)),
            "empty data should be InsufficientData"
        );
        assert!(
            matches!(model.fit(&[3.0]), Err(Error::InsufficientData)),
            "single point should be InsufficientData"
        );
        assert!(
            matches!(model.fit(&[3.0, 3.0, 3.0]), Err(Error::DegenerateInput(_))),
            "zero variance should be DegenerateInput"
        );
    }

    #[test]
    fn log_likelihood_matches_scipy() {
        // python3:
        //   import numpy as np; from scipy import stats
        //   data = np.array([2.,4.,4.,4.,5.,5.,7.,9.])
        //   stats.norm.logpdf(data, 4.5, 2.3).sum()  # -17.228391835129557
        let model = NormalLikelihood::default();
        let got = model.log_likelihood(&[4.5, 2.3], &DATA);
        let want = -17.228_391_835_129_557;
        assert!(
            ((got - want) / want).abs() < 1e-10,
            "log_likelihood was {got}, want {want}"
        );
    }

    #[test]
    fn log_likelihood_out_of_domain_is_neg_inf() {
        let model = NormalLikelihood::default();
        assert!(
            is_neg_inf(model.log_likelihood(&[1.0, 0.0], &DATA)),
            "sigma = 0 should be NEG_INFINITY"
        );
        assert!(
            is_neg_inf(model.log_likelihood(&[1.0, -1.0], &DATA)),
            "negative sigma should be NEG_INFINITY"
        );
        assert!(
            is_neg_inf(model.log_likelihood(&[f64::NAN, 2.0], &DATA)),
            "non-finite mu should be NEG_INFINITY"
        );
        assert!(
            is_neg_inf(model.log_likelihood(&[1.0, f64::INFINITY], &DATA)),
            "non-finite sigma should be NEG_INFINITY"
        );
    }

    #[test]
    fn log_likelihood_non_finite_observation_is_neg_inf() {
        let model = NormalLikelihood::default();
        // A NaN observation makes the whole log-likelihood NEG_INFINITY, not the
        // NaN that (xᵢ − μ)² would otherwise propagate.
        assert!(
            is_neg_inf(model.log_likelihood(&[5.0, 2.0], &[1.0, f64::NAN, 3.0])),
            "NaN observation should give NEG_INFINITY, got {}",
            model.log_likelihood(&[5.0, 2.0], &[1.0, f64::NAN, 3.0])
        );
        // A +∞ observation likewise — pinned explicitly rather than left to the
        // sign that `−inv_var · Σ(xᵢ − μ)²` happens to produce.
        assert!(
            is_neg_inf(model.log_likelihood(&[5.0, 2.0], &[1.0, f64::INFINITY, 3.0])),
            "+inf observation should give NEG_INFINITY, got {}",
            model.log_likelihood(&[5.0, 2.0], &[1.0, f64::INFINITY, 3.0])
        );
    }

    #[test]
    fn fit_recovers_closed_form_and_information_criteria() -> Result<()> {
        // python3:
        //   import numpy as np; from scipy import stats
        //   data = np.array([2.,4.,4.,4.,5.,5.,7.,9.])
        //   data.mean()             # 5.0
        //   data.std(ddof=0)        # 2.0  (biased MLE)
        //   ll = stats.norm.logpdf(data, 5.0, 2.0).sum()  # -16.896685710116945
        let model = NormalLikelihood::default();
        let fit = model.fit(&DATA)?;
        let mu_hat = param(&fit, 0)?;
        let sigma_hat = param(&fit, 1)?;
        assert!(
            (mu_hat - 5.0).abs() < 1e-12,
            "mu_hat was {mu_hat}, want 5.0"
        );
        assert!(
            (sigma_hat - 2.0).abs() < 1e-12,
            "sigma_hat was {sigma_hat}, want 2.0"
        );
        let want_ll = -16.896_685_710_116_945;
        let ll = fit.log_likelihood();
        assert!(
            ((ll - want_ll) / want_ll).abs() < 1e-10,
            "log_likelihood was {ll}, want {want_ll}"
        );
        assert!(fit.converged(), "closed-form fit must report converged");
        assert_eq!(fit.iterations(), 0, "closed-form fit does no iterations");
        // Information-criteria arithmetic identity: 2k − 2ℓ and k·ln(n) − 2ℓ, k = 2.
        let n = count_to_f64(DATA.len());
        let akaike = 2.0f64.mul_add(2.0, -2.0 * ll);
        let bayesian = 2.0f64.mul_add(n.ln(), -2.0 * ll);
        assert!(
            (fit.aic() - akaike).abs() < 1e-12,
            "aic was {}, want {akaike}",
            fit.aic()
        );
        assert!(
            (fit.bic() - bayesian).abs() < 1e-12,
            "bic was {}, want {bayesian}",
            fit.bic()
        );
        Ok(())
    }

    #[test]
    fn fit_mle_from_perturbed_init_recovers_closed_form() -> Result<()> {
        // The free L-BFGS optimizer, started away from the optimum, must land on
        // the same [μ̂, σ̂] the closed form gives (μ = 5, σ = 2).
        let model = NormalLikelihood::default();
        let fit = fit_mle(&model, &DATA, &[3.5, 3.0], 1e-10)?;
        let mu_hat = param(&fit, 0)?;
        let sigma_hat = param(&fit, 1)?;
        assert!(
            (mu_hat - 5.0).abs() <= 1e-5,
            "mu_hat was {mu_hat}, want 5.0"
        );
        assert!(
            (sigma_hat - 2.0).abs() <= 1e-5,
            "sigma_hat was {sigma_hat}, want 2.0"
        );
        Ok(())
    }
}
