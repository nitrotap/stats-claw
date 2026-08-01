//! Exponential-distribution maximum-likelihood, for the
//! [`ExponentialLikelihood`](crate::likelihood::ExponentialLikelihood).
//!
//! The model is the rate parameterization `f(x; λ) = λ·e^(−λx)` for `x ≥ 0`,
//! `λ > 0`, so the total log-likelihood of a sample is
//! `ℓ(λ; x) = n·ln λ − λ·Σxᵢ`. This is `scipy.stats.expon` with `scale = 1/λ`.
//! [`ExponentialLikelihood`](crate::likelihood::ExponentialLikelihood) supplies the
//! parameter struct; the numerics — the [`LogLikelihood`] impl and the
//! closed-form MLE `λ̂ = n / Σxᵢ` in [`fit`](crate::likelihood::ExponentialLikelihood::fit)
//! — are written here.
//!
//! # Examples
//!
//! ```
//! use stats_claw::likelihood::ExponentialLikelihood;
//!
//! let model = ExponentialLikelihood::default();
//! // Closed-form rate MLE of [0.5, 1.2, 2.3, 0.8, 3.1] is 5 / 7.9.
//! let fit = model.fit(&[0.5, 1.2, 2.3, 0.8, 3.1])?;
//! assert!((fit.params()[0] - 5.0 / 7.9).abs() < 1e-12, "lambda_hat was {}", fit.params()[0]);
//! # Ok::<(), stats_claw::error::Error>(())
//! ```

use crate::algorithms::count_to_f64;
use crate::error::{Error, Result};
use crate::likelihood::{LogLikelihood, MleFit};

impl crate::likelihood::ExponentialLikelihood {
    /// Fits this exponential model to `data` by its closed-form maximum-likelihood
    /// estimate `λ̂ = n / Σxᵢ`.
    ///
    /// The rate MLE is analytic, so the returned [`MleFit`] reports
    /// [`converged`](MleFit::converged) `= true` and zero
    /// [`iterations`](MleFit::iterations); its
    /// [`log_likelihood`](MleFit::log_likelihood) is `ℓ(λ̂; data)` and the AIC/BIC
    /// follow from `k = 1` parameter and `n = data.len()`.
    ///
    /// # Arguments
    ///
    /// * `data` — the observed sample; every value must be `≥ 0` (the exponential
    ///   support) and the sample must be non-empty with a strictly positive sum.
    ///
    /// # Returns
    ///
    /// An [`MleFit`] whose single parameter is the rate estimate `λ̂`.
    ///
    /// # Errors
    ///
    /// * [`Error::InsufficientData`] if `data` is empty.
    /// * [`Error::InvalidInput`] if any observation is negative (outside the
    ///   exponential support) or non-finite.
    /// * [`Error::DegenerateInput`] if every observation is zero (`Σxᵢ = 0`), which
    ///   would make `λ̂ = n / 0` infinite.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::likelihood::ExponentialLikelihood;
    ///
    /// let fit = ExponentialLikelihood::default().fit(&[1.0, 2.0, 3.0])?;
    /// // λ̂ = 3 / 6 = 0.5.
    /// assert!((fit.params()[0] - 0.5).abs() < 1e-12, "lambda_hat was {}", fit.params()[0]);
    /// assert!(fit.converged());
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    pub fn fit(&self, data: &[f64]) -> Result<MleFit> {
        if data.is_empty() {
            return Err(Error::InsufficientData);
        }
        let mut sum = 0.0_f64;
        for &x in data {
            if !x.is_finite() || x < 0.0 {
                return Err(Error::InvalidInput(format!(
                    "exponential data must be finite and >= 0, got {x}"
                )));
            }
            sum += x;
        }
        if sum <= 0.0 {
            return Err(Error::DegenerateInput(
                "all observations are zero, so the rate MLE n / sum(x) is infinite".to_owned(),
            ));
        }
        let n = count_to_f64(data.len());
        let lambda_hat = n / sum;
        // ℓ(λ̂) = n·ln λ̂ − λ̂·Σx, evaluated through the trait for a single source
        // of truth for the formula.
        let log_likelihood = self.log_likelihood(&[lambda_hat], data);
        Ok(MleFit::from_closed_form(
            vec![lambda_hat],
            log_likelihood,
            data.len(),
        ))
    }
}

impl LogLikelihood for crate::likelihood::ExponentialLikelihood {
    /// Returns `1`: the exponential rate model has the single parameter `λ`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::likelihood::ExponentialLikelihood;
    /// use stats_claw::likelihood::LogLikelihood;
    ///
    /// assert_eq!(ExponentialLikelihood::default().n_params(), 1);
    /// ```
    fn n_params(&self) -> usize {
        1
    }

    /// Evaluates the total exponential log-likelihood `ℓ(λ; data) = n·ln λ − λ·Σxᵢ`.
    ///
    /// # Arguments
    ///
    /// * `params` — the one-element rate vector `[λ]`; only `params[0]` is read.
    /// * `data` — the observed sample.
    ///
    /// # Returns
    ///
    /// The scalar log-likelihood, or [`f64::NEG_INFINITY`] when `λ ≤ 0`, or any
    /// observation is negative or non-finite — all lie outside the model's valid
    /// domain. Per the [`LogLikelihood`] contract a non-finite observation (`NaN`
    /// or `±∞`) yields `−∞` rather than propagating a `NaN`/incidental `−∞`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::likelihood::ExponentialLikelihood;
    /// use stats_claw::likelihood::LogLikelihood;
    ///
    /// let model = ExponentialLikelihood::default();
    /// // ℓ(1; [1, 2]) = 2·ln 1 − 1·3 = −3.
    /// assert!((model.log_likelihood(&[1.0], &[1.0, 2.0]) + 3.0).abs() < 1e-12);
    /// // A non-positive rate is outside the domain.
    /// assert_eq!(model.log_likelihood(&[0.0], &[1.0]), f64::NEG_INFINITY);
    /// ```
    fn log_likelihood(&self, params: &[f64], data: &[f64]) -> f64 {
        let lambda = *params.first().unwrap_or(&0.0);
        // A non-positive rate is outside the domain; `is_nan` guards the case a
        // caller passes a NaN rate directly (for which `lambda <= 0.0` is false).
        if lambda <= 0.0 || lambda.is_nan() {
            return f64::NEG_INFINITY;
        }
        let mut sum = 0.0_f64;
        for &x in data {
            // A non-finite observation (NaN/±∞) is outside the support; guard it
            // explicitly since `x < 0.0` is false for NaN and `+∞` would leak an
            // incidental `−∞` through `−λ·Σx`.
            if !x.is_finite() || x < 0.0 {
                return f64::NEG_INFINITY;
            }
            sum += x;
        }
        let n = count_to_f64(data.len());
        // n·ln λ − λ·Σx.
        n.mul_add(lambda.ln(), -lambda * sum)
    }
}

#[cfg(test)]
mod tests {
    use crate::error::Error;
    use crate::likelihood::ExponentialLikelihood;
    use crate::likelihood::{LogLikelihood, fit_mle};

    /// Shared golden sample. Reference values below were produced by:
    ///
    /// ```python
    /// import numpy as np
    /// from scipy import stats
    /// data = np.array([0.5, 1.2, 2.3, 0.8, 3.1])
    /// n, sx = len(data), data.sum()            # n = 5, sx = 7.9
    /// lam_hat = n / sx                          # 0.6329113924050632
    /// stats.expon.logpdf(data, scale=1/0.5).sum()   # -7.415735902799727
    /// stats.expon.logpdf(data, scale=1/lam_hat).sum()  # -7.287124235194378 (= ll at lam_hat)
    /// 2*1 - 2*ll_hat                            # aic = 16.574248470388756
    /// 1*np.log(n) - 2*ll_hat                    # bic = 16.183686382822856
    /// ```
    const DATA: [f64; 5] = [0.5, 1.2, 2.3, 0.8, 3.1];
    /// Closed-form rate MLE `λ̂ = 5 / 7.9`.
    const LAMBDA_HAT: f64 = 0.632_911_392_405_063_2;
    /// `scipy.stats.expon.logpdf(DATA, scale=1/0.5).sum()`.
    const LL_AT_HALF: f64 = -7.415_735_902_799_727;
    /// `ℓ(λ̂; DATA)` — the log-likelihood at the fitted rate.
    const LL_AT_HAT: f64 = -7.287_124_235_194_378;
    /// `2k − 2ℓ(λ̂)` with `k = 1`.
    const AIC_AT_HAT: f64 = 16.574_248_470_388_756;
    /// `k·ln n − 2ℓ(λ̂)` with `k = 1`, `n = 5`.
    const BIC_AT_HAT: f64 = 16.183_686_382_822_856;

    /// Builds the model under test; its descriptive string fields are irrelevant to
    /// the numerics, so they are left at their defaults.
    fn model() -> ExponentialLikelihood {
        ExponentialLikelihood::default()
    }

    /// Asserts `a` and `b` agree to `rel` relative error.
    fn rel_close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs().max(1.0)
    }

    /// Returns whether `x` is exactly negative infinity, without an exact `==`
    /// float comparison (which the lint gate rejects).
    fn is_neg_inf(x: f64) -> bool {
        x.is_infinite() && x.is_sign_negative()
    }

    /// Reads the sole fitted rate without slice indexing (kept lint-clean).
    fn first_param(fit: &crate::likelihood::MleFit) -> f64 {
        fit.params().first().copied().unwrap_or(f64::NAN)
    }

    #[test]
    fn fit_rejects_empty_data() {
        let got = model().fit(&[]);
        assert!(
            matches!(got, Err(Error::InsufficientData)),
            "empty fit was {got:?}"
        );
    }

    #[test]
    fn fit_rejects_negative_observation() {
        let got = model().fit(&[1.0, -0.5, 2.0]);
        assert!(
            matches!(got, Err(Error::InvalidInput(_))),
            "negative fit was {got:?}"
        );
    }

    #[test]
    fn fit_rejects_all_zero_data() {
        let got = model().fit(&[0.0, 0.0, 0.0]);
        assert!(
            matches!(got, Err(Error::DegenerateInput(_))),
            "all-zero fit was {got:?}"
        );
    }

    #[test]
    fn log_likelihood_matches_scipy() {
        let got = model().log_likelihood(&[0.5], &DATA);
        assert!(
            rel_close(got, LL_AT_HALF, 1e-10),
            "ll@0.5 was {got}, expected {LL_AT_HALF}"
        );
    }

    #[test]
    fn log_likelihood_is_neg_inf_for_nonpositive_rate() {
        let m = model();
        assert!(
            is_neg_inf(m.log_likelihood(&[0.0], &DATA)),
            "ll at rate 0 was {}",
            m.log_likelihood(&[0.0], &DATA)
        );
        assert!(
            is_neg_inf(m.log_likelihood(&[-1.0], &DATA)),
            "ll at rate -1 was {}",
            m.log_likelihood(&[-1.0], &DATA)
        );
    }

    #[test]
    fn log_likelihood_is_neg_inf_for_negative_observation() {
        let got = model().log_likelihood(&[1.0], &[1.0, -2.0]);
        assert!(is_neg_inf(got), "ll with negative x was {got}");
    }

    #[test]
    fn log_likelihood_non_finite_observation_is_neg_inf() {
        let m = model();
        // A NaN observation gives NEG_INFINITY, not the NaN that `Σxᵢ` would
        // otherwise propagate (`x < 0.0` is false for NaN).
        assert!(
            is_neg_inf(m.log_likelihood(&[1.0], &[1.0, f64::NAN, 2.0])),
            "NaN observation should give NEG_INFINITY, got {}",
            m.log_likelihood(&[1.0], &[1.0, f64::NAN, 2.0])
        );
        // A +∞ observation likewise — pinned explicitly rather than left to the
        // `−λ·Σxᵢ` sign.
        assert!(
            is_neg_inf(m.log_likelihood(&[1.0], &[1.0, f64::INFINITY, 2.0])),
            "+inf observation should give NEG_INFINITY, got {}",
            m.log_likelihood(&[1.0], &[1.0, f64::INFINITY, 2.0])
        );
    }

    #[test]
    fn fit_recovers_closed_form_estimate() -> Result<(), Error> {
        let fit = model().fit(&DATA)?;
        let lambda = first_param(&fit);
        assert!(
            (lambda - LAMBDA_HAT).abs() <= 1e-12,
            "lambda_hat was {lambda}"
        );
        assert!(
            rel_close(fit.log_likelihood(), LL_AT_HAT, 1e-10),
            "ll was {}",
            fit.log_likelihood()
        );
        assert!(fit.converged(), "closed-form fit should report converged");
        assert_eq!(fit.iterations(), 0, "closed-form fit does zero iterations");
        assert!(
            rel_close(fit.aic(), AIC_AT_HAT, 1e-10),
            "aic was {}",
            fit.aic()
        );
        assert!(
            rel_close(fit.bic(), BIC_AT_HAT, 1e-10),
            "bic was {}",
            fit.bic()
        );
        Ok(())
    }

    #[test]
    fn fit_mle_from_perturbed_init_recovers_lambda_hat() -> Result<(), Error> {
        // Start well away from λ̂ ≈ 0.633 but inside the valid domain (λ > 0).
        let fit = fit_mle(&model(), &DATA, &[0.9], 1e-10)?;
        let lambda = first_param(&fit);
        assert!(
            (lambda - LAMBDA_HAT).abs() <= 1e-5,
            "lambda_hat from optimizer was {lambda}"
        );
        Ok(())
    }
}
