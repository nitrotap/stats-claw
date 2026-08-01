//! Generic maximum-likelihood estimation for the
//! [`MaximumLikelihood`](crate::likelihood::MaximumLikelihood).
//!
//! A parametric model implements [`LogLikelihood`] — the log-likelihood
//! `ℓ(θ; data)` of a parameter vector `θ` given `f64` observations. [`fit_mle`]
//! then finds the maximum-likelihood estimate `θ̂ = argmaxθ ℓ` by *minimizing*
//! `−ℓ` with the framework's L-BFGS optimizer, reporting the fitted parameters
//! alongside the Akaike and Bayesian information criteria in an [`MleFit`].
//!
//! This module is the shared foundation the concrete likelihood models
//! (Normal, Poisson, Binomial, Categorical, Exponential) build on: each supplies
//! its own [`LogLikelihood`] and defers the numerical optimization to [`fit_mle`].

use crate::algorithms::count_to_f64;
use crate::error::{Error, Result};
use crate::optimizers::second_order::lbfgs;
use crate::optimizers::{ConvergenceStatus, Objective};

/// Iteration budget handed to the underlying L-BFGS optimizer. Large enough that
/// convergence is governed by the gradient-norm `tolerance` rather than the
/// budget for the smooth likelihoods this framework fits.
const MAX_ITER: usize = 1_000;

/// Finite penalty substituted for `+∞` when the model reports `ℓ = −∞` (a `θ`
/// outside its valid domain). A large finite value keeps the optimizer's line
/// search and finite-difference gradient well defined and steers trial steps
/// back toward the interior instead of stalling on a non-finite objective.
const DOMAIN_PENALTY: f64 = 1e300;

/// Fallback gradient-norm tolerance used when
/// [`MaximumLikelihood::fit`](crate::likelihood::MaximumLikelihood::fit) is
/// called with a non-positive stored `convergence_tolerance` (e.g. the default
/// `0.0`).
const DEFAULT_TOLERANCE: f64 = 1e-8;

/// A parametric log-likelihood `ℓ(θ; data)` over `f64` observations.
///
/// Implementors describe a family of probability models indexed by a parameter
/// vector `params` (`θ`); [`log_likelihood`](LogLikelihood::log_likelihood)
/// returns the total log-likelihood of `data` under `θ`. Returning
/// [`f64::NEG_INFINITY`] marks a `θ` outside the valid parameter domain (for
/// example a non-positive standard deviation), which [`fit_mle`] treats as a
/// hard constraint.
///
/// Implementations must return [`f64::NEG_INFINITY`] for any non-finite
/// observation (`NaN` or `±∞`) rather than propagating a `NaN`: a non-finite
/// datum lies outside every model's support and must not silently corrupt the
/// objective the optimizer sees.
///
/// # Examples
///
/// ```
/// use stats_claw::likelihood::LogLikelihood;
///
/// // A one-parameter Gaussian-mean model: ℓ(μ) = −Σ(xᵢ − μ)².
/// struct MeanModel;
/// impl LogLikelihood for MeanModel {
///     fn n_params(&self) -> usize { 1 }
///     fn log_likelihood(&self, p: &[f64], d: &[f64]) -> f64 { -d.iter().map(|x| (x - p[0]).powi(2)).sum::<f64>() }
/// }
///
/// let m = MeanModel;
/// // The likelihood is higher at the sample mean (2) than away from it (0).
/// assert!(m.log_likelihood(&[2.0], &[1.0, 3.0]) > m.log_likelihood(&[0.0], &[1.0, 3.0]));
/// ```
pub trait LogLikelihood {
    /// Returns the number of free parameters, i.e. the required length of
    /// `params`.
    fn n_params(&self) -> usize;

    /// Evaluates `ℓ(params; data)`, the total log-likelihood of `data`.
    ///
    /// # Arguments
    ///
    /// * `params` — the parameter vector `θ`; length must equal
    ///   [`n_params`](LogLikelihood::n_params).
    /// * `data` — the observed sample.
    ///
    /// # Returns
    ///
    /// The scalar log-likelihood, or [`f64::NEG_INFINITY`] when `params` lies
    /// outside the model's valid domain.
    fn log_likelihood(&self, params: &[f64], data: &[f64]) -> f64;
}

/// The outcome of a maximum-likelihood fit produced by [`fit_mle`].
///
/// The fitted parameters and diagnostics are read through accessor methods (see
/// [`MleFit::params`]); the fields are private because the struct owns a heap
/// parameter vector and the framework keeps memory-owning types fully
/// encapsulated.
#[derive(Debug, Clone)]
pub struct MleFit {
    /// The fitted maximum-likelihood estimate `θ̂` (length `n_params`).
    params: Vec<f64>,
    /// The log-likelihood `ℓ(θ̂; data)` at the fitted parameters.
    log_likelihood: f64,
    /// Whether the optimizer's convergence criterion was met (vs. budget hit).
    converged: bool,
    /// Number of optimizer iterations performed.
    iterations: usize,
    /// Akaike information criterion, `2k − 2ℓ` (`k` = number of parameters).
    aic: f64,
    /// Bayesian information criterion, `k·ln(n) − 2ℓ` (`n` = sample size).
    bic: f64,
}

impl MleFit {
    /// Returns the fitted maximum-likelihood estimate `θ̂` as a slice.
    ///
    /// # Returns
    ///
    /// The fitted parameters in the model's parameter order (length
    /// `n_params`).
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::likelihood::{fit_mle, LogLikelihood};
    /// # struct M;
    /// # impl LogLikelihood for M {
    /// #     fn n_params(&self) -> usize { 1 }
    /// #     fn log_likelihood(&self, p: &[f64], d: &[f64]) -> f64 { -d.iter().map(|x| (x - p[0]).powi(2)).sum::<f64>() }
    /// # }
    /// let fit = fit_mle(&M, &[2.0, 4.0], &[0.0], 1e-9)?;
    /// assert_eq!(fit.params().len(), 1);
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    #[must_use]
    pub fn params(&self) -> &[f64] {
        &self.params
    }

    /// Returns the log-likelihood `ℓ(θ̂; data)` at the fitted parameters.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::likelihood::{fit_mle, LogLikelihood};
    /// # struct M;
    /// # impl LogLikelihood for M {
    /// #     fn n_params(&self) -> usize { 1 }
    /// #     fn log_likelihood(&self, p: &[f64], d: &[f64]) -> f64 { -d.iter().map(|x| (x - p[0]).powi(2)).sum::<f64>() }
    /// # }
    /// let fit = fit_mle(&M, &[2.0, 4.0], &[0.0], 1e-9)?;
    /// assert!(fit.log_likelihood().is_finite());
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    #[must_use]
    pub const fn log_likelihood(&self) -> f64 {
        self.log_likelihood
    }

    /// Returns whether the optimizer's convergence criterion was met (as opposed
    /// to exhausting its iteration budget).
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::likelihood::{fit_mle, LogLikelihood};
    /// # struct M;
    /// # impl LogLikelihood for M {
    /// #     fn n_params(&self) -> usize { 1 }
    /// #     fn log_likelihood(&self, p: &[f64], d: &[f64]) -> f64 { -d.iter().map(|x| (x - p[0]).powi(2)).sum::<f64>() }
    /// # }
    /// let fit = fit_mle(&M, &[2.0, 4.0], &[0.0], 1e-9)?;
    /// assert!(fit.converged());
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    #[must_use]
    pub const fn converged(&self) -> bool {
        self.converged
    }

    /// Returns the number of optimizer iterations performed.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::likelihood::{fit_mle, LogLikelihood};
    /// # struct M;
    /// # impl LogLikelihood for M {
    /// #     fn n_params(&self) -> usize { 1 }
    /// #     fn log_likelihood(&self, p: &[f64], d: &[f64]) -> f64 { -d.iter().map(|x| (x - p[0]).powi(2)).sum::<f64>() }
    /// # }
    /// let fit = fit_mle(&M, &[2.0, 4.0], &[0.0], 1e-9)?;
    /// assert!(fit.iterations() >= 1);
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    #[must_use]
    pub const fn iterations(&self) -> usize {
        self.iterations
    }

    /// Returns the Akaike information criterion, `2k − 2ℓ`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::likelihood::{fit_mle, LogLikelihood};
    /// # struct M;
    /// # impl LogLikelihood for M {
    /// #     fn n_params(&self) -> usize { 1 }
    /// #     fn log_likelihood(&self, p: &[f64], d: &[f64]) -> f64 { -d.iter().map(|x| (x - p[0]).powi(2)).sum::<f64>() }
    /// # }
    /// let fit = fit_mle(&M, &[2.0, 4.0], &[0.0], 1e-9)?;
    /// assert!(fit.aic().is_finite());
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    #[must_use]
    pub const fn aic(&self) -> f64 {
        self.aic
    }

    /// Returns the Bayesian information criterion, `k·ln(n) − 2ℓ`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::likelihood::{fit_mle, LogLikelihood};
    /// # struct M;
    /// # impl LogLikelihood for M {
    /// #     fn n_params(&self) -> usize { 1 }
    /// #     fn log_likelihood(&self, p: &[f64], d: &[f64]) -> f64 { -d.iter().map(|x| (x - p[0]).powi(2)).sum::<f64>() }
    /// # }
    /// let fit = fit_mle(&M, &[2.0, 4.0], &[0.0], 1e-9)?;
    /// assert!(fit.bic().is_finite());
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    #[must_use]
    pub const fn bic(&self) -> f64 {
        self.bic
    }

    /// Builds a fit from a closed-form (analytic) maximum-likelihood solution.
    ///
    /// Sibling likelihood modules whose MLE has a closed form (e.g. the Normal
    /// mean/variance) use this to return an [`MleFit`] without exposing the
    /// private fields. Because the solution is exact, `converged` is `true` and
    /// `iterations` is `0`; the AIC/BIC are computed from `k = params.len()` and
    /// `n = n_observations` with the same [`info_criteria`] formulas as
    /// [`fit_mle`].
    ///
    /// # Arguments
    ///
    /// * `params` — the analytic estimate `θ̂`.
    /// * `log_likelihood` — the log-likelihood `ℓ(θ̂; data)` at that estimate.
    /// * `n_observations` — the sample size `n`, used only for the BIC.
    // Exercised by this module's tests and consumed by the sibling likelihood
    // modules (Normal/Poisson/…) landing in follow-up tasks; it therefore has no
    // in-crate caller yet in a non-test build.
    #[allow(dead_code)]
    pub(crate) fn from_closed_form(
        params: Vec<f64>,
        log_likelihood: f64,
        n_observations: usize,
    ) -> Self {
        let (aic, bic) = info_criteria(params.len(), n_observations, log_likelihood);
        Self {
            params,
            log_likelihood,
            converged: true,
            iterations: 0,
            aic,
            bic,
        }
    }
}

/// Numerically maximizes `ℓ` from `init` by minimizing `−ℓ` with L-BFGS.
///
/// # Arguments
///
/// * `model` — the parametric log-likelihood to fit.
/// * `data` — the observed sample; must be non-empty.
/// * `init` — the starting parameter vector; length must equal
///   `model.n_params()`.
/// * `tolerance` — the gradient-norm convergence threshold; must be `> 0`.
///
/// # Returns
///
/// An [`MleFit`] with the fitted parameters, the attained log-likelihood, the
/// convergence flag, the iteration count, and the AIC/BIC.
///
/// # Errors
///
/// * [`Error::InsufficientData`] if `data` is empty.
/// * [`Error::InvalidInput`] if `init.len() != model.n_params()`, if
///   `tolerance <= 0`, or if `init` lies outside the model's valid domain (i.e.
///   `model.log_likelihood(init, data)` is not finite).
///
/// # Examples
///
/// ```
/// use stats_claw::likelihood::{fit_mle, LogLikelihood};
///
/// struct MeanModel;
/// impl LogLikelihood for MeanModel {
///     fn n_params(&self) -> usize { 1 }
///     fn log_likelihood(&self, p: &[f64], d: &[f64]) -> f64 { -d.iter().map(|x| (x - p[0]).powi(2)).sum::<f64>() }
/// }
///
/// // The MLE of the mean model is the sample mean, here 2.0.
/// let fit = fit_mle(&MeanModel, &[1.0, 2.0, 3.0], &[0.0], 1e-9)?;
/// assert!((fit.params()[0] - 2.0).abs() < 1e-5, "mu_hat was {}", fit.params()[0]);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn fit_mle(
    model: &impl LogLikelihood,
    data: &[f64],
    init: &[f64],
    tolerance: f64,
) -> Result<MleFit> {
    if data.is_empty() {
        return Err(Error::InsufficientData);
    }
    if init.len() != model.n_params() {
        return Err(Error::InvalidInput(format!(
            "init has {} entries but model has {} parameters",
            init.len(),
            model.n_params()
        )));
    }
    if tolerance <= 0.0 {
        return Err(Error::InvalidInput("tolerance must be > 0".to_owned()));
    }
    // The objective substitutes a flat penalty for `ℓ = −∞`, so a start outside
    // the valid domain yields a zero numerical gradient and lbfgs would "converge"
    // immediately at the invalid point. Reject such a start up front.
    if !model.log_likelihood(init, data).is_finite() {
        return Err(Error::InvalidInput(
            "init is outside the model's valid parameter domain (log-likelihood is not finite)"
                .to_owned(),
        ));
    }
    let objective = NegLogLikelihood { model, data };
    let result = lbfgs(&objective, init, MAX_ITER, tolerance);
    let params = result.x;
    let log_likelihood = model.log_likelihood(&params, data);
    let (aic, bic) = info_criteria(model.n_params(), data.len(), log_likelihood);
    Ok(MleFit {
        params,
        log_likelihood,
        converged: matches!(result.status, ConvergenceStatus::Converged),
        iterations: result.iterations,
        aic,
        bic,
    })
}

/// Computes the Akaike and Bayesian information criteria.
///
/// # Arguments
///
/// * `k` — the number of free parameters.
/// * `n` — the sample size (`≥ 1`, guaranteed by [`fit_mle`]'s guards).
/// * `log_likelihood` — the maximized log-likelihood `ℓ(θ̂)`.
///
/// # Returns
///
/// The pair `(aic, bic)` where `aic = 2k − 2ℓ` and `bic = k·ln(n) − 2ℓ`.
fn info_criteria(k: usize, n: usize, log_likelihood: f64) -> (f64, f64) {
    let kf = count_to_f64(k);
    let nf = count_to_f64(n);
    let aic = 2.0_f64.mul_add(kf, -2.0 * log_likelihood);
    let bic = kf.mul_add(nf.ln(), -2.0 * log_likelihood);
    (aic, bic)
}

/// Adapts `−ℓ` of a [`LogLikelihood`] into an [`Objective`] for the optimizer.
///
/// [`value`](Objective::value) returns `−ℓ`, substituting [`DOMAIN_PENALTY`] for
/// any non-finite value so out-of-domain trial points stay usable;
/// [`grad`](Objective::grad) is a central finite-difference approximation, since
/// the generic likelihood exposes no analytic gradient. The model is held as a
/// trait object so the [`Objective`] impl is fully concrete.
///
/// # Notes
///
/// The finite-difference step `h = max(1e-6, 1e-6·|xᵢ|)` is deliberately coarse:
/// for parameters legitimately scaled far below `1e-3`, the absolute floor of
/// `1e-6` dominates and the differencing step is large relative to `|xᵢ|`, so the
/// gradient there is only crudely accurate. Rescale such parameters before
/// fitting if a tight gradient is required.
struct NegLogLikelihood<'a> {
    /// The wrapped log-likelihood model.
    model: &'a dyn LogLikelihood,
    /// The observed sample the log-likelihood is evaluated against.
    data: &'a [f64],
}

impl Objective for NegLogLikelihood<'_> {
    fn value(&self, x: &[f64]) -> f64 {
        let ll = self.model.log_likelihood(x, self.data);
        if ll.is_finite() { -ll } else { DOMAIN_PENALTY }
    }

    fn grad(&self, x: &[f64]) -> Vec<f64> {
        // Central difference with a relative step h = max(1e-6, 1e-6·|xᵢ|).
        // Near a domain boundary a probe can land on the penalized (`ℓ = −∞`)
        // side; using it would corrupt the derivative, so fall back to a
        // one-sided difference against the (finite) centre when that happens.
        let center = self.value(x);
        (0..x.len())
            .map(|i| {
                let xi = *x.get(i).unwrap_or(&0.0);
                let h = 1e-6_f64.max(1e-6 * xi.abs());
                let mut probe = x.to_vec();
                if let Some(slot) = probe.get_mut(i) {
                    *slot = xi + h;
                }
                let forward = self.value(&probe);
                if let Some(slot) = probe.get_mut(i) {
                    *slot = xi - h;
                }
                let backward = self.value(&probe);
                let fwd_ok = forward < DOMAIN_PENALTY;
                let bwd_ok = backward < DOMAIN_PENALTY;
                match (fwd_ok, bwd_ok) {
                    (true, true) => (forward - backward) / (2.0 * h),
                    (true, false) => (forward - center) / h,
                    (false, true) => (center - backward) / h,
                    // Reachable: when the valid domain is narrower than 2h around
                    // the iterate, both probes fall on the penalized side and
                    // neither one-sided difference is trustworthy. Report a zero
                    // component — the conservative choice, since it neither pushes
                    // the optimizer out of the feasible region nor invents a slope
                    // from the flat penalty; the line search then relies on the
                    // other (finite) components to make progress.
                    (false, false) => 0.0,
                }
            })
            .collect()
    }
}

impl crate::likelihood::MaximumLikelihood {
    /// Fits `model` to `data` from `init`, using this instance's
    /// [`convergence_tolerance`](crate::likelihood::MaximumLikelihood::convergence_tolerance).
    ///
    /// The stored tolerance is the gradient-norm threshold forwarded to
    /// [`fit_mle`]. When the field is left at its default of `0.0` (a
    /// non-positive, unusable threshold), it falls back to `1e-8`.
    ///
    /// # Arguments
    ///
    /// * `model` — the parametric log-likelihood to fit.
    /// * `data` — the observed sample; must be non-empty.
    /// * `init` — the starting parameter vector; length must equal
    ///   `model.n_params()`.
    ///
    /// # Errors
    ///
    /// Propagates the errors of [`fit_mle`]: [`Error::InsufficientData`] for
    /// empty `data`, and [`Error::InvalidInput`] for an `init`/parameter-count
    /// mismatch, a non-positive resolved tolerance, or an `init` outside the
    /// model's valid domain (non-finite log-likelihood).
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::likelihood::MaximumLikelihood;
    /// use stats_claw::likelihood::LogLikelihood;
    ///
    /// struct MeanModel;
    /// impl LogLikelihood for MeanModel {
    ///     fn n_params(&self) -> usize { 1 }
    ///     fn log_likelihood(&self, p: &[f64], d: &[f64]) -> f64 { -d.iter().map(|x| (x - p[0]).powi(2)).sum::<f64>() }
    /// }
    ///
    /// let mle = MaximumLikelihood { convergence_tolerance: 1e-9, ..Default::default() };
    /// let fit = mle.fit(&MeanModel, &[2.0, 4.0, 6.0], &[0.0])?;
    /// assert!((fit.params()[0] - 4.0).abs() < 1e-5, "mu_hat was {}", fit.params()[0]);
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    pub fn fit(&self, model: &impl LogLikelihood, data: &[f64], init: &[f64]) -> Result<MleFit> {
        let tolerance = if self.convergence_tolerance > 0.0 {
            self.convergence_tolerance
        } else {
            DEFAULT_TOLERANCE
        };
        fit_mle(model, data, init, tolerance)
    }
}

#[cfg(test)]
mod tests;
