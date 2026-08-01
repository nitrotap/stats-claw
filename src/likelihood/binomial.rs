//! Binomial maximum-likelihood numerics, for the
//! [`BinomialLikelihood`].
//!
//! # Model
//!
//! `N` independent draws from `Binomial(n_trials, p)`, observed as per-draw
//! success counts `x ∈ {0, 1, …, n_trials}` stored as `f64`. The single free
//! parameter is the success probability `p ∈ (0, 1)`, so `n_params() == 1`. The
//! total log-likelihood is
//!
//! ```text
//! ℓ(p; x) = Σᵢ [ ln C(n, xᵢ) + xᵢ ln p + (n − xᵢ) ln(1 − p) ]
//! ```
//!
//! matching `scipy.stats.binom.logpmf(x, n, p).sum()`.
//!
//! # Design: where `n_trials` lives
//!
//! The frozen [`LogLikelihood`] trait carries
//! no slot for the number of trials — its `log_likelihood(params, data)` sees
//! only the free parameter vector and the data. The
//! [`BinomialLikelihood`] struct, however, *already* carries a
//! [`number_of_trials`](crate::likelihood::BinomialLikelihood) field, so this
//! module attaches the numerics directly to that struct and reads `n_trials`
//! from it (rather than introducing a separate wrapper type). This keeps the
//! trait impl honest — `n_trials` is genuine model state owned by the model —
//! and lets [`fit_mle`](crate::likelihood::fit_mle) fit a `&BinomialLikelihood`
//! directly. Construct the model with the trials set, e.g.
//! `BinomialLikelihood { number_of_trials: 10, ..Default::default() }`.
//!
//! # Examples
//!
//! ```
//! use stats_claw::likelihood::BinomialLikelihood;
//! use stats_claw::likelihood::LogLikelihood;
//!
//! let model = BinomialLikelihood { number_of_trials: 10, ..Default::default() };
//! // scipy: binom.logpmf([3,5,2,4,6], 10, 0.4).sum() = -8.83278496896409
//! let ll = model.log_likelihood(&[0.4], &[3.0, 5.0, 2.0, 4.0, 6.0]);
//! assert!((ll - -8.832_784_968_964_091_4).abs() < 1e-10, "ll was {ll}");
//! ```

use crate::algorithms::count_to_f64;
use crate::error::{Error, Result};
use crate::likelihood::BinomialLikelihood;
use crate::likelihood::{LogLikelihood, MleFit};
use crate::special::ln_choose;

/// Absolute tolerance within which a data point is accepted as an integer
/// success count. Success counts are exact integers stored as `f64`, so this
/// only forgives sub-ULP rounding noise; genuinely fractional values (e.g.
/// `2.5`) are rejected as invalid.
const INTEGRALITY_TOL: f64 = 1e-9;

/// Upper bit used when recovering the `usize` value of an integral `f64`. Covers
/// success counts up to `2^40`, far beyond any realistic binomial trial count.
const FLOOR_SEARCH_TOP_BIT: usize = 1 << 40;

impl BinomialLikelihood {
    /// Closed-form maximum-likelihood fit of the success probability `p`.
    ///
    /// The binomial MLE is `p̂ = (Σᵢ xᵢ) / (N · n_trials)` — the pooled success
    /// rate across all `N` draws. Because the estimate is analytic, the returned
    /// [`MleFit`] reports `converged() == true` and `iterations() == 0`, with the
    /// AIC/BIC computed for `k = 1` parameter over `N` observations.
    ///
    /// # Arguments
    ///
    /// * `data` — the observed per-draw success counts; each must be an integer
    ///   in `[0, n_trials]`. Must be non-empty.
    ///
    /// # Returns
    ///
    /// An [`MleFit`] whose single parameter is `p̂`.
    ///
    /// # Errors
    ///
    /// * [`Error::InsufficientData`] if `data` is empty.
    /// * [`Error::InvalidInput`] if this model's `number_of_trials` is not a
    ///   positive integer, or if any observation is negative, non-integral, or
    ///   greater than `number_of_trials`.
    /// * [`Error::DegenerateInput`] if every observation is `0` or every
    ///   observation is `n_trials` (so `p̂` lands on the boundary `{0, 1}` and the
    ///   log-likelihood is degenerate).
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::likelihood::BinomialLikelihood;
    ///
    /// let model = BinomialLikelihood { number_of_trials: 10, ..Default::default() };
    /// let fit = model.fit(&[3.0, 5.0, 2.0, 4.0, 6.0])?;
    /// // p̂ = 20 / (5·10) = 0.4.
    /// assert!((fit.params()[0] - 0.4).abs() < 1e-12, "p_hat was {}", fit.params()[0]);
    /// assert!(fit.converged());
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    pub fn fit(&self, data: &[f64]) -> Result<MleFit> {
        if data.is_empty() {
            return Err(Error::InsufficientData);
        }
        let n = self.trials().ok_or_else(|| {
            Error::InvalidInput("number_of_trials must be a positive integer".to_owned())
        })?;
        let n_f = count_to_f64(n);
        let mut sum_x = 0.0;
        for &x in data {
            let k = success_count(x, n).ok_or_else(|| {
                Error::InvalidInput(format!(
                    "observation {x} is not an integer success count in [0, {n}]"
                ))
            })?;
            sum_x += count_to_f64(k);
        }
        let total = count_to_f64(data.len()) * n_f;
        let p_hat = sum_x / total;
        if !(p_hat > 0.0 && p_hat < 1.0) {
            return Err(Error::DegenerateInput(format!(
                "all observations on the boundary (p_hat = {p_hat}); \
                 the binomial log-likelihood is degenerate"
            )));
        }
        let log_likelihood = self.log_likelihood(&[p_hat], data);
        Ok(MleFit::from_closed_form(
            vec![p_hat],
            log_likelihood,
            data.len(),
        ))
    }

    /// Returns this model's number of trials as a positive `usize`, or `None`
    /// when the stored `number_of_trials` is zero or negative (an unusable
    /// model).
    fn trials(&self) -> Option<usize> {
        usize::try_from(self.number_of_trials)
            .ok()
            .filter(|&n| n > 0)
    }
}

impl LogLikelihood for BinomialLikelihood {
    /// The binomial has a single free parameter, the success probability `p`.
    fn n_params(&self) -> usize {
        1
    }

    /// Evaluates `ℓ(p; data)` for the success probability `params[0]`.
    ///
    /// Returns [`f64::NEG_INFINITY`] outside the valid domain: when
    /// `number_of_trials` is not positive, `p ∉ (0, 1)`, or any observation is
    /// negative, non-integral, or exceeds `number_of_trials`.
    fn log_likelihood(&self, params: &[f64], data: &[f64]) -> f64 {
        let Some(&p) = params.first() else {
            return f64::NEG_INFINITY;
        };
        let Some(n) = self.trials() else {
            return f64::NEG_INFINITY;
        };
        if !(p > 0.0 && p < 1.0) {
            return f64::NEG_INFINITY;
        }
        let ln_p = p.ln();
        let ln_q = (1.0 - p).ln();
        let n_f = count_to_f64(n);
        let mut sum = 0.0;
        for &x in data {
            let Some(k) = success_count(x, n) else {
                return f64::NEG_INFINITY;
            };
            let k_f = count_to_f64(k);
            // ln C(n, k) + k ln p + (n − k) ln(1 − p).
            sum += ln_choose(n, k) + k_f.mul_add(ln_p, (n_f - k_f) * ln_q);
        }
        sum
    }
}

/// Validates and converts one observation to an integer success count.
///
/// # Arguments
///
/// * `x` — a candidate observation.
/// * `n_trials` — the binomial trial count; the returned count must not exceed
///   it.
///
/// # Returns
///
/// `Some(k)` when `x` is a finite, non-negative integer (within
/// [`INTEGRALITY_TOL`]) no greater than `n_trials`; `None` for negative,
/// non-integral, non-finite, or out-of-range `x`.
fn success_count(x: f64, n_trials: usize) -> Option<usize> {
    if !x.is_finite() || x < 0.0 {
        return None;
    }
    let rounded = x.round();
    if (x - rounded).abs() > INTEGRALITY_TOL {
        return None;
    }
    let k = f64_integer_to_usize(rounded)?;
    (k <= n_trials).then_some(k)
}

/// Recovers the `usize` value of a non-negative, exactly integral `f64` without
/// an `as` cast (which the crate's style gate forbids in `src/`).
///
/// # Arguments
///
/// * `x` — a non-negative `f64` that is an exact integer (e.g. the output of
///   [`f64::round`]).
///
/// # Returns
///
/// `Some(k)` such that `count_to_f64(k) == x`, or `None` if `x` is negative or
/// lies beyond the [`FLOOR_SEARCH_TOP_BIT`] search range.
fn f64_integer_to_usize(x: f64) -> Option<usize> {
    if x < 0.0 {
        return None;
    }
    let mut acc: usize = 0;
    let mut bit: usize = FLOOR_SEARCH_TOP_BIT;
    while bit > 0 {
        let candidate = acc | bit;
        if count_to_f64(candidate) <= x {
            acc = candidate;
        }
        bit >>= 1;
    }
    // `x` is an exact integer, so an in-range search reproduces it exactly; a
    // half-unit tolerance both confirms the match and rejects out-of-range `x`.
    ((count_to_f64(acc) - x).abs() < 0.5).then_some(acc)
}

#[cfg(test)]
mod tests;
