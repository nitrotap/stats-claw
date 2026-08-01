//! Categorical (multinomial-per-observation) likelihood, layered onto the
//! [`CategoricalLikelihood`](crate::likelihood::CategoricalLikelihood).
//!
//! Models `n` i.i.d. draws from a `Categorical(p₀ … p_{k−1})` distribution whose
//! observations are category indices (`0 ≤ j < k`) stored as `f64`. The
//! log-likelihood of a sample is `Σⱼ countⱼ · ln pⱼ`, and the closed-form
//! maximum-likelihood estimate is the empirical frequency `p̂ⱼ = countⱼ / n`.
//!
//! # `n_categories` as an explicit argument
//!
//! The number of categories `k` is the model's structural dimension, not a
//! fitted quantity, so the inherent methods take it as an explicit
//! `n_categories` argument (mirroring the binomial `trials` parameter). The
//! struct's own `number_of_categories` field is descriptive metadata and
//! is deliberately not consulted, keeping the numerics independent of how the
//! struct was populated. The [`LogLikelihood`] trait — whose signature is frozen
//! and carries no `k` — is implemented on the small wrapper
//! [`CategoricalLikelihoodModel`], which stores `k` so that
//! [`n_params`](LogLikelihood::n_params) can report it.
//!
//! # Simplex and zero-count conventions
//!
//! A parameter vector must sum to `1` within [`SIMPLEX_TOLERANCE`] to be a valid
//! probability vector; otherwise the log-likelihood is
//! [`f64::NEG_INFINITY`]. Empty categories are permitted: a category with zero
//! observed count contributes `0` to the log-likelihood under the convention
//! `0 · ln 0 ≔ 0`, so a fitted `p̂ⱼ = 0` does not make the fitted
//! log-likelihood degenerate.
//!
//! # Simplex constraint and the log-odds escape hatch
//!
//! [`CategoricalLikelihoodModel`]'s parameters live on the probability simplex
//! `Σ pⱼ = 1, pⱼ ≥ 0`, a constraint the unconstrained L-BFGS optimizer behind
//! [`fit_mle`](crate::likelihood::fit_mle) cannot respect: its iterates leave the
//! valid domain, so fitting the *simplex* parameterization directly is not
//! meaningful. The escape hatch is [`CategoricalLogOdds`], which reparametrizes
//! the family by `k − 1` free real logits (softmax back to probabilities); it is
//! unconstrained, fits cleanly through [`fit_mle`](crate::likelihood::fit_mle), and its
//! [`fit_mle`](crate::likelihood::fit_mle) estimate matches this module's
//! closed-form `p̂` after softmax.
//!
//! # Examples
//!
//! ```
//! use stats_claw::likelihood::CategoricalLikelihood;
//!
//! let model = CategoricalLikelihood::default();
//! // Six draws over k = 3 categories; MLE is the empirical frequency.
//! let fit = model.fit(3, &[0.0, 0.0, 1.0, 2.0, 2.0, 2.0])?;
//! assert!((fit.params()[0] - 2.0 / 6.0).abs() < 1e-12, "p0 was {}", fit.params()[0]);
//! # Ok::<(), stats_claw::error::Error>(())
//! ```

use crate::algorithms::count_to_f64;
use crate::error::{Error, Result};
use crate::likelihood::{LogLikelihood, MleFit};

/// Maximum absolute deviation of `Σ pⱼ` from `1` for a valid probability vector.
///
/// A small finite tolerance absorbs the floating-point rounding of an
/// otherwise-normalized vector (e.g. the sum of closed-form empirical
/// frequencies) without admitting genuinely unnormalized input.
pub const SIMPLEX_TOLERANCE: f64 = 1e-9;

/// Maximum absolute distance a datum may sit from the nearest integer and still
/// be accepted as a category index. Observations are category labels, so they
/// must be (floating-point representations of) non-negative integers below `k`.
const INDEX_TOLERANCE: f64 = 1e-9;

/// A [`LogLikelihood`] wrapper carrying the category count `k` for the
/// categorical family.
///
/// The frozen [`LogLikelihood`] trait exposes no place for the structural
/// dimension `k`, so this small struct stores it and reports it from
/// [`n_params`](LogLikelihood::n_params). Construct it directly with the number
/// of categories.
///
/// # Examples
///
/// ```
/// use stats_claw::likelihood::LogLikelihood;
/// use stats_claw::likelihood::categorical::CategoricalLikelihoodModel;
///
/// let model = CategoricalLikelihoodModel { n_categories: 3 };
/// assert_eq!(model.n_params(), 3);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CategoricalLikelihoodModel {
    /// The number of categories `k`, i.e. the required parameter-vector length.
    pub n_categories: usize,
}

impl LogLikelihood for CategoricalLikelihoodModel {
    fn n_params(&self) -> usize {
        self.n_categories
    }

    fn log_likelihood(&self, params: &[f64], data: &[f64]) -> f64 {
        log_likelihood_core(self.n_categories, params, data)
    }
}

/// Returns whether `x` is a valid category index for `k` categories, i.e. a
/// non-negative integer (within [`INDEX_TOLERANCE`]) strictly below `k`.
fn is_valid_index(x: f64, k: usize) -> bool {
    x.is_finite() && x >= 0.0 && (x - x.round()).abs() < INDEX_TOLERANCE && x < count_to_f64(k)
}

/// Counts how many observations in `data` fall in category `j`.
///
/// Indices are integers, so an observation belongs to category `j` when it lies
/// within half a unit of `j`. Callers validate index range separately; this only
/// tallies membership, returning the count directly as `f64` for the
/// log-likelihood sum.
fn category_count(data: &[f64], j: usize) -> f64 {
    let target = count_to_f64(j);
    let count = data.iter().filter(|&&x| (x - target).abs() < 0.5).count();
    count_to_f64(count)
}

/// Computes `Σⱼ countⱼ · ln pⱼ`, the categorical log-likelihood.
///
/// Returns [`f64::NEG_INFINITY`] when `params` is not length `n_categories`, does
/// not sum to `1` within [`SIMPLEX_TOLERANCE`], contains a non-positive
/// probability for a non-empty category, or when `data` holds an index outside
/// `0 ≤ j < n_categories`. Empty categories contribute `0` under the convention
/// `0 · ln 0 ≔ 0`.
fn log_likelihood_core(n_categories: usize, params: &[f64], data: &[f64]) -> f64 {
    if params.len() != n_categories {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = params.iter().sum();
    if (sum - 1.0).abs() > SIMPLEX_TOLERANCE {
        return f64::NEG_INFINITY;
    }
    if !data.iter().all(|&x| is_valid_index(x, n_categories)) {
        return f64::NEG_INFINITY;
    }
    let mut total = 0.0;
    for (j, &p) in params.iter().enumerate() {
        let count = category_count(data, j);
        if count > 0.0 {
            if p <= 0.0 {
                return f64::NEG_INFINITY;
            }
            total = count.mul_add(p.ln(), total);
        }
        // A zero-count category contributes 0 (the 0·ln0 ≔ 0 convention), so it
        // is skipped even when p ≤ 0.
    }
    total
}

impl crate::likelihood::CategoricalLikelihood {
    /// Evaluates the categorical log-likelihood `Σⱼ countⱼ · ln pⱼ` of `data`
    /// under the probability vector `params`.
    ///
    /// The number of categories `k` is passed explicitly rather than read from
    /// the struct (see the [module docs](self)); the frozen [`LogLikelihood`]
    /// trait is instead implemented on [`CategoricalLikelihoodModel`].
    ///
    /// # Arguments
    ///
    /// * `n_categories` — the number of categories `k`.
    /// * `params` — the probability vector `[p₀ … p_{k−1}]`; length must equal
    ///   `n_categories` and the entries must sum to `1` within
    ///   [`SIMPLEX_TOLERANCE`].
    /// * `data` — the observed category indices, each a non-negative integer
    ///   below `k` stored as `f64`.
    ///
    /// # Returns
    ///
    /// The log-likelihood, or [`f64::NEG_INFINITY`] when `params` has the wrong
    /// length, is not normalized, assigns a non-positive probability to a
    /// non-empty category, or `data` contains an invalid index. A category with
    /// zero observed count contributes `0` (the `0 · ln 0 ≔ 0` convention).
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::likelihood::CategoricalLikelihood;
    ///
    /// let model = CategoricalLikelihood::default();
    /// // counts = [2, 1, 3] over k = 3; ℓ = 2·ln0.2 + 1·ln0.3 + 3·ln0.5.
    /// let ll = model.log_likelihood(3, &[0.2, 0.3, 0.5], &[0.0, 0.0, 1.0, 2.0, 2.0, 2.0]);
    /// assert!((ll - (-6.502_290_170_873_972)).abs() < 1e-10, "ll was {ll}");
    /// ```
    #[must_use]
    pub fn log_likelihood(&self, n_categories: usize, params: &[f64], data: &[f64]) -> f64 {
        log_likelihood_core(n_categories, params, data)
    }

    /// Closed-form maximum-likelihood fit of the category probabilities,
    /// `p̂ⱼ = countⱼ / n`.
    ///
    /// # Arguments
    ///
    /// * `n_categories` — the number of categories `k`.
    /// * `data` — the observed category indices, each a non-negative integer
    ///   below `k` stored as `f64`.
    ///
    /// # Returns
    ///
    /// An [`MleFit`] whose parameters are the empirical frequencies (length
    /// `n_categories`, summing to `1`) and whose log-likelihood is `ℓ(p̂; data)`.
    /// Categories with zero observed count are allowed and receive `p̂ⱼ = 0`,
    /// contributing `0` to the log-likelihood.
    ///
    /// # Errors
    ///
    /// * [`Error::InsufficientData`] if `data` is empty.
    /// * [`Error::InvalidInput`] if any datum is not a non-negative integer
    ///   below `n_categories`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::likelihood::CategoricalLikelihood;
    ///
    /// let model = CategoricalLikelihood::default();
    /// let fit = model.fit(3, &[0.0, 0.0, 1.0, 2.0, 2.0, 2.0])?;
    /// assert!((fit.params()[2] - 0.5).abs() < 1e-12, "p2 was {}", fit.params()[2]);
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    pub fn fit(&self, n_categories: usize, data: &[f64]) -> Result<MleFit> {
        if data.is_empty() {
            return Err(Error::InsufficientData);
        }
        if let Some(&bad) = data.iter().find(|&&x| !is_valid_index(x, n_categories)) {
            return Err(Error::InvalidInput(format!(
                "datum {bad} is not a non-negative integer below {n_categories} categories"
            )));
        }
        let n = count_to_f64(data.len());
        let params: Vec<f64> = (0..n_categories)
            .map(|j| category_count(data, j) / n)
            .collect();
        let log_likelihood = log_likelihood_core(n_categories, &params, data);
        Ok(MleFit::from_closed_form(params, log_likelihood, data.len()))
    }
}

/// Unconstrained log-odds (softmax) reparametrization of the categorical family.
///
/// [`CategoricalLikelihoodModel`] parameterizes the categorical distribution by
/// the probabilities themselves, which live on the simplex `Σ pⱼ = 1, pⱼ ≥ 0` — a
/// constraint the framework's free L-BFGS optimizer cannot honor. This type
/// instead carries `k − 1` *free* real logits `z₁ … z_{k−1}` (category 0 is the
/// reference with logit `z₀ ≔ 0`) and recovers the probabilities through the
/// softmax
/// `pⱼ = exp(zⱼ) / Σᵢ exp(zᵢ)`.
///
/// Because the logits range over all of `ℝᵏ⁻¹` with no constraint, the model can
/// be fed directly to [`fit_mle`](crate::likelihood::fit_mle): the optimizer's
/// iterates are always valid and the softmax maps them back onto the simplex.
/// Use [`probabilities`](CategoricalLogOdds::probabilities) to read the fitted
/// probabilities and [`from_probabilities`](CategoricalLogOdds::from_probabilities)
/// to build an initial logit vector from a probability guess.
///
/// # Numerical stability
///
/// All probabilities are computed in log-space via the log-sum-exp identity
/// `ln Σᵢ exp(zᵢ) = m + ln Σᵢ exp(zᵢ − m)` with `m = maxᵢ zᵢ`, so no term ever
/// overflows even for large logits (each `exp(zᵢ − m) ≤ 1`).
///
/// # Examples
///
/// ```
/// use stats_claw::likelihood::LogLikelihood;
/// use stats_claw::likelihood::categorical::{CategoricalLikelihoodModel, CategoricalLogOdds};
///
/// let p = [0.2, 0.3, 0.5];
/// let data = [0.0, 0.0, 1.0, 2.0, 2.0, 2.0];
/// // The log-odds model reproduces the simplex model's log-likelihood exactly.
/// let z = CategoricalLogOdds::from_probabilities(&p)?;
/// let via_logodds = CategoricalLogOdds { n_categories: 3 }.log_likelihood(&z, &data);
/// let via_simplex = CategoricalLikelihoodModel { n_categories: 3 }.log_likelihood(&p, &data);
/// assert!((via_logodds - via_simplex).abs() < 1e-12);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CategoricalLogOdds {
    /// The number of categories `k`; the free-parameter count is `k − 1`.
    pub n_categories: usize,
}

impl CategoricalLogOdds {
    /// Computes the log-probabilities `[ln p₀ … ln p_{k−1}]` from the free logits.
    ///
    /// The full logit vector pins the reference category at `z₀ = 0` and appends
    /// the `k − 1` free logits; the normalizer is evaluated with log-sum-exp for
    /// stability.
    ///
    /// # Arguments
    ///
    /// * `params` — the `k − 1` free logits `z₁ … z_{k−1}`.
    ///
    /// # Returns
    ///
    /// `Some([ln p₀ … ln p_{k−1}])` (length `k`), or `None` when `k == 0`, when
    /// `params.len() != k − 1`, or when any logit is non-finite.
    fn log_probabilities(self, params: &[f64]) -> Option<Vec<f64>> {
        let k = self.n_categories;
        if k == 0 || params.len() + 1 != k {
            return None;
        }
        if !params.iter().all(|z| z.is_finite()) {
            return None;
        }
        // Full logits: the reference category 0 pinned at 0, then the free logits.
        let mut logits = Vec::with_capacity(k);
        logits.push(0.0_f64);
        logits.extend_from_slice(params);
        // log-sum-exp normalizer: m + ln Σ exp(zᵢ − m). Subtracting the max keeps
        // every exponential in (0, 1], so nothing overflows.
        let m = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = logits.iter().map(|z| (z - m).exp()).sum();
        let lse = m + sum_exp.ln();
        Some(logits.iter().map(|z| z - lse).collect())
    }

    /// Converts the free logits `params` into the probability vector `[p₀ … p_{k−1}]`.
    ///
    /// This is the softmax `pⱼ = exp(zⱼ) / Σᵢ exp(zᵢ)` (with `z₀ ≔ 0`), the
    /// inverse of [`from_probabilities`](CategoricalLogOdds::from_probabilities).
    ///
    /// # Arguments
    ///
    /// * `params` — the `k − 1` free logits `z₁ … z_{k−1}`.
    ///
    /// # Returns
    ///
    /// The probability vector `[p₀ … p_{k−1}]`, which sums to `1`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] when `params` does not have length `k − 1` (or
    /// `k == 0`), or when any logit is non-finite.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::likelihood::categorical::CategoricalLogOdds;
    ///
    /// let model = CategoricalLogOdds { n_categories: 3 };
    /// // Equal logits give the uniform distribution.
    /// let p = model.probabilities(&[0.0, 0.0])?;
    /// assert!((p[0] - 1.0 / 3.0).abs() < 1e-12, "p0 was {}", p[0]);
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    pub fn probabilities(&self, params: &[f64]) -> Result<Vec<f64>> {
        self.log_probabilities(params)
            .map(|log_p| log_p.iter().map(|l| l.exp()).collect())
            .ok_or_else(|| {
                Error::InvalidInput(format!(
                    "params must be {} finite logits for {} categories",
                    self.n_categories.saturating_sub(1),
                    self.n_categories
                ))
            })
    }

    /// Builds a free-logit vector from a probability vector `p`, `zⱼ = ln(pⱼ / p₀)`.
    ///
    /// This is the inverse of [`probabilities`](CategoricalLogOdds::probabilities):
    /// it maps an interior simplex point to the `k − 1` free logits, taking
    /// category 0 as the reference. It is the natural way to seed
    /// [`fit_mle`](crate::likelihood::fit_mle) from a probability guess.
    ///
    /// # Arguments
    ///
    /// * `p` — an interior probability vector `[p₀ … p_{k−1}]`: non-empty, every
    ///   entry finite and strictly positive, summing to `1` within
    ///   [`SIMPLEX_TOLERANCE`].
    ///
    /// # Returns
    ///
    /// The `k − 1` free logits `[z₁ … z_{k−1}]` (empty when `k == 1`).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidInput`] when `p` is empty, when any entry (in particular
    /// the reference `p₀`) is non-finite or not strictly positive, or when the
    /// entries do not sum to `1` within [`SIMPLEX_TOLERANCE`]. Strict positivity
    /// is required because `ln(pⱼ / p₀)` is finite only on the open simplex.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::likelihood::categorical::CategoricalLogOdds;
    ///
    /// let z = CategoricalLogOdds::from_probabilities(&[0.2, 0.3, 0.5])?;
    /// // z₁ = ln(0.3 / 0.2) = ln 1.5.
    /// assert!((z[0] - 1.5_f64.ln()).abs() < 1e-12, "z1 was {}", z[0]);
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    pub fn from_probabilities(p: &[f64]) -> Result<Vec<f64>> {
        let Some((&p0, rest)) = p.split_first() else {
            return Err(Error::InvalidInput(
                "probability vector must be non-empty".to_owned(),
            ));
        };
        if !p.iter().all(|&pj| pj.is_finite() && pj > 0.0) {
            return Err(Error::InvalidInput(format!(
                "probabilities must be finite and strictly positive for log-odds, p0 was {p0}"
            )));
        }
        let sum: f64 = p.iter().sum();
        if (sum - 1.0).abs() > SIMPLEX_TOLERANCE {
            return Err(Error::InvalidInput(format!(
                "probabilities must sum to 1 within tolerance, sum was {sum}"
            )));
        }
        let ln_p0 = p0.ln();
        Ok(rest.iter().map(|&pj| pj.ln() - ln_p0).collect())
    }
}

impl LogLikelihood for CategoricalLogOdds {
    /// Returns `k − 1`, the number of free logits (`0` when `k ≤ 1`).
    fn n_params(&self) -> usize {
        self.n_categories.saturating_sub(1)
    }

    /// Evaluates `Σⱼ countⱼ · ln pⱼ(z)`, the categorical log-likelihood under the
    /// softmax of the free logits `params`.
    ///
    /// # Arguments
    ///
    /// * `params` — the `k − 1` free logits `z₁ … z_{k−1}`.
    /// * `data` — the observed category indices, each a non-negative integer
    ///   below `k` stored as `f64`.
    ///
    /// # Returns
    ///
    /// The log-likelihood, or [`f64::NEG_INFINITY`] when `params` has the wrong
    /// length or a non-finite logit, or when `data` holds an out-of-range or
    /// non-finite index (the same data-validation rules as
    /// [`CategoricalLikelihoodModel`], per the [`LogLikelihood`] contract). Every
    /// `pⱼ` is strictly positive here (finite logits), so `ln pⱼ` is always
    /// finite; a zero-count category still contributes `0` (the `0 · ln 0 ≔ 0`
    /// convention).
    fn log_likelihood(&self, params: &[f64], data: &[f64]) -> f64 {
        let Some(log_p) = self.log_probabilities(params) else {
            return f64::NEG_INFINITY;
        };
        let k = self.n_categories;
        if !data.iter().all(|&x| is_valid_index(x, k)) {
            return f64::NEG_INFINITY;
        }
        let mut total = 0.0;
        for (j, &log_pj) in log_p.iter().enumerate() {
            let count = category_count(data, j);
            // 0·ln0 ≔ 0: an unobserved category adds nothing (log_pj is finite).
            if count > 0.0 {
                total = count.mul_add(log_pj, total);
            }
        }
        total
    }
}

#[cfg(test)]
mod tests;
