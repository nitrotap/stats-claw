//! Deterministic jackknife (leave-one-out) resampling.
//!
//! The jackknife recomputes a statistic on each of the `n` subsamples that omit
//! one observation, then combines the resulting replicates into an estimate of
//! the statistic's bias and standard error. Unlike the bootstrap it draws no
//! random numbers, so [`jackknife_statistic`] is fully deterministic for a given
//! input.
//!
//! # Examples
//!
//! ```
//! use stats_claw::resampling::jackknife_indices;
//!
//! // Three leave-one-out index sets, each omitting one position in order.
//! assert_eq!(jackknife_indices(3)?, vec![vec![1, 2], vec![0, 2], vec![0, 1]]);
//! # Ok::<(), stats_claw::error::Error>(())
//! ```

use crate::error::{Error, Result};
use crate::numeric::count_to_f64;
use crate::resampling::JackknifeResampling;

/// The jackknife bias and standard-error estimate for a statistic.
///
/// Bundles the statistic evaluated on the full sample together with the
/// leave-one-out replicates and the two classical jackknife summaries computed
/// from them (Efron & Tibshirani, *An Introduction to the Bootstrap*, 1993,
/// §10.2). The fields are private (the struct owns a `Vec`, so it fully
/// encapsulates its storage); read them through the [`estimate`](Self::estimate),
/// [`bias`](Self::bias), [`std_error`](Self::std_error), and
/// [`replicates`](Self::replicates) accessors.
///
/// # Examples
///
/// ```
/// use stats_claw::resampling::jackknife_statistic;
///
/// let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// let mean = |s: &[f64]| s.iter().sum::<f64>() / f64::from(u32::try_from(s.len()).unwrap_or(0));
/// let est = jackknife_statistic(&data, mean)?;
/// assert!((est.estimate() - 5.0).abs() < 1e-12, "estimate was {}", est.estimate());
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct JackknifeEstimate {
    /// The statistic evaluated on the full sample.
    estimate: f64,
    /// Jackknife bias estimate, `(n - 1) * (mean(replicates) - estimate)`.
    bias: f64,
    /// Jackknife standard error,
    /// `sqrt((n - 1) / n * Σ (replicate_i - mean(replicates))²)`.
    std_error: f64,
    /// The statistic evaluated on each leave-one-out subsample, in index order.
    replicates: Vec<f64>,
}

impl JackknifeEstimate {
    /// Returns the statistic evaluated on the full sample.
    ///
    /// # Returns
    ///
    /// The full-sample estimate `stat(data)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::resampling::jackknife_statistic;
    ///
    /// let data = [1.0, 2.0, 3.0];
    /// let est = jackknife_statistic(&data, |s| s.iter().copied().fold(f64::MIN, f64::max))?;
    /// assert!((est.estimate() - 3.0).abs() < 1e-12, "estimate was {}", est.estimate());
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    #[must_use]
    pub const fn estimate(&self) -> f64 {
        self.estimate
    }

    /// Returns the jackknife bias estimate.
    ///
    /// # Returns
    ///
    /// `(n - 1) * (mean(replicates) - estimate)`; subtract it from
    /// [`estimate`](Self::estimate) for the bias-corrected value.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::resampling::jackknife_statistic;
    ///
    /// let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    /// let mean = |s: &[f64]| s.iter().sum::<f64>() / f64::from(u32::try_from(s.len()).unwrap_or(0));
    /// let est = jackknife_statistic(&data, mean)?;
    /// // The jackknife bias of the mean is identically zero.
    /// assert!(est.bias().abs() < 1e-12, "bias was {}", est.bias());
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    #[must_use]
    pub const fn bias(&self) -> f64 {
        self.bias
    }

    /// Returns the jackknife standard error.
    ///
    /// # Returns
    ///
    /// `sqrt((n - 1) / n * Σ (replicate_i - mean(replicates))²)`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::resampling::jackknife_statistic;
    ///
    /// let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    /// let mean = |s: &[f64]| s.iter().sum::<f64>() / f64::from(u32::try_from(s.len()).unwrap_or(0));
    /// let est = jackknife_statistic(&data, mean)?;
    /// // Jackknife SE of the mean equals the classic sd(ddof=1)/sqrt(n).
    /// assert!((est.std_error() - 0.755_928_946_018_454_4).abs() < 1e-12, "se was {}", est.std_error());
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    #[must_use]
    pub const fn std_error(&self) -> f64 {
        self.std_error
    }

    /// Returns the per-subsample replicates.
    ///
    /// # Returns
    ///
    /// A borrowed slice of the statistic evaluated on each leave-one-out
    /// subsample, in index order; its length equals the input sample size.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::resampling::jackknife_statistic;
    ///
    /// let data = [1.0, 2.0, 3.0, 4.0];
    /// let est = jackknife_statistic(&data, |s| s.iter().sum::<f64>())?;
    /// assert_eq!(est.replicates().len(), data.len(), "one replicate per observation");
    /// // Leaving out the largest value yields the smallest sum.
    /// assert!((est.replicates()[3] - 6.0).abs() < 1e-12, "was {}", est.replicates()[3]);
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    #[must_use]
    pub fn replicates(&self) -> &[f64] {
        &self.replicates
    }
}

/// Builds the leave-one-out index sets for a sample of size `n`.
///
/// Returns `n` index vectors; set `i` contains every index in `0..n` except `i`,
/// in ascending order. These are the subsamples the jackknife evaluates a
/// statistic on.
///
/// # Arguments
///
/// * `n` — the sample size; must be `>= 2` (a jackknife needs at least two
///   observations to leave one out and still have data).
///
/// # Returns
///
/// A length-`n` vector whose `i`-th entry lists the `n - 1` indices `0..n` with
/// `i` removed.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] when `n < 2`.
///
/// # Examples
///
/// ```
/// use stats_claw::resampling::jackknife_indices;
///
/// assert_eq!(jackknife_indices(3)?, vec![vec![1, 2], vec![0, 2], vec![0, 1]]);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn jackknife_indices(n: usize) -> Result<Vec<Vec<usize>>> {
    if n < 2 {
        return Err(Error::InsufficientData);
    }
    Ok((0..n)
        .map(|i| (0..n).filter(|&j| j != i).collect())
        .collect())
}

/// Computes the jackknife bias and standard error of `stat` over `data`.
///
/// Evaluates `stat` on the full sample and on each leave-one-out subsample, then
/// combines the replicates into the classical jackknife summaries (Efron &
/// Tibshirani, 1993, §10.2). Fully deterministic — no random numbers are drawn.
///
/// # Arguments
///
/// * `data` — the observed sample; must contain at least two values.
/// * `stat` — the statistic to jackknife, mapping a sample view to a scalar.
///
/// # Returns
///
/// A [`JackknifeEstimate`] holding the full-sample estimate, the bias and
/// standard-error estimates, and the per-subsample replicates.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] when `data` has fewer than two elements.
///
/// # Examples
///
/// ```
/// use stats_claw::resampling::jackknife_statistic;
///
/// let data = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
/// let mean = |s: &[f64]| s.iter().sum::<f64>() / f64::from(u32::try_from(s.len()).unwrap_or(0));
/// let est = jackknife_statistic(&data, mean)?;
/// assert!((est.estimate() - 5.0).abs() < 1e-12, "estimate was {}", est.estimate());
/// // Jackknife SE of the mean equals the classic sd(ddof=1)/sqrt(n).
/// assert!((est.std_error() - 0.755_928_946_018_454_4).abs() < 1e-12, "se was {}", est.std_error());
/// assert_eq!(est.replicates().len(), data.len(), "one replicate per observation");
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn jackknife_statistic(
    data: &[f64],
    stat: impl Fn(&[f64]) -> f64,
) -> Result<JackknifeEstimate> {
    let n = data.len();
    if n < 2 {
        return Err(Error::InsufficientData);
    }
    let estimate = stat(data);
    let mut replicates = Vec::with_capacity(n);
    for i in 0..n {
        let sample: Vec<f64> = data
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(_, &value)| value)
            .collect();
        replicates.push(stat(&sample));
    }
    let n_f = count_to_f64(n);
    // `replicates.len() == n`, so this shared mean matches the old inline
    // `sum / n_f` bit-for-bit; the jackknife bias/SE below keep their own
    // (non-`sample_variance`) `(n-1)/n` scaling.
    let mean_rep = crate::numeric::mean(&replicates);
    let bias = (n_f - 1.0) * (mean_rep - estimate);
    let sum_sq: f64 = replicates
        .iter()
        .map(|&r| {
            let deviation = r - mean_rep;
            deviation * deviation
        })
        .sum();
    let std_error = ((n_f - 1.0) / n_f * sum_sq).sqrt();
    Ok(JackknifeEstimate {
        estimate,
        bias,
        std_error,
        replicates,
    })
}

impl JackknifeResampling {
    /// Computes the jackknife bias and standard error of `stat` over `data`.
    ///
    /// Convenience wrapper that lets the [`JackknifeResampling`] scheme
    /// drive the numerics directly; it delegates to [`jackknife_statistic`] and
    /// shares its contract exactly.
    ///
    /// # Arguments
    ///
    /// * `data` — the observed sample; must contain at least two values.
    /// * `stat` — the statistic to jackknife, mapping a sample view to a scalar.
    ///
    /// # Returns
    ///
    /// A [`JackknifeEstimate`] with the full-sample estimate, bias,
    /// standard error, and per-subsample replicates.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InsufficientData`] when `data` has fewer than two
    /// elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::resampling::JackknifeResampling;
    ///
    /// let scheme = JackknifeResampling::default();
    /// let data = [1.0, 2.0, 3.0, 4.0];
    /// let est = scheme.estimate(&data, |s| s.iter().copied().fold(f64::MIN, f64::max))?;
    /// assert!((est.estimate() - 4.0).abs() < 1e-12, "max was {}", est.estimate());
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    pub fn estimate(
        &self,
        data: &[f64],
        stat: impl Fn(&[f64]) -> f64,
    ) -> Result<JackknifeEstimate> {
        jackknife_statistic(data, stat)
    }
}

/// Kani formal-verification harnesses for the deterministic jackknife index
/// construction.
///
/// [`jackknife_indices`] draws no random numbers, so these prove its
/// input-validation and index-safety properties over symbolic and small-fixed
/// sizes rather than the sampled fixtures the `#[cfg(test)]` suite uses. Compiled
/// only under `cargo kani` (behind `#[cfg(kani)]`); invisible to normal
/// build/test/clippy. Run e.g. with
/// `cargo kani -Z stubbing -p stats-claw --harness resampling_jackknife_rejects_small_n`.
#[cfg(kani)]
mod verification {
    use super::{Error, jackknife_indices};

    /// Proves the input-validation path: for *every* symbolic `n < 2`,
    /// [`jackknife_indices`] returns [`Error::InsufficientData`] and never panics —
    /// a single observation has no held-out complement.
    ///
    /// The `#[kani::unwind(2)]` bound caps the (unreachable-on-feasible-paths)
    /// index-building loop: with `n < 2` the guard returns before it, and CBMC
    /// discharges the over-unwinding of the infeasible `n >= 2` branch vacuously.
    #[kani::proof]
    #[kani::unwind(2)]
    fn resampling_jackknife_rejects_small_n() {
        let n: usize = kani::any();
        kani::assume(n < 2);
        let result = jackknife_indices(n);
        assert!(
            matches!(result, Err(Error::InsufficientData)),
            "n < 2 must be rejected with InsufficientData"
        );
    }

    /// Proves the leave-one-out index sets are total and in bounds for `n = 3`:
    /// exactly `n` sets, each of size `n - 1`, every index in `0..n`, and set `i`
    /// omitting its own index `i`. This is the index-safety guarantee the jackknife
    /// statistic loop relies on when it views each subsample.
    #[kani::proof]
    #[kani::unwind(5)]
    fn resampling_jackknife_indices_in_bounds() {
        const N: usize = 3;
        let result = jackknife_indices(N);
        assert!(result.is_ok(), "n >= 2 must produce leave-one-out sets");
        if let Ok(sets) = result {
            assert!(sets.len() == N, "expected one set per observation");
            for (i, set) in sets.iter().enumerate() {
                assert!(set.len() == N - 1, "each set must omit exactly one index");
                for &j in set {
                    assert!(j < N, "jackknife index {j} escaped 0..N");
                    assert!(j != i, "set {i} must not contain its own index");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
