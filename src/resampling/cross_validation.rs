//! k-fold cross-validation evaluator for the
//! [`CrossValidation`].
//!
//! The raw fold-index partition is produced by
//! [`kfold_indices`]; this module adds the
//! evaluation loop that runs a user `fit_score` closure over every
//! `(train, test)` split and aggregates the per-fold scores into a mean and a
//! standard error. Every split is driven by the deterministic
//! [`SplitMix64`] PRNG, so a fixed seed reproduces the
//! folds — and therefore the scores — bit-for-bit.

use super::schemes::kfold_indices;
use crate::error::{Error, Result};
use crate::numeric::count_to_f64;
use crate::resampling::CrossValidation;
use crate::rng::SplitMix64;

/// Aggregated cross-validation scores over the `k` folds.
///
/// Produced by [`cross_validate`] (and [`CrossValidation::run`]): the raw per-fold
/// scores together with their mean and standard error. The fields are private;
/// read them through the [`fold_scores`](Self::fold_scores), [`mean`](Self::mean),
/// and [`std_error`](Self::std_error) accessors.
///
/// # Examples
///
/// ```
/// use stats_claw::resampling::cross_validate;
/// use stats_claw::rng::SplitMix64;
///
/// let mut rng = SplitMix64::new(1);
/// let scores = cross_validate(12, 4, &mut rng, |_train, _test| 0.5)?;
/// assert_eq!(scores.fold_scores(), &[0.5; 4]);
/// assert!((scores.mean() - 0.5).abs() < 1e-12);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CvScores {
    /// The score returned by `fit_score` for each fold, in fold order.
    fold_scores: Vec<f64>,
    /// The arithmetic mean of the per-fold scores.
    mean: f64,
    /// The standard error of the mean: `sd(fold_scores, ddof=1) / sqrt(k)`.
    std_error: f64,
}

impl CvScores {
    /// Aggregates raw per-fold scores into a [`CvScores`], computing the mean and
    /// standard error once.
    ///
    /// This is the single aggregation site shared by k-fold [`cross_validate`] and
    /// leave-one-out [`loo_cross_validate`](super::loocv::loo_cross_validate), so
    /// both report identically-defined summaries. The standard error is
    /// `sd(fold_scores, ddof=1) / sqrt(k)`.
    ///
    /// # Arguments
    ///
    /// * `scores` — the per-fold scores; callers guarantee at least two entries so
    ///   the `ddof = 1` variance denominator (`k − 1`) is `>= 1`.
    ///
    /// # Returns
    ///
    /// A [`CvScores`] holding `scores` alongside their mean and standard error.
    pub(crate) fn new(scores: Vec<f64>) -> Self {
        let kf = count_to_f64(scores.len());
        // Callers guarantee k >= 2, so the mean and the ddof=1 variance
        // (`kf - 1 >= 1`) are both well defined. Shared helpers keep this
        // aggregation identical to the Monte-Carlo estimator's.
        let mean = crate::numeric::mean(&scores);
        let variance = crate::numeric::sample_variance(&scores);
        let std_error = variance.sqrt() / kf.sqrt();
        Self {
            fold_scores: scores,
            mean,
            std_error,
        }
    }

    /// Returns the per-fold scores, in fold order.
    ///
    /// # Returns
    ///
    /// A borrowed slice of the `k` scores, one per fold, as returned by the
    /// user's `fit_score` closure.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::resampling::cross_validate;
    /// use stats_claw::rng::SplitMix64;
    ///
    /// let mut rng = SplitMix64::new(1);
    /// let scores = cross_validate(9, 3, &mut rng, |_train, _test| 0.7)?;
    /// assert_eq!(scores.fold_scores(), &[0.7, 0.7, 0.7]);
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    #[must_use]
    pub fn fold_scores(&self) -> &[f64] {
        &self.fold_scores
    }

    /// Returns the arithmetic mean of the per-fold scores.
    ///
    /// # Returns
    ///
    /// The mean cross-validated score.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::resampling::cross_validate;
    /// use stats_claw::rng::SplitMix64;
    ///
    /// let mut rng = SplitMix64::new(1);
    /// let scores = cross_validate(9, 3, &mut rng, |_train, _test| 0.7)?;
    /// assert!((scores.mean() - 0.7).abs() < 1e-12);
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    #[must_use]
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// Returns the standard error of the mean: `sd(fold_scores, ddof=1) / sqrt(k)`.
    ///
    /// # Returns
    ///
    /// The standard error of the cross-validated mean; `0.0` when every fold
    /// scored identically.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::resampling::cross_validate;
    /// use stats_claw::rng::SplitMix64;
    ///
    /// let mut rng = SplitMix64::new(1);
    /// let scores = cross_validate(9, 3, &mut rng, |_train, _test| 0.7)?;
    /// assert!(scores.std_error().abs() < 1e-12);
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    #[must_use]
    pub const fn std_error(&self) -> f64 {
        self.std_error
    }
}

/// Runs k-fold cross-validation over `n` observations.
///
/// Partitions `0..n` into `k` folds via [`kfold_indices`], calls
/// `fit_score(train_idx, test_idx)` once per fold, and aggregates the returned
/// scores into their mean and standard error. The folds are drawn from `rng`, so
/// two calls with identically seeded generators produce identical scores.
/// The standard error is `sd(fold_scores, ddof=1) / sqrt(k)`, the
/// usual estimate of the uncertainty in the cross-validated mean.
///
/// # Arguments
///
/// * `n` — number of observations to partition.
/// * `k` — number of folds; must be in `2..=n`.
/// * `rng` — the deterministic generator driving the fold shuffle.
/// * `fit_score` — invoked once per fold as `fit_score(train_idx, test_idx)`,
///   returning that fold's score (e.g. a validation accuracy). Taken as `FnMut`
///   so it may carry mutable state across folds.
///
/// # Returns
///
/// A [`CvScores`] holding the per-fold scores, their mean, and the standard error.
///
/// # Errors
///
/// * [`Error::InvalidInput`] when `k < 2`: a single fold has no validation split,
///   so it is a bad *parameter* rather than a data shortage (matching
///   [`stratified_kfold_indices`](super::stratified::stratified_kfold_indices)).
/// * [`Error::InsufficientData`] when `k > n`: there are fewer observations than
///   requested folds, so [`kfold_indices`] cannot give every fold at least one
///   observation.
///
/// # Examples
///
/// ```
/// use stats_claw::resampling::cross_validate;
/// use stats_claw::rng::SplitMix64;
///
/// let mut rng = SplitMix64::new(42);
/// // A constant scorer: every fold reports 0.9, so the mean is 0.9 with no spread.
/// let scores = cross_validate(20, 5, &mut rng, |_train, _test| 0.9)?;
/// assert_eq!(scores.fold_scores().len(), 5);
/// assert!((scores.mean() - 0.9).abs() < 1e-12);
/// assert!(scores.std_error().abs() < 1e-12);
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn cross_validate(
    n: usize,
    k: usize,
    rng: &mut SplitMix64,
    mut fit_score: impl FnMut(&[usize], &[usize]) -> f64,
) -> Result<CvScores> {
    if k < 2 {
        return Err(Error::InvalidInput("k must be >= 2".to_owned()));
    }
    if k > n {
        return Err(Error::InsufficientData);
    }
    let fold_scores: Vec<f64> = kfold_indices(n, k, rng)
        .iter()
        .map(|(train, test)| fit_score(train, test))
        .collect();
    Ok(CvScores::new(fold_scores))
}

impl CrossValidation {
    /// Runs k-fold cross-validation using this scheme's own configuration.
    ///
    /// Reads the fold count from [`number_of_folds`](Self::number_of_folds) and
    /// seeds the deterministic PRNG from [`random_seed`](Self::random_seed), then
    /// delegates to [`cross_validate`]. The `i64` seed is reinterpreted to `u64`
    /// bit-for-bit via [`i64::cast_unsigned`] (not a numeric `as` cast, which the
    /// `style.rs` guard bans), so a positive seed maps to the same magnitude.
    /// This makes the parameter struct itself executable against the numerics.
    ///
    /// # Arguments
    ///
    /// * `n` — number of observations to partition into folds.
    /// * `fit_score` — invoked once per fold as `fit_score(train_idx, test_idx)`,
    ///   returning that fold's score.
    ///
    /// # Returns
    ///
    /// The aggregated [`CvScores`] over the configured number of folds.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidInput`] when `number_of_folds` is negative or
    ///   unrepresentable as `usize`, or when the resolved fold count is `< 2`
    ///   (matching [`StratifiedCrossValidation::folds`](crate::resampling::StratifiedCrossValidation::folds)).
    /// * [`Error::InsufficientData`] when the fold count exceeds `n`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::resampling::CrossValidation;
    ///
    /// let cv = CrossValidation { number_of_folds: 5, random_seed: 42, ..Default::default() };
    /// let scores = cv.run(20, |_train, _test| 1.0)?;
    /// assert_eq!(scores.fold_scores().len(), 5);
    /// // Every fold scored 1.0, so the mean is 1.0 and the spread is zero.
    /// assert!((scores.mean() - 1.0).abs() < 1e-12);
    /// assert!(scores.std_error().abs() < 1e-12);
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    pub fn run(
        &self,
        n: usize,
        fit_score: impl FnMut(&[usize], &[usize]) -> f64,
    ) -> Result<CvScores> {
        let k = usize::try_from(self.number_of_folds)
            .map_err(|_| Error::InvalidInput("number_of_folds must be non-negative".to_owned()))?;
        let mut rng = SplitMix64::new(self.random_seed.cast_unsigned());
        cross_validate(n, k, &mut rng, fit_score)
    }
}

// Kani note (dropped harness): a `resampling_cross_validate_rejects_out_of_domain`
// harness — proving `cross_validate` returns `Err(InsufficientData)` for every
// out-of-domain `(n, k)` — was attempted but dropped. `cross_validate` guards its
// fold count (`k < 2 || k > n`) and then, on the in-domain branch, delegates to
// `kfold_indices`. CBMC structurally unwinds that in-domain branch's
// `permutation`/`kfold_indices` loops over the symbolic `n` even though the guard
// makes them unreachable on the feasible out-of-domain paths, and the resulting
// unwinding assertions do not discharge within the ~3-minute per-harness budget.
// The one-line domain guard's downstream is already fully verified elsewhere:
// `schemes::verification::resampling_kfold_test_sets_partition` and
// `resampling_kfold_zero_k_is_empty` prove the k-fold index-safety and partition
// invariants over all generator states, and `index::verification` proves the
// underlying index arithmetic. The guard itself is exercised by the
// `#[cfg(test)]` unit suite (`rejects_fewer_than_two_folds`,
// `rejects_more_folds_than_observations`).

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_scores_via_accessors() -> Result<()> {
        let mut rng = SplitMix64::new(5);
        let scores = cross_validate(8, 4, &mut rng, |_train, _test| 0.5)?;
        assert_eq!(
            scores.fold_scores(),
            &[0.5, 0.5, 0.5, 0.5],
            "fold_scores() must return the per-fold scores slice"
        );
        assert!(
            (scores.mean() - 0.5).abs() < 1e-12,
            "mean() was {}",
            scores.mean()
        );
        assert!(
            scores.std_error().abs() < 1e-12,
            "std_error() was {}",
            scores.std_error()
        );
        Ok(())
    }

    #[test]
    fn rejects_fewer_than_two_folds() {
        let mut rng = SplitMix64::new(1);
        let result = cross_validate(10, 1, &mut rng, |_train, _test| 0.0);
        assert!(
            matches!(result, Err(Error::InvalidInput(_))),
            "k < 2 is a bad parameter and must be InvalidInput, got {result:?}"
        );
    }

    #[test]
    fn rejects_more_folds_than_observations() {
        let mut rng = SplitMix64::new(1);
        let result = cross_validate(3, 5, &mut rng, |_train, _test| 0.0);
        assert_eq!(
            result,
            Err(Error::InsufficientData),
            "k > n must be rejected, got {result:?}"
        );
    }

    #[test]
    fn folds_partition_the_observations() -> Result<()> {
        let n = 10;
        let k = 5;
        let mut splits: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
        let mut rng = SplitMix64::new(7);
        let scores = cross_validate(n, k, &mut rng, |train, test| {
            splits.push((train.to_vec(), test.to_vec()));
            0.0
        })?;

        assert_eq!(scores.fold_scores().len(), k, "one score per fold");
        assert_eq!(splits.len(), k, "fit_score must run once per fold");

        // Test sets partition 0..n: every index appears exactly once across folds.
        let mut seen = vec![0u32; n];
        for (train, test) in &splits {
            for &i in test {
                if let Some(count) = seen.get_mut(i) {
                    *count += 1;
                }
            }
            // train and test are disjoint within a fold.
            for &t in test {
                assert!(!train.contains(&t), "index {t} in both train and test");
            }
            // train is the complement of test.
            assert_eq!(train.len() + test.len(), n, "train ∪ test must cover all n");
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "each index must land in exactly one test fold, counts: {seen:?}"
        );
        Ok(())
    }

    #[test]
    fn aggregates_mean_and_standard_error() -> Result<()> {
        // Predetermined per-fold scores drive the aggregation independently of the
        // fold assignment: fit_score returns the next value on each call (FnMut).
        // Golden values (python3):
        //   import numpy as np
        //   s = np.array([0.80, 0.75, 0.82, 0.79, 0.85])
        //   s.mean()                       -> 0.8019999999999999
        //   s.std(ddof=1)/np.sqrt(len(s))  -> 0.016552945357246843
        let predetermined = [0.80, 0.75, 0.82, 0.79, 0.85];
        let mut supply = predetermined.iter().copied();
        let mut rng = SplitMix64::new(3);
        let scores = cross_validate(10, 5, &mut rng, |_train, _test| {
            supply.next().unwrap_or(f64::NAN)
        })?;

        assert_eq!(
            scores.fold_scores(),
            &predetermined,
            "scores recorded in order"
        );
        assert!(
            (scores.mean() - 0.801_999_999_999_999_9).abs() < 1e-12,
            "mean was {}",
            scores.mean()
        );
        assert!(
            (scores.std_error() - 0.016_552_945_357_246_843).abs() < 1e-12,
            "std_error was {}",
            scores.std_error()
        );
        Ok(())
    }

    #[test]
    fn identical_seeds_reproduce_scores() -> Result<()> {
        // fit_score depends on the split (sum of test indices), so identical
        // scores imply identical folds — the determinism contract.
        let score_of =
            |_train: &[usize], test: &[usize]| -> f64 { count_to_f64(test.iter().sum::<usize>()) };
        let a = cross_validate(12, 4, &mut SplitMix64::new(2024), score_of)?;
        let b = cross_validate(12, 4, &mut SplitMix64::new(2024), score_of)?;
        assert_eq!(a, b, "identical seeds must reproduce the CV scores");

        let c = cross_validate(12, 4, &mut SplitMix64::new(99), score_of)?;
        assert_ne!(
            a.fold_scores(),
            c.fold_scores(),
            "different seeds should generally yield different folds"
        );
        Ok(())
    }

    #[test]
    fn run_rejects_negative_fold_count_as_invalid_input() {
        let cv = CrossValidation {
            number_of_folds: -3,
            random_seed: 1,
            ..Default::default()
        };
        let result = cv.run(10, |_train, _test| 0.0);
        assert!(
            matches!(result, Err(Error::InvalidInput(_))),
            "a negative number_of_folds is a bad parameter and must be InvalidInput, got {result:?}"
        );
    }

    #[test]
    fn run_matches_free_function_with_equivalent_seed() -> Result<()> {
        let score_of =
            |_train: &[usize], test: &[usize]| -> f64 { count_to_f64(test.iter().sum::<usize>()) };
        let cv = CrossValidation {
            number_of_folds: 5,
            random_seed: 42,
            ..Default::default()
        };
        let via_run = cv.run(10, score_of)?;
        // random_seed 42 (i64) reinterprets bit-for-bit to 42u64.
        let via_free = cross_validate(10, 5, &mut SplitMix64::new(42), score_of)?;
        assert_eq!(
            via_run, via_free,
            "run() must match cross_validate() with the equivalent seed"
        );
        Ok(())
    }
}
