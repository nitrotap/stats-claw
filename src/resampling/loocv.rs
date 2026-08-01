//! Leave-one-out cross-validation (LOO-CV) splits and scoring, layered onto the
//! [`LeaveOneOutCrossValidation`].
//!
//! LOO-CV is the deterministic `k = n` limit of k-fold CV: each of the `n`
//! observations is held out as a singleton test set exactly once while the
//! remaining `n − 1` train it. There is no RNG anywhere in this module — the
//! folds are a fixed function of `n`, so results are trivially reproducible.

use super::cross_validation::CvScores;
use crate::error::{Error, Result};
use crate::resampling::LeaveOneOutCrossValidation;

/// Builds the `n` leave-one-out folds for a dataset of size `n`.
///
/// Fold `i` holds out observation `i` as the sole test index and trains on every
/// other index in ascending order, so the returned vector has exactly `n` entries
/// and the test singletons partition `0..n`. The split is a pure function of `n`
/// — no randomness is involved.
///
/// # Arguments
///
/// * `n` — the number of observations; must be at least 2 (a singleton has no
///   held-out complement to train on).
///
/// # Returns
///
/// An `n`-element vector of `(train_indices, test_indices)` pairs where each
/// `test_indices` is `[i]` and each `train_indices` is `0..n` with `i` removed,
/// order preserved.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] when `n < 2`.
///
/// # Examples
///
/// ```
/// use stats_claw::resampling::loo_indices;
///
/// let folds = loo_indices(3)?;
/// assert_eq!(folds[0], (vec![1, 2], vec![0]));
/// assert_eq!(folds[2], (vec![0, 1], vec![2]));
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn loo_indices(n: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    if n < 2 {
        return Err(Error::InsufficientData);
    }
    let folds = (0..n)
        .map(|i| {
            let train: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            (train, vec![i])
        })
        .collect();
    Ok(folds)
}

/// Runs leave-one-out cross-validation over `n` observations.
///
/// Builds the `n` LOO folds (via [`loo_indices`]), calls `fit_score(train, test)`
/// once per fold to obtain that fold's score, then summarises the scores by their
/// mean and standard error. The evaluator receives the ordered train complement
/// and the singleton test index of each fold, so every observation is scored as
/// the held-out point exactly once.
///
/// # Arguments
///
/// * `n` — the number of observations; must be at least 2.
/// * `fit_score` — invoked as `fit_score(train_idx, test_idx)` for each fold and
///   returning that fold's score (e.g. a held-out error). Takes `FnMut` so the
///   evaluator may carry mutable state across folds.
///
/// # Returns
///
/// A [`CvScores`] with the `n` per-fold scores, their mean, and their standard
/// error (`sd` with `ddof = 1`, divided by `sqrt(n)`) — the same score type
/// k-fold [`cross_validate`](super::cross_validation::cross_validate) returns.
///
/// # Errors
///
/// Returns [`Error::InsufficientData`] when `n < 2` (propagated from
/// [`loo_indices`]).
///
/// # Examples
///
/// ```
/// use stats_claw::resampling::loo_cross_validate;
///
/// // Constant score per fold: mean equals it, standard error is zero.
/// let scores = loo_cross_validate(5, |_train, _test| 3.0)?;
/// assert_eq!(scores.fold_scores().len(), 5);
/// assert!((scores.mean() - 3.0).abs() < 1e-12, "mean was {}", scores.mean());
/// assert!(scores.std_error().abs() < 1e-12, "std_error was {}", scores.std_error());
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn loo_cross_validate(
    n: usize,
    mut fit_score: impl FnMut(&[usize], &[usize]) -> f64,
) -> Result<CvScores> {
    let folds = loo_indices(n)?;
    let fold_scores: Vec<f64> = folds
        .iter()
        .map(|(train, test)| fit_score(train, test))
        .collect();
    Ok(CvScores::new(fold_scores))
}

impl LeaveOneOutCrossValidation {
    /// Runs leave-one-out cross-validation for this scheme over `n` observations.
    ///
    /// A thin inherent wrapper over [`loo_cross_validate`] so the
    /// [`LeaveOneOutCrossValidation`] type carries its own numerics: it ignores
    /// the scheme's descriptive fields and forwards `n` and `fit_score` unchanged.
    ///
    /// # Arguments
    ///
    /// * `n` — the number of observations; must be at least 2.
    /// * `fit_score` — invoked as `fit_score(train_idx, test_idx)` per fold,
    ///   returning that fold's score. `FnMut` so it may carry state across folds.
    ///
    /// # Returns
    ///
    /// A [`CvScores`] with the per-fold scores, their mean, and their standard
    /// error.
    ///
    /// # Errors
    ///
    /// Returns [`Error::InsufficientData`] when `n < 2`.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::resampling::LeaveOneOutCrossValidation;
    ///
    /// let scheme = LeaveOneOutCrossValidation::default();
    /// let scores = scheme.run(4, |_train, _test| 1.0)?;
    /// assert_eq!(scores.fold_scores().len(), 4);
    /// assert!((scores.mean() - 1.0).abs() < 1e-12);
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    pub fn run(
        &self,
        n: usize,
        fit_score: impl FnMut(&[usize], &[usize]) -> f64,
    ) -> Result<CvScores> {
        loo_cross_validate(n, fit_score)
    }
}

/// Kani formal-verification harnesses for the leave-one-out fold construction.
///
/// [`loo_indices`] is a pure function of `n` (no RNG), so these prove its
/// input-validation and partition invariants over a symbolic `n` and a small fixed
/// `n`, rather than the sampled fixtures the `#[cfg(test)]` suite uses. Compiled
/// only under `cargo kani` (behind `#[cfg(kani)]`); invisible to normal
/// build/test/clippy. Run e.g. with
/// `cargo kani -Z stubbing -p stats-claw --harness resampling_loo_rejects_small_n`.
#[cfg(kani)]
mod verification {
    use super::{Error, loo_indices};

    /// Proves the input-validation path: for *every* symbolic `n < 2`,
    /// [`loo_indices`] returns [`Error::InsufficientData`] and never panics — a
    /// singleton has no held-out complement to train on.
    ///
    /// The `#[kani::unwind(2)]` bound caps the (unreachable-on-feasible-paths)
    /// fold-building loop: with `n < 2` the guard returns before it, and CBMC
    /// discharges the over-unwinding of the infeasible `n >= 2` branch vacuously.
    #[kani::proof]
    #[kani::unwind(2)]
    fn resampling_loo_rejects_small_n() {
        let n: usize = kani::any();
        kani::assume(n < 2);
        let result = loo_indices(n);
        assert!(
            matches!(result, Err(Error::InsufficientData)),
            "n < 2 must be rejected with InsufficientData"
        );
    }

    /// Proves the LOO folds partition `0..N` and are in bounds for `n = 3`: exactly
    /// `N` folds, each test set the singleton `[i]`, each train set the ordered
    /// complement (size `N - 1`, every index `< N`, none equal to `i`), and the
    /// test singletons covering every observation exactly once.
    #[kani::proof]
    #[kani::unwind(5)]
    fn resampling_loo_indices_partition() {
        const N: usize = 3;
        let result = loo_indices(N);
        assert!(result.is_ok(), "n >= 2 must produce LOO folds");
        if let Ok(folds) = result {
            assert!(folds.len() == N, "expected one fold per observation");
            let mut seen = [0u8; N];
            for (i, (train, test)) in folds.iter().enumerate() {
                assert!(test.len() == 1, "each test set must be a singleton");
                for &t in test {
                    assert!(t == i, "fold {i} must test its own index");
                    assert!(t < N, "test index {t} escaped 0..N");
                    seen[t] += 1;
                }
                assert!(train.len() == N - 1, "train must be the complement");
                for &tr in train {
                    assert!(tr < N, "train index {tr} escaped 0..N");
                    assert!(tr != i, "train must not contain the held-out index");
                }
            }
            assert!(
                seen.iter().all(|&c| c == 1),
                "test singletons must partition 0..N"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use crate::resampling::LeaveOneOutCrossValidation;

    #[test]
    fn loo_indices_rejects_fewer_than_two() {
        assert_eq!(
            loo_indices(1),
            Err(Error::InsufficientData),
            "n < 2 must be rejected: a singleton has no held-out complement"
        );
    }

    #[test]
    fn loo_indices_three_gives_each_singleton_test() -> Result<()> {
        let folds = loo_indices(3)?;
        assert_eq!(
            folds,
            vec![
                (vec![1, 2], vec![0]),
                (vec![0, 2], vec![1]),
                (vec![0, 1], vec![2]),
            ],
            "each fold i must test [i] and train on the ordered complement"
        );
        Ok(())
    }

    #[test]
    fn cross_validate_calls_each_index_once_as_the_test_point() -> Result<()> {
        let mut seen: Vec<(Vec<usize>, Vec<usize>)> = Vec::new();
        let scores = loo_cross_validate(4, |train, test| {
            seen.push((train.to_vec(), test.to_vec()));
            0.0
        })?;
        assert_eq!(scores.fold_scores().len(), 4, "one score per fold");
        assert_eq!(
            seen,
            vec![
                (vec![1, 2, 3], vec![0]),
                (vec![0, 2, 3], vec![1]),
                (vec![0, 1, 3], vec![2]),
                (vec![0, 1, 2], vec![3]),
            ],
            "evaluator must receive each index once with its ordered complement for training"
        );
        Ok(())
    }

    #[test]
    fn cross_validate_computes_mean_and_std_error() -> Result<()> {
        // Predetermined fold scores fed in fold order; summaries checked against
        // numpy: np.mean([0.5,1.5,2.5,3.5]) == 2.0,
        // np.std(..., ddof=1)/np.sqrt(4) == 0.6454972243679028.
        let predetermined = [0.5_f64, 1.5, 2.5, 3.5];
        let mut next = predetermined.into_iter();
        let scores = loo_cross_validate(4, |_train, _test| next.next().unwrap_or(f64::NAN))?;
        for (i, (&got, &want)) in scores
            .fold_scores()
            .iter()
            .zip(predetermined.iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1e-12,
                "fold {i} score was {got}, want {want}"
            );
        }
        assert!(
            (scores.mean() - 2.0).abs() < 1e-12,
            "mean was {}",
            scores.mean()
        );
        assert!(
            (scores.std_error() - 0.645_497_224_367_902_8).abs() < 1e-12,
            "std_error was {}",
            scores.std_error()
        );
        Ok(())
    }

    #[test]
    fn cross_validate_predict_train_mean_squared_error_golden() -> Result<()> {
        // Analytic golden: LOO-CV predicting the training mean, scored by squared
        // error, over data = [2,4,6,8,10]. Fold i error = (train_mean - x_i)^2
        // with train_mean = (n*xbar - x_i)/(n-1). numpy reference:
        //   fold errors = [25.0, 6.25, 0.0, 6.25, 25.0]
        //   mean = 12.5, std(ddof=1)/sqrt(5) = 5.229125165837972.
        let data = [2.0_f64, 4.0, 6.0, 8.0, 10.0];
        let scores = loo_cross_validate(data.len(), |train, test| {
            let train_sum: f64 = train
                .iter()
                .map(|&j| data.get(j).copied().unwrap_or(f64::NAN))
                .sum();
            let train_count = f64::from(u32::try_from(train.len()).unwrap_or(0));
            let prediction = train_sum / train_count;
            let held_out = test
                .first()
                .and_then(|&i| data.get(i).copied())
                .unwrap_or(f64::NAN);
            (prediction - held_out).powi(2)
        })?;
        let expected = [25.0_f64, 6.25, 0.0, 6.25, 25.0];
        for (i, (&got, &want)) in scores.fold_scores().iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-10,
                "fold {i} error was {got}, want {want}"
            );
        }
        assert!(
            (scores.mean() - 12.5).abs() < 1e-10,
            "mean was {}",
            scores.mean()
        );
        assert!(
            (scores.std_error() - 5.229_125_165_837_972).abs() < 1e-10,
            "std_error was {}",
            scores.std_error()
        );
        Ok(())
    }

    #[test]
    fn loo_cross_validate_returns_unified_cv_scores() -> Result<()> {
        // LOO-CV must report the shared k-fold score type, not a bespoke struct.
        let scores: CvScores = loo_cross_validate(4, |_train, _test| 1.0)?;
        assert_eq!(scores.fold_scores().len(), 4, "one score per fold");
        assert!(
            (scores.mean() - 1.0).abs() < 1e-12,
            "mean was {}",
            scores.mean()
        );
        Ok(())
    }

    #[test]
    fn run_delegates_to_cross_validate() -> Result<()> {
        let scheme = LeaveOneOutCrossValidation::default();
        let scores = scheme.run(5, |_train, _test| 3.0)?;
        assert_eq!(scores.fold_scores().len(), 5, "one score per fold");
        assert!(
            scores
                .fold_scores()
                .iter()
                .all(|&s| (s - 3.0).abs() < 1e-12),
            "every fold score should be the constant 3.0"
        );
        assert!(
            (scores.mean() - 3.0).abs() < 1e-12,
            "mean was {}",
            scores.mean()
        );
        assert!(
            scores.std_error().abs() < 1e-12,
            "std_error was {}",
            scores.std_error()
        );
        Ok(())
    }

    #[test]
    fn accessors_expose_the_stored_summaries() -> Result<()> {
        let scores = loo_cross_validate(5, |_train, _test| 3.0)?;
        assert_eq!(
            scores.fold_scores().len(),
            5,
            "fold_scores accessor exposes one score per fold"
        );
        assert!(
            scores
                .fold_scores()
                .iter()
                .all(|&s| (s - 3.0).abs() < 1e-12),
            "fold_scores accessor returns the stored slice"
        );
        assert!(
            (scores.mean() - 3.0).abs() < 1e-12,
            "mean accessor was {}",
            scores.mean()
        );
        assert!(
            scores.std_error().abs() < 1e-12,
            "std_error accessor was {}",
            scores.std_error()
        );
        Ok(())
    }

    /// D4 boundary: `loo_indices(2)` yields both singleton-test folds in order.
    #[test]
    fn loo_indices_two_gives_both_singleton_folds() -> Result<()> {
        assert_eq!(
            loo_indices(2)?,
            vec![(vec![1], vec![0]), (vec![0], vec![1])],
            "n=2 LOO folds must be ([1],[0]) then ([0],[1])"
        );
        Ok(())
    }

    /// D4 boundary: at `n = 2`, LOO-CV aggregates predetermined fold scores into
    /// the exact mean and standard error.
    ///
    /// Scores `[1.0, 3.0]`: mean `2.0`; `sd(ddof=1) = sqrt(2)` and
    /// `SE = sd / sqrt(2) = 1.0`.
    #[test]
    fn n_two_aggregates_mean_and_standard_error() -> Result<()> {
        let predetermined = [1.0_f64, 3.0];
        let mut next = predetermined.into_iter();
        let scores = loo_cross_validate(2, |_train, _test| next.next().unwrap_or(f64::NAN))?;
        assert_eq!(
            scores.fold_scores(),
            &predetermined,
            "fold scores recorded in order"
        );
        assert!(
            (scores.mean() - 2.0).abs() < 1e-12,
            "n=2 mean was {}, expected 2.0",
            scores.mean()
        );
        assert!(
            (scores.std_error() - 1.0).abs() < 1e-12,
            "n=2 std_error was {}, expected sd(ddof=1)/sqrt(2) = 1.0",
            scores.std_error()
        );
        Ok(())
    }
}
