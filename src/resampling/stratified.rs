//! Stratified k-fold cross-validation splits, for the
//! [`StratifiedCrossValidation`].
//!
//! Equivalent to `sklearn.model_selection.StratifiedKFold(shuffle=True)`: each
//! fold's per-class counts match the overall label proportions as closely as
//! possible — for every class `c`, a fold holds either `floor(m_c / k)` or
//! `ceil(m_c / k)` members of `c`, so any two folds differ by at most one. The
//! shuffle draws from the deterministic [`SplitMix64`] PRNG (the same Fisher–Yates
//! idiom as [`permutation`](super::schemes::permutation)), so a fixed seed
//! reproduces the split bit-for-bit.

use std::collections::HashMap;

use super::index::uniform_index;
use crate::error::{Error, Result};
use crate::resampling::StratifiedCrossValidation;
use crate::rng::SplitMix64;

/// Partitions labelled observations into `k` stratified cross-validation folds.
///
/// Groups the observation indices by class, shuffles each class in place with the
/// Fisher–Yates idiom driven by `rng` (the same shuffle
/// [`permutation`](super::schemes::permutation) uses), then deals each class's
/// members round-robin across the `k` folds. Round-robin dealing keeps every
/// fold's count of a class within one of the ideal `m_c / k`, matching
/// scikit-learn's `StratifiedKFold` semantics: the class proportions of each fold
/// track the overall label proportions as closely as possible. Class ids are
/// arbitrary `usize` values (need not be contiguous). The split is deterministic
/// for a fixed seed and label slice.
///
/// # Arguments
///
/// * `labels` — the class id of each observation; observation `i` has label
///   `labels[i]`. Must be non-empty.
/// * `k` — the number of folds; must be `>= 2` and no greater than the smallest
///   class count.
/// * `rng` — the deterministic generator driving the per-class shuffle.
///
/// # Returns
///
/// A `k`-element vector of `(train_indices, test_indices)` pairs. The test sets
/// form a partition of `0..labels.len()`, and each train set is its complement.
///
/// # Errors
///
/// * [`Error::InvalidInput`] if `k < 2`.
/// * [`Error::InsufficientData`] if `labels` is empty, or if `k` exceeds the
///   smallest class count (a fold would then lack a member of that class —
///   scikit-learn raises here too).
///
/// # Examples
///
/// ```
/// use stats_claw::resampling::stratified_kfold_indices;
/// use stats_claw::rng::SplitMix64;
///
/// // 15 of class 0, 10 of class 1; k = 5 divides both evenly.
/// let mut labels = vec![0usize; 15];
/// labels.extend(std::iter::repeat_n(1usize, 10));
/// let mut rng = SplitMix64::new(7);
/// let folds = stratified_kfold_indices(&labels, 5, &mut rng)?;
///
/// // Every test fold holds exactly 3 of class 0 and 2 of class 1.
/// for (_, test) in &folds {
///     let c0 = test.iter().filter(|&&i| labels.get(i) == Some(&0)).count();
///     let c1 = test.iter().filter(|&&i| labels.get(i) == Some(&1)).count();
///     assert_eq!((c0, c1), (3, 2));
/// }
/// # Ok::<(), stats_claw::error::Error>(())
/// ```
pub fn stratified_kfold_indices(
    labels: &[usize],
    k: usize,
    rng: &mut SplitMix64,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    if k < 2 {
        return Err(Error::InvalidInput("k must be >= 2".to_owned()));
    }
    if labels.is_empty() {
        return Err(Error::InsufficientData);
    }

    // Group observation indices by class, preserving first-seen class order so
    // the per-class shuffle consumes `rng` in a seed-deterministic sequence.
    let mut group_of: HashMap<usize, usize> = HashMap::new();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for (i, &class) in labels.iter().enumerate() {
        let slot = *group_of.entry(class).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        if let Some(members) = groups.get_mut(slot) {
            members.push(i);
        }
    }

    let min_count = groups
        .iter()
        .map(Vec::len)
        .min()
        .ok_or(Error::InsufficientData)?;
    if k > min_count {
        return Err(Error::InsufficientData);
    }

    // Deal each class round-robin across the folds. Record (fold, observation)
    // so the complement (train) can be built without index-into-slice access.
    let mut assignments: Vec<(usize, usize)> = Vec::with_capacity(labels.len());
    for mut members in groups {
        // Fisher–Yates shuffle in place (the idiom `permutation` uses).
        for i in (1..members.len()).rev() {
            let j = uniform_index(rng, i + 1);
            members.swap(i, j);
        }
        for (position, observation) in members.into_iter().enumerate() {
            assignments.push((position % k, observation));
        }
    }

    Ok((0..k)
        .map(|fold| {
            let mut train = Vec::new();
            let mut test = Vec::new();
            for &(assigned, observation) in &assignments {
                if assigned == fold {
                    test.push(observation);
                } else {
                    train.push(observation);
                }
            }
            (train, test)
        })
        .collect())
}

impl StratifiedCrossValidation {
    /// Splits labelled observations into stratified folds using this scheme's
    /// configured [`number_of_folds`](Self::number_of_folds) and
    /// [`random_seed`](Self::random_seed).
    ///
    /// Seeds a fresh [`SplitMix64`] from `random_seed` (reinterpreting the signed
    /// seed's bits as unsigned) and delegates to [`stratified_kfold_indices`], so
    /// two calls on structs with equal fields yield identical splits.
    ///
    /// # Arguments
    ///
    /// * `labels` — the class id of each observation; see
    ///   [`stratified_kfold_indices`] for the partitioning contract.
    ///
    /// # Returns
    ///
    /// A vector of `(train_indices, test_indices)` pairs, one per fold.
    ///
    /// # Errors
    ///
    /// * [`Error::InvalidInput`] if `number_of_folds` is negative or `< 2`.
    /// * [`Error::InsufficientData`] if `labels` is empty or `number_of_folds`
    ///   exceeds the smallest class count.
    ///
    /// # Examples
    ///
    /// ```
    /// use stats_claw::resampling::StratifiedCrossValidation;
    ///
    /// let cv = StratifiedCrossValidation {
    ///     number_of_folds: 2,
    ///     random_seed: 42,
    ///     ..Default::default()
    /// };
    /// let labels = [0usize, 1, 0, 1];
    /// let folds = cv.folds(&labels)?;
    /// assert_eq!(folds.len(), 2);
    /// # Ok::<(), stats_claw::error::Error>(())
    /// ```
    pub fn folds(&self, labels: &[usize]) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let k = usize::try_from(self.number_of_folds)
            .map_err(|_| Error::InvalidInput("number_of_folds must be non-negative".to_owned()))?;
        let mut rng = SplitMix64::new(self.random_seed.cast_unsigned());
        stratified_kfold_indices(labels, k, &mut rng)
    }
}

/// Kani formal-verification harnesses for stratified k-fold input validation.
///
/// [`stratified_kfold_indices`] guards `k` and the label slice before building any
/// per-class groups, so these prove the two rejection paths over a symbolic `k` (or
/// an empty label slice) and every generator state, rather than the sampled sizes
/// the `#[cfg(test)]` suite uses. The interior per-class shuffle draws through the
/// same [`uniform_index`] proven in-bounds in [`super::index`], so the full
/// partition proof is not re-derived here (the `HashMap` grouping is left to the
/// unit suite to keep the symbolic model tractable). Compiled only under
/// `cargo kani` (behind `#[cfg(kani)]`); invisible to normal build/test/clippy. Run
/// e.g. with
/// `cargo kani -Z stubbing -p stats-claw --harness resampling_stratified_rejects_small_k`.
#[cfg(kani)]
mod verification {
    use super::{Error, SplitMix64, stratified_kfold_indices};

    /// Proves the fold-count guard: for *every* symbolic `k < 2` and generator
    /// state, [`stratified_kfold_indices`] returns [`Error::InvalidInput`] and never
    /// panics — a stratified split needs at least two folds.
    #[kani::proof]
    fn resampling_stratified_rejects_small_k() {
        let k: usize = kani::any();
        kani::assume(k < 2);
        let labels = [0usize, 1usize];
        let state: u64 = kani::any();
        let mut rng = SplitMix64::new(state);
        let result = stratified_kfold_indices(&labels, k, &mut rng);
        assert!(
            matches!(result, Err(Error::InvalidInput(_))),
            "k < 2 must be rejected with InvalidInput"
        );
    }

    /// Proves the empty-input guard: an empty label slice yields
    /// [`Error::InsufficientData`] for every valid `k` and generator state, with no
    /// panic — there is nothing to partition.
    #[kani::proof]
    fn resampling_stratified_empty_labels_insufficient() {
        let labels: [usize; 0] = [];
        let state: u64 = kani::any();
        let mut rng = SplitMix64::new(state);
        let result = stratified_kfold_indices(&labels, 2, &mut rng);
        assert!(
            matches!(result, Err(Error::InsufficientData)),
            "empty labels must be rejected with InsufficientData"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `k < 2` is rejected as invalid input, matching scikit-learn's requirement
    /// that a k-fold split have at least two folds.
    #[test]
    fn k_below_two_is_invalid() {
        let labels = [0usize, 1, 0, 1];
        let mut rng = SplitMix64::new(1);
        assert!(
            matches!(
                stratified_kfold_indices(&labels, 1, &mut rng),
                Err(Error::InvalidInput(_))
            ),
            "k = 1 should be InvalidInput"
        );
    }

    /// Empty labels have nothing to partition, so the split reports insufficient
    /// data rather than returning empty folds.
    #[test]
    fn empty_labels_is_insufficient_data() {
        let labels: [usize; 0] = [];
        let mut rng = SplitMix64::new(1);
        assert_eq!(
            stratified_kfold_indices(&labels, 2, &mut rng),
            Err(Error::InsufficientData),
            "empty labels should be InsufficientData"
        );
    }

    /// `k` greater than the smallest class count cannot stratify (a fold would be
    /// left without a member of that class), so it reports insufficient data —
    /// scikit-learn raises here too.
    #[test]
    fn k_above_smallest_class_is_insufficient_data() {
        // class 0 has 3 members, class 1 has 2 — smallest class count is 2.
        let labels = [0usize, 0, 0, 1, 1];
        let mut rng = SplitMix64::new(1);
        assert_eq!(
            stratified_kfold_indices(&labels, 3, &mut rng),
            Err(Error::InsufficientData),
            "k = 3 exceeds smallest class count 2"
        );
    }

    /// The test folds partition `0..n`: each observation lands in exactly one test
    /// fold, and each train set is precisely the complement of its test set.
    #[test]
    fn test_folds_partition_and_train_is_complement() -> Result<()> {
        let labels = [0usize, 0, 0, 1, 1, 1, 0, 1, 0, 1];
        let n = labels.len();
        let mut rng = SplitMix64::new(99);
        let folds = stratified_kfold_indices(&labels, 3, &mut rng)?;

        let mut seen = vec![0usize; n];
        for (train, test) in &folds {
            for &t in test {
                if let Some(count) = seen.get_mut(t) {
                    *count += 1;
                }
            }
            // train is the exact complement: sizes sum to n and sets are disjoint.
            assert_eq!(train.len() + test.len(), n, "train+test must cover all n");
            for &tr in train {
                assert!(
                    !test.contains(&tr),
                    "index {tr} appears in both train and test of a fold"
                );
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "every index must appear in exactly one test fold, got {seen:?}"
        );
        Ok(())
    }

    /// Counts how many test-fold members belong to `class`.
    fn class_count(test: &[usize], labels: &[usize], class: usize) -> usize {
        test.iter()
            .filter(|&&i| labels.get(i) == Some(&class))
            .count()
    }

    /// With class counts divisible by `k`, every fold holds exactly the ideal
    /// per-class share. 15 of class 0 and 10 of class 1 over k=5 → exactly 3 and 2
    /// per fold. This is the scikit-learn `StratifiedKFold` guarantee in the
    /// evenly-divisible case (hand-computed: 15/5 = 3, 10/5 = 2); it holds for any
    /// seed because round-robin dealing distributes m divisible-by-k members
    /// exactly m/k per fold.
    #[test]
    fn exact_stratification_when_divisible() -> Result<()> {
        let mut labels = vec![0usize; 15];
        labels.extend(std::iter::repeat_n(1usize, 10));
        let mut rng = SplitMix64::new(2024);
        let folds = stratified_kfold_indices(&labels, 5, &mut rng)?;
        assert_eq!(folds.len(), 5, "expected 5 folds");
        for (_, test) in &folds {
            assert_eq!(
                (class_count(test, &labels, 0), class_count(test, &labels, 1)),
                (3, 2),
                "each fold must hold exactly 3 of class 0 and 2 of class 1"
            );
        }
        Ok(())
    }

    /// General shape: for a 60/40 split of n=20 (12 of class 0, 8 of class 1) with
    /// k=5, no fold's class-c count differs from the ideal `m_c / k` by more than
    /// one. Hand-computed folds: class 0 (12 = 5·2+2) → two folds get 3, three get
    /// 2; class 1 (8 = 5·1+3) → three folds get 2, two get 1. So every fold's
    /// count is `floor` or `ceil` of the ideal — scikit-learn's "as balanced as
    /// possible" guarantee.
    #[test]
    fn general_shape_within_one_of_ideal() -> Result<()> {
        let mut labels = vec![0usize; 12];
        labels.extend(std::iter::repeat_n(1usize, 8));
        let k = 5;
        let mut rng = SplitMix64::new(7);
        let folds = stratified_kfold_indices(&labels, k, &mut rng)?;
        for (class, m) in [(0usize, 12usize), (1usize, 8usize)] {
            let floor = m / k;
            let ceil = m.div_ceil(k);
            for (_, test) in &folds {
                let c = class_count(test, &labels, class);
                assert!(
                    c == floor || c == ceil,
                    "class {class} fold count {c} not in {{{floor}, {ceil}}}"
                );
            }
        }
        Ok(())
    }

    /// The split is reproducible: identical seeds give identical folds, and
    /// different seeds give a different assignment (for data large enough that the
    /// shuffle can differ).
    #[test]
    fn deterministic_by_seed() -> Result<()> {
        let mut labels = vec![0usize; 30];
        labels.extend(std::iter::repeat_n(1usize, 20));
        let same_a = stratified_kfold_indices(&labels, 5, &mut SplitMix64::new(11))?;
        let same_b = stratified_kfold_indices(&labels, 5, &mut SplitMix64::new(11))?;
        assert_eq!(same_a, same_b, "same seed must reproduce the split");

        let different = stratified_kfold_indices(&labels, 5, &mut SplitMix64::new(999))?;
        assert_ne!(
            same_a, different,
            "different seeds should shuffle to a different assignment"
        );
        Ok(())
    }

    /// Class ids are arbitrary `usize` values, not dense `0..c`. Non-contiguous
    /// ids like {7, 42} stratify exactly as contiguous ones would.
    #[test]
    fn non_contiguous_class_ids() -> Result<()> {
        let labels = [7usize, 42, 7, 42, 7, 42, 7, 42];
        let mut rng = SplitMix64::new(5);
        let folds = stratified_kfold_indices(&labels, 2, &mut rng)?;
        assert_eq!(folds.len(), 2, "expected 2 folds");
        for (_, test) in &folds {
            // 4 of each class over 2 folds → exactly 2 of each per fold.
            assert_eq!(
                (
                    class_count(test, &labels, 7),
                    class_count(test, &labels, 42)
                ),
                (2, 2),
                "each fold must hold 2 of class 7 and 2 of class 42"
            );
        }
        Ok(())
    }

    /// The inherent [`StratifiedCrossValidation::folds`] delegates to the free
    /// function using its configured fields, so it produces the same split as
    /// calling the free function with a matching seed.
    #[test]
    fn inherent_folds_matches_free_function() -> Result<()> {
        let labels = [0usize, 1, 0, 1, 0, 1];
        let cv = StratifiedCrossValidation {
            number_of_folds: 3,
            random_seed: 123,
            ..Default::default()
        };
        let via_method = cv.folds(&labels)?;
        let via_free = stratified_kfold_indices(&labels, 3, &mut SplitMix64::new(123))?;
        assert_eq!(
            via_method, via_free,
            "folds() must match the free function with the same seed"
        );
        Ok(())
    }
}
