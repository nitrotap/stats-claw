//! Seeded resampling schemes: with-replacement bootstrap, Fisher–Yates
//! permutation, and k-fold cross-validation splits.
//!
//! Every draw comes from the deterministic [`SplitMix64`] PRNG, so two calls with
//! identically seeded generators produce identical output.

use super::index::uniform_index;
use crate::rng::SplitMix64;

/// Draws `b` bootstrap resamples of `n` with-replacement indices.
///
/// Each resample is an independent length-`n` vector of indices drawn uniformly
/// from `0..n` with replacement, the standard nonparametric bootstrap scheme.
/// The draws come from `rng`, so two calls with identically seeded generators
/// produce identical collections.
///
/// # Arguments
///
/// * `n` — sample size; each resample has this many indices. With `n == 0` every
///   resample is empty.
/// * `b` — number of resamples to draw.
/// * `rng` — the deterministic generator driving the draws.
///
/// # Returns
///
/// A `b`-element vector of length-`n` index vectors.
///
/// # Examples
///
/// ```
/// use stats_claw::resampling::bootstrap_indices;
/// use stats_claw::rng::SplitMix64;
///
/// let a = bootstrap_indices(10, 3, &mut SplitMix64::new(7));
/// let b = bootstrap_indices(10, 3, &mut SplitMix64::new(7));
/// assert_eq!(a, b); // identical seeds reproduce the draws
/// assert_eq!(a.len(), 3);
/// assert!(a.iter().all(|r| r.len() == 10 && r.iter().all(|&i| i < 10)));
/// ```
#[must_use]
pub fn bootstrap_indices(n: usize, b: usize, rng: &mut SplitMix64) -> Vec<Vec<usize>> {
    (0..b)
        .map(|_| {
            if n == 0 {
                Vec::new()
            } else {
                (0..n).map(|_| uniform_index(rng, n)).collect()
            }
        })
        .collect()
}

/// Draws a uniform random permutation of `0..n` via the Fisher–Yates shuffle.
///
/// Walks the index vector from the back, swapping each position with a uniformly
/// chosen earlier-or-equal position, which yields each of the `n!` orderings with
/// equal probability. Reproducible for a fixed seed.
///
/// # Arguments
///
/// * `n` — number of elements to permute. With `n == 0` the result is empty.
/// * `rng` — the deterministic generator driving the swaps.
///
/// # Returns
///
/// A vector containing `0..n` in a uniformly random order.
///
/// # Examples
///
/// ```
/// use stats_claw::resampling::permutation;
/// use stats_claw::rng::SplitMix64;
///
/// let mut p = permutation(6, &mut SplitMix64::new(42));
/// p.sort_unstable();
/// assert_eq!(p, vec![0, 1, 2, 3, 4, 5]); // a permutation contains each index once
/// ```
#[must_use]
pub fn permutation(n: usize, rng: &mut SplitMix64) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = uniform_index(rng, i + 1);
        idx.swap(i, j);
    }
    idx
}

/// Partitions `0..n` into `k` cross-validation folds.
///
/// Shuffles the indices once, then assigns position `p` to test fold `p % k`, so
/// every observation lands in exactly one test fold and the corresponding train
/// set is its complement. The split is reproducible for a fixed seed.
///
/// # Arguments
///
/// * `n` — number of observations.
/// * `k` — number of folds; must be `> 0` (callers guarantee this). With `k == 0`
///   the result is empty.
/// * `rng` — the deterministic generator driving the shuffle.
///
/// # Returns
///
/// A `k`-element vector of `(train_indices, test_indices)` pairs whose test sets
/// form a partition of `0..n`.
///
/// # Examples
///
/// ```
/// use stats_claw::resampling::kfold_indices;
/// use stats_claw::rng::SplitMix64;
///
/// let folds = kfold_indices(10, 5, &mut SplitMix64::new(1));
/// let total_test: usize = folds.iter().map(|(_, test)| test.len()).sum();
/// assert_eq!(total_test, 10); // test sets partition the data
/// ```
#[must_use]
pub fn kfold_indices(n: usize, k: usize, rng: &mut SplitMix64) -> Vec<(Vec<usize>, Vec<usize>)> {
    if k == 0 {
        return Vec::new();
    }
    let perm = permutation(n, rng);
    (0..k)
        .map(|fold| {
            let mut train = Vec::new();
            let mut test = Vec::new();
            for (pos, &obs) in perm.iter().enumerate() {
                if pos % k == fold {
                    test.push(obs);
                } else {
                    train.push(obs);
                }
            }
            (train, test)
        })
        .collect()
}
