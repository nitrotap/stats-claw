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

/// Kani formal-verification harnesses for the seeded resampling schemes.
///
/// These prove structural index-safety and partition invariants over *all*
/// generator states (symbolic `kani::any()` `u64`) for small fixed collection
/// sizes, rather than the sampled draws the `#[cfg(test)]` suite exercises. Every
/// scheme derives its indices from [`uniform_index`](super::index::uniform_index),
/// whose in-bounds proof lives in [`super::index`]; the harnesses here add the
/// scheme-level invariants (bijection, partition) that compose on top. Compiled
/// only under `cargo kani` (behind `#[cfg(kani)]`); invisible to normal
/// build/test/clippy. Run e.g. with
/// `cargo kani -Z stubbing -p stats-claw --harness resampling_permutation_is_bijection`.
#[cfg(kani)]
mod verification {
    use super::{SplitMix64, bootstrap_indices, kfold_indices, permutation};

    /// Proves [`permutation`] returns a genuine bijection of `0..N` for *every*
    /// generator state: length `N`, every element in `0..N`, and each index
    /// appearing exactly once. The Fisher–Yates swaps draw `j = uniform_index(i+1)`
    /// with `j <= i < N`, so every `swap(i, j)` is in bounds — Kani discharges the
    /// slice-access safety while the `seen` tally proves no index is dropped or
    /// duplicated. `N = 4` keeps the three symbolic swaps tractable.
    #[kani::proof]
    #[kani::unwind(8)]
    fn resampling_permutation_is_bijection() {
        const N: usize = 4;
        let state: u64 = kani::any();
        let mut rng = SplitMix64::new(state);
        let perm = permutation(N, &mut rng);
        assert!(perm.len() == N, "permutation length changed from {N}");
        let mut seen = [false; N];
        for &v in &perm {
            assert!(v < N, "permutation produced an out-of-range index {v}");
            assert!(!seen[v], "permutation repeated index {v}");
            seen[v] = true;
        }
        assert!(seen.iter().all(|&s| s), "permutation dropped an index");
    }

    /// Proves the k-fold test sets partition `0..N` for *every* generator state:
    /// each observation appears in exactly one fold's test set. Because
    /// [`kfold_indices`] shuffles once into a bijection and assigns position `p` to
    /// fold `p % k`, every observation lands in one test fold; the `seen` tally
    /// checks that structural invariant directly. `N = 4`, `k = 2` keeps the
    /// composed shuffle-and-deal loop within the unwind budget.
    #[kani::proof]
    #[kani::unwind(8)]
    fn resampling_kfold_test_sets_partition() {
        const N: usize = 4;
        let state: u64 = kani::any();
        let mut rng = SplitMix64::new(state);
        let folds = kfold_indices(N, 2, &mut rng);
        let mut seen = [0u8; N];
        for (_, test) in &folds {
            for &obs in test {
                assert!(obs < N, "k-fold test index {obs} escaped 0..N");
                seen[obs] += 1;
            }
        }
        assert!(
            seen.iter().all(|&c| c == 1),
            "each observation must appear in exactly one test fold"
        );
    }

    /// Proves the `k == 0` input-validation path: [`kfold_indices`] returns an
    /// empty split for *any* symbolic `n` without drawing from the generator or
    /// panicking — the guard fires before the shuffle.
    #[kani::proof]
    fn resampling_kfold_zero_k_is_empty() {
        let n: usize = kani::any();
        let state: u64 = kani::any();
        let mut rng = SplitMix64::new(state);
        let folds = kfold_indices(n, 0, &mut rng);
        assert!(folds.is_empty(), "k == 0 must yield no folds");
    }

    /// Proves every index [`bootstrap_indices`] draws is in bounds for `0..N`, for
    /// *every* generator state: each of the `B` resamples holds exactly `N`
    /// with-replacement indices, all `< N`. This is the crown index-safety property
    /// at the scheme level — a bootstrap resample can never index past its data.
    /// `N = 3`, `B = 2` bounds the six symbolic draws.
    #[kani::proof]
    #[kani::unwind(6)]
    fn resampling_bootstrap_indices_in_bounds() {
        const N: usize = 3;
        const B: usize = 2;
        let state: u64 = kani::any();
        let mut rng = SplitMix64::new(state);
        let draws = bootstrap_indices(N, B, &mut rng);
        assert!(draws.len() == B, "expected B resamples");
        for resample in &draws {
            assert!(resample.len() == N, "each resample must hold N indices");
            for &i in resample {
                assert!(i < N, "bootstrap index {i} escaped 0..N");
            }
        }
    }
}
