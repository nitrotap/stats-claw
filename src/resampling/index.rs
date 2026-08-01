//! Cast-free integer/float index helpers shared by the resampling schemes and
//! the interval estimator.
//!
//! The protected `style.rs` guard bans `as` casts in `src/`, so every
//! `usize`/`f64` conversion here routes through `From`/`TryFrom` or stays in
//! integer arithmetic.

use crate::numeric::count_to_f64;
use crate::rng::SplitMix64;

/// Returns `floor(fraction * n)` as a `usize`, computed without an `as` cast.
///
/// Binary-searches the integer ranks in `0..=n`, comparing each candidate's exact
/// `f64` widening against the float target, so no `f64 -> integer` conversion is
/// ever needed (the `style.rs` guard bans `as`). The result lies in `0..=n`.
///
/// # Arguments
///
/// * `fraction` — a probability in `[0, 1]`.
/// * `n` — the sample size.
///
/// # Returns
///
/// `floor(fraction * n)` as a `usize`.
pub(super) fn floor_rank(fraction: f64, n: usize) -> usize {
    let target = fraction * count_to_f64(n);
    let (mut lo, mut hi) = (0usize, n);
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if count_to_f64(mid) <= target {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    lo
}

/// Draws a uniform index in `0..n` from `rng`.
///
/// Reduces a full-width `u64` draw modulo `n`, keeping the computation in integer
/// arithmetic so no `f64 -> integer` cast is needed (the `style.rs` guard bans
/// `as`, and a float round-trip would risk an off-by-one at the upper end). The
/// modulo bias is negligible for the small `n` used in resampling (`n` far below
/// `2^32`), and the stream stays byte-identical for a fixed seed.
///
/// # Arguments
///
/// * `rng` — the deterministic generator to draw from.
/// * `n` — collection length; must be `> 0` (callers guarantee this).
///
/// # Returns
///
/// A uniformly distributed index in `0..n`.
pub(super) fn uniform_index(rng: &mut SplitMix64, n: usize) -> usize {
    let span = u64::try_from(n).unwrap_or(u64::MAX);
    let draw = rng.next_u64() % span;
    usize::try_from(draw).unwrap_or(n - 1)
}

/// Kani formal-verification harnesses for the shared resampling index arithmetic.
///
/// These prove properties over *all* generator states (symbolic `kani::any()`
/// `u64`) and small symbolic collection sizes, rather than the sampled draws the
/// `#[cfg(test)]` suites exercise. The crown result is
/// [`uniform_index_in_bounds`]: every index the resamplers derive from a
/// `SplitMix64` draw lands in `0..n`, for every one of the `2^64` generator
/// states. Compiled only under `cargo kani` (behind `#[cfg(kani)]`) and invisible
/// to normal build/test/clippy. Run e.g. with
/// `cargo kani -Z stubbing -p stats-claw --harness resampling_uniform_index_in_bounds`.
#[cfg(kani)]
mod verification {
    use super::{SplitMix64, floor_rank, uniform_index};

    /// Upper bound on the symbolic collection size the index harnesses explore.
    ///
    /// The index arithmetic is size-independent (a modulo reduction and a
    /// binary search), so a small representative range keeps the symbolic modulo
    /// and loop unrolling tractable while still covering the empty-to-tiny sizes
    /// resampling actually uses. Matches the task's "small symbolic-or-fixed n"
    /// guidance.
    const MAX_N: usize = 5;

    /// Proves the crown index-safety property: for *every* generator state and
    /// every collection size `1..=MAX_N`, [`uniform_index`] returns an index
    /// strictly less than `n` — so any slice access the resamplers make with it is
    /// in bounds. Composes with `rng::next_u64` over the full symbolic state space:
    /// `draw = next_u64() % n < n`, and the `usize` widening of a value below `n`
    /// cannot fail, so the fallback branch is never taken.
    #[kani::proof]
    fn resampling_uniform_index_in_bounds() {
        let state: u64 = kani::any();
        let n: usize = kani::any();
        kani::assume(n > 0);
        kani::assume(n <= MAX_N);
        let mut rng = SplitMix64::new(state);
        let idx = uniform_index(&mut rng, n);
        assert!(idx < n, "uniform_index escaped 0..n: {idx} >= {n}");
    }

    /// Proves [`floor_rank`] never panics and always returns a rank in `0..=n` for
    /// a symbolic probability `fraction` and every collection size `0..=MAX_N`.
    /// The binary search shrinks `hi` toward `lo` monotonically; the `hi = mid - 1`
    /// step only runs when `lo < hi`, forcing `mid >= lo + 1 >= 1`, so the
    /// subtraction cannot underflow. Bounding the rank keeps every percentile
    /// index the interval estimator derives inside `0..n`.
    #[kani::proof]
    #[kani::unwind(6)]
    fn resampling_floor_rank_in_range() {
        let fraction: f64 = kani::any();
        kani::assume(fraction.is_finite());
        kani::assume(fraction >= 0.0);
        kani::assume(fraction <= 1.0);
        let n: usize = kani::any();
        kani::assume(n <= MAX_N);
        let rank = floor_rank(fraction, n);
        assert!(rank <= n, "floor_rank escaped 0..=n: {rank} > {n}");
    }
}
