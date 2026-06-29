//! Cast-free integer/float index helpers shared by the resampling schemes and
//! the interval estimator.
//!
//! The protected `style.rs` guard bans `as` casts in `src/`, so every
//! `usize`/`f64` conversion here routes through `From`/`TryFrom` or stays in
//! integer arithmetic.

use crate::rng::SplitMix64;

/// `2^32`, used to split an integer-valued `f64` into 32-bit halves.
const TWO_POW_32: f64 = 4_294_967_296.0;

/// Widens a `usize` count to `f64` without an `as` cast.
///
/// Splits the value into 32-bit halves (each losslessly representable as `f64`)
/// and recombines them, satisfying the `style.rs` no-`as` guard. Counts far below
/// `2^53` (every realistic sample size) convert exactly.
///
/// # Arguments
///
/// * `n` — the count to widen.
///
/// # Returns
///
/// `n` as an `f64`.
pub(super) fn count_to_f64(n: usize) -> f64 {
    let wide = u64::try_from(n).unwrap_or(u64::MAX);
    let hi = u32::try_from(wide >> 32).unwrap_or(0);
    let lo = u32::try_from(wide & 0xFFFF_FFFF).unwrap_or(0);
    f64::from(hi).mul_add(TWO_POW_32, f64::from(lo))
}

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
