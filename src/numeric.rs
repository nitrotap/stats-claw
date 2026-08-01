//! Shared cast-free numeric primitives: count widening and the sample mean and
//! (Bessel-corrected) variance.
//!
//! These are the canonical implementations of three tiny statistics that were
//! previously re-derived in several modules (the resampling schemes, the
//! parametric tests, the algorithms layer). Centralising them here removes the
//! duplication while keeping every call site bit-for-bit identical: the widening
//! splits a `usize` into 32-bit halves so it never uses an `as` cast (the
//! protected `style.rs` guard bans `as` in `src/`), and the mean/variance use the
//! same two-pass formula and the same widening the old inline copies used.
//!
//! The module itself is private (`mod numeric;`) and its items are `pub(crate)`
//! so the whole crate can share them while they stay out of the public API.
//!
//! `#[allow(clippy::redundant_pub_crate)]` is required and justified: for a
//! crate-wide helper in a *private* module the `redundant_pub_crate` lint (which
//! would have us write `pub`) and `unreachable_pub` (which forbids a bare `pub`
//! that cannot escape the crate) are mutually exclusive — CI runs
//! `clippy -- -D warnings`, so both are hard errors. Keeping the honest
//! `pub(crate)` and silencing the redundant-ness lint is the only conflict-free
//! option short of exposing the module publicly.
#![allow(clippy::redundant_pub_crate)]

/// `2^32`, used to split an integer-valued `f64` into 32-bit halves.
const TWO_POW_32: f64 = 4_294_967_296.0;

/// Widens a `usize` count to `f64` without an `as` cast.
///
/// Splits the value into its high and low 32-bit halves — each losslessly
/// representable as an `f64` — and recombines them with a single fused
/// multiply-add. This routes every conversion through `From`/`TryFrom` and so
/// satisfies the `style.rs` no-`as` guard.
///
/// # Arguments
///
/// * `n` — the count to widen.
///
/// # Returns
///
/// `n` as an `f64`. The result is *exact* for every `n < 2^53` (the `f64`
/// integer-exactness bound), which covers every realistic sample, cluster, or
/// iteration count this crate handles; above `2^53` the value rounds to the
/// nearest representable `f64` like any other `usize`-to-`f64` conversion.
#[must_use]
pub(crate) fn count_to_f64(n: usize) -> f64 {
    let wide = u64::try_from(n).unwrap_or(u64::MAX);
    let hi = u32::try_from(wide >> 32).unwrap_or(0);
    let lo = u32::try_from(wide & 0xFFFF_FFFF).unwrap_or(0);
    f64::from(hi).mul_add(TWO_POW_32, f64::from(lo))
}

/// Returns the arithmetic mean of `xs`, or `0.0` for an empty slice.
///
/// The empty-slice case returns `0.0` rather than `NaN` so callers that have
/// already validated non-emptiness (every current caller has) get the plain
/// average, and callers that have not get a benign sentinel instead of a
/// propagating `NaN`.
///
/// # Arguments
///
/// * `xs` — the observations to average.
///
/// # Returns
///
/// `Σ xs / xs.len()`, or `0.0` when `xs` is empty.
#[must_use]
pub(crate) fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.iter().sum::<f64>() / count_to_f64(xs.len())
}

/// Returns the unbiased (Bessel-corrected, `ddof = 1`) sample variance of `xs`.
///
/// Uses the classic two-pass estimator: first the [`mean`], then the mean of the
/// squared deviations divided by `n − 1`. Returns `0.0` for fewer than two
/// observations, where the `n − 1` denominator is undefined; callers that require
/// a positive variance check the result and raise a typed error themselves.
///
/// # Arguments
///
/// * `xs` — the observations whose spread to measure.
///
/// # Returns
///
/// `Σ (xᵢ − x̄)² / (n − 1)` with `n = xs.len()`, or `0.0` when `n < 2`.
#[must_use]
pub(crate) fn sample_variance(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    let m = mean(xs);
    let ss: f64 = xs.iter().map(|&x| (x - m) * (x - m)).sum();
    ss / count_to_f64(n - 1)
}

#[cfg(test)]
mod tests {
    use super::{count_to_f64, mean, sample_variance};

    /// The widening reproduces small counts exactly.
    #[test]
    fn count_widens_small_values_exactly() {
        assert!((count_to_f64(0) - 0.0).abs() < 1e-12, "zero");
        assert!((count_to_f64(1) - 1.0).abs() < 1e-12, "one");
        assert!((count_to_f64(150) - 150.0).abs() < 1e-12, "150");
    }

    /// The widening stays exact past the 32-bit boundary (both halves in play).
    #[test]
    fn count_widens_across_the_32_bit_boundary_exactly() {
        let n: usize = 5_000_000_000; // > 2^32, still far below 2^53
        assert!(
            (count_to_f64(n) - 5_000_000_000.0).abs() < 1e-3,
            "was {}",
            count_to_f64(n)
        );
    }

    /// The mean of a known sample matches the hand computation.
    #[test]
    fn mean_matches_hand_computation() {
        let m = mean(&[2.0, 4.0, 6.0]);
        assert!((m - 4.0).abs() < 1e-12, "mean was {m}");
    }

    /// An empty slice yields the `0.0` sentinel, not `NaN`.
    #[test]
    fn mean_of_empty_is_zero() {
        let m = mean(&[]);
        assert!(m.abs() < 1e-12, "empty mean was {m}");
    }

    /// The variance is Bessel-corrected (divides by `n − 1`).
    #[test]
    fn variance_is_bessel_corrected() {
        // var([2,4,6], ddof=1) = ((2-4)^2 + 0 + (6-4)^2) / (3-1) = 8/2 = 4.
        let v = sample_variance(&[2.0, 4.0, 6.0]);
        assert!((v - 4.0).abs() < 1e-12, "variance was {v}");
    }

    /// Fewer than two observations leaves the `ddof = 1` variance at `0.0`.
    #[test]
    fn variance_of_short_sample_is_zero() {
        assert!(sample_variance(&[]).abs() < 1e-12, "empty");
        assert!(sample_variance(&[3.0]).abs() < 1e-12, "singleton");
    }
}
