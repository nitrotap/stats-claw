//! Deterministic pseudo-random number generation and sampling primitives.
//!
//! This module provides the framework's reproducible PRNG (`SplitMix64`) and the
//! low-level samplers that distributions build their `Sample` implementations
//! on. Determinism is a hard requirement: identical seeds must produce
//! identical streams so that golden-fixture equivalence tests are stable.

use std::f64::consts::PI;

/// Increment used to walk the `SplitMix64` state (the golden-ratio constant).
const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
/// First mixing multiplier from Steele, Lea & Flood (2014).
const MIX_A: u64 = 0xBF58_476D_1CE4_E5B9;
/// Second mixing multiplier from Steele, Lea & Flood (2014).
const MIX_B: u64 = 0x94D0_49BB_1331_11EB;
/// `2^53`, the number of representable mantissa steps for a uniform `f64`.
const TWO_POW_53: f64 = 9_007_199_254_740_992.0;
/// `2^32`, used to widen a `u64` to `f64` without an `as` cast.
const TWO_POW_32: f64 = 4_294_967_296.0;

/// Widens a `u64` to the nearest `f64` without an `as` cast.
///
/// Splits the value into 32-bit halves (each losslessly representable as `f64`)
/// and recombines them, so the `style.rs` no-`as` guard is satisfied while large
/// values still round exactly as a single cast would.
///
/// # Arguments
///
/// * `x` — the value to widen.
///
/// # Returns
///
/// `x` as an `f64` (exact for the 53-bit values this module feeds it).
fn u64_to_f64(x: u64) -> f64 {
    let hi = u32::try_from(x >> 32).unwrap_or(0);
    let lo = u32::try_from(x & 0xFFFF_FFFF).unwrap_or(0);
    f64::from(hi).mul_add(TWO_POW_32, f64::from(lo))
}

/// `SplitMix64` generator (Steele, Lea & Flood 2014).
///
/// A single-`u64`-state PRNG chosen for its tiny, portable, branch-free core: it
/// produces a byte-identical stream for a given seed on every platform, which the
/// equivalence harness relies on. That cross-platform guarantee covers
/// [`next_u64`](Self::next_u64) and [`next_f64`](Self::next_f64), which are
/// integer arithmetic plus one exact division by a power of two; it stops at
/// [`standard_normal`](Self::standard_normal), whose Box–Muller transform routes
/// through the platform math library — see that method's portability note. The
/// generator also caches the second Box–Muller variate, so `standard_normal`
/// costs one transcendental pair per two draws.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    /// Current generator state, advanced by [`GOLDEN_GAMMA`] each draw.
    state: u64,
    /// The spare standard-normal variate from the previous Box–Muller draw.
    cached_normal: Option<f64>,
}

impl SplitMix64 {
    /// Creates a generator seeded with `seed`.
    ///
    /// # Arguments
    ///
    /// * `seed` — initial state; any `u64` is valid (including `0`).
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self {
            state: seed,
            cached_normal: None,
        }
    }

    /// Draws the next 64-bit value and advances the state.
    ///
    /// # Returns
    ///
    /// A pseudo-random `u64` uniformly distributed over the full range.
    pub const fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(MIX_A);
        z = (z ^ (z >> 27)).wrapping_mul(MIX_B);
        z ^ (z >> 31)
    }

    /// Draws a uniform value in `[0, 1)` with 53 bits of resolution.
    ///
    /// # Returns
    ///
    /// A pseudo-random `f64` in the half-open unit interval.
    pub fn next_f64(&mut self) -> f64 {
        u64_to_f64(self.next_u64() >> 11) / TWO_POW_53
    }

    /// Draws a standard-normal variate via Box–Muller, caching the spare.
    ///
    /// The transform produces two independent N(0,1) variates per call; the
    /// second is cached and returned on the following call, halving the
    /// transcendental cost over a long stream.
    ///
    /// # Portability
    ///
    /// The draw is deterministic: one seed gives one sequence, and the `u64`
    /// stream underneath it is byte-identical on every target. The variate
    /// itself is not. Box–Muller evaluates `ln`, `sin`, and `cos` through the
    /// platform math library, which is accurate to well under an ulp but is not
    /// correctly rounded, so different targets — and even different
    /// optimisation levels on one target — can disagree in the last ulp or two.
    /// Compare standard-normal draws, and any statistic accumulated from them,
    /// with a tolerance; never bit-for-bit across builds or machines.
    ///
    /// # Returns
    ///
    /// A pseudo-random `f64` distributed as N(0, 1).
    pub fn standard_normal(&mut self) -> f64 {
        if let Some(z) = self.cached_normal.take() {
            return z;
        }
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * PI * u2;
        self.cached_normal = Some(r * theta.sin());
        r * theta.cos()
    }
}

/// Kani formal-verification harnesses for the PRNG core.
///
/// These prove properties over *all* inputs (symbolic `kani::any()` state), not
/// the sampled inputs the `#[cfg(test)]` suite exercises. They are compiled only
/// under `cargo kani` (behind `#[cfg(kani)]`) and are invisible to normal
/// build/test/clippy. Run with e.g.
/// `cargo kani --harness rng_next_f64_in_unit_interval -p stats-claw`.
#[cfg(kani)]
mod verification {
    use super::{SplitMix64, TWO_POW_32, TWO_POW_53, u64_to_f64};

    /// Proves [`u64_to_f64`] is a faithful widening for every 53-bit input: the
    /// split-and-recombine reconstruction is finite, non-negative, and stays
    /// strictly below `2^53` (the property [`SplitMix64::next_f64`] relies on to
    /// keep its quotient in `[0, 1)`).
    ///
    /// Restricting to `x < 2^53` matches the module's only caller
    /// (`next_u64() >> 11`) and keeps every intermediate exactly representable, so
    /// the widening is provably lossless there.
    #[kani::proof]
    fn rng_u64_to_f64_faithful() {
        let x: u64 = kani::any();
        kani::assume(x < (1u64 << 53));
        let y = u64_to_f64(x);
        assert!(y.is_finite(), "u64_to_f64 produced non-finite output");
        assert!(y >= 0.0, "u64_to_f64 produced a negative value");
        assert!(y < TWO_POW_53, "u64_to_f64 escaped the 2^53 window");
    }

    /// Proves the 32-bit split constants are consistent: the widened halves of a
    /// symbolic `u64` recombine to a finite, non-negative `f64` no greater than
    /// `2^64` — i.e. the widening never overflows to `∞` over the full range.
    #[kani::proof]
    fn rng_u64_to_f64_no_overflow() {
        let x: u64 = kani::any();
        let y = u64_to_f64(x);
        assert!(
            y.is_finite(),
            "full-range u64_to_f64 overflowed to non-finite"
        );
        assert!(y >= 0.0, "full-range u64_to_f64 produced a negative value");
        // Both 32-bit halves are < 2^32, so the exact recombination is < 2^64. In
        // `f64` the largest inputs (near `2^64 − 1`) round up to exactly `2^64`, so
        // the tight, provable ceiling is `<= 2^64`, not `<`.
        assert!(y <= TWO_POW_32 * TWO_POW_32, "u64_to_f64 exceeded 2^64");
    }

    /// Proves [`SplitMix64::next_u64`] is panic-/overflow-free for a fully symbolic
    /// generator state. Every arithmetic step is `wrapping_*`, so this discharges
    /// Kani's default overflow checks over the entire state space.
    #[kani::proof]
    fn rng_next_u64_no_panic() {
        let state: u64 = kani::any();
        let mut rng = SplitMix64::new(state);
        let _ = rng.next_u64();
    }

    /// Proves [`SplitMix64::next_f64`] lands in the half-open unit interval
    /// `[0.0, 1.0)` for *every* possible generator state — the uniform-sampler
    /// contract the whole distributions layer builds on, verified symbolically
    /// rather than over the 10 000 sampled draws the unit test checks.
    #[kani::proof]
    fn rng_next_f64_in_unit_interval() {
        let state: u64 = kani::any();
        let mut rng = SplitMix64::new(state);
        let u = rng.next_f64();
        assert!(u >= 0.0, "next_f64 produced a negative value: {u}");
        assert!(u < 1.0, "next_f64 reached or exceeded 1.0: {u}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = SplitMix64::new(42);
        let mut b = SplitMix64::new(42);
        for _ in 0..1000 {
            assert_eq!(
                a.next_u64(),
                b.next_u64(),
                "identical seeds must yield identical streams"
            );
        }
    }

    #[test]
    fn uniform_in_unit_interval() {
        let mut r = SplitMix64::new(7);
        for _ in 0..10_000 {
            let u = r.next_f64();
            assert!((0.0..1.0).contains(&u), "next_f64 escaped [0, 1): {u}");
        }
    }

    #[test]
    fn standard_normal_is_deterministic_and_finite() {
        let mut a = SplitMix64::new(99);
        let mut b = SplitMix64::new(99);
        for _ in 0..1000 {
            let za = a.standard_normal();
            assert!(za.is_finite(), "standard_normal produced non-finite {za}");
            assert!(
                (za - b.standard_normal()).abs() < 1e-15,
                "standard_normal stream diverged for identical seeds"
            );
        }
    }
}
