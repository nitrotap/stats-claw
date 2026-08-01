//! Welford's online mean/variance accumulator.

use super::count_to_f64;

/// Welford's online accumulator for the running mean and variance of a stream.
///
/// Maintains the count, running mean, and the sum of squared deviations (`M2`) so
/// that the mean and the Bessel-corrected sample variance are available after any
/// number of updates without storing the values themselves. Numerically stable:
/// it avoids the catastrophic cancellation of the naive "sum of squares minus
/// square of sum" formula.
///
/// # Invariants
///
/// The struct holds exactly three scalar fields, so `size_of::<RunningMoments>()`
/// is constant regardless of stream length — the bounded-memory guarantee.
///
/// # Examples
///
/// ```
/// use stats_claw::streaming::RunningMoments;
///
/// let mut m = RunningMoments::new();
/// for x in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
///     m.update(x);
/// }
/// // Mean of the eight values is 5.0.
/// assert!((m.mean() - 5.0).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RunningMoments {
    /// Number of values consumed so far.
    count: u64,
    /// Running arithmetic mean of the values consumed so far.
    mean: f64,
    /// Running sum of squared deviations from the current mean (Welford's `M2`).
    m2: f64,
}

impl RunningMoments {
    /// Creates an empty accumulator that has consumed no values.
    ///
    /// # Returns
    ///
    /// A `RunningMoments` with zero count; [`Self::mean`] is `0.0` until the first
    /// [`Self::update`].
    #[must_use]
    pub const fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Folds one observation into the running summary using Welford's update.
    ///
    /// # Arguments
    ///
    /// * `x` — the next value of the stream. Any finite `f64`; `NaN`/`±∞`
    ///   propagate into the running statistics unchanged.
    pub fn update(&mut self, x: f64) {
        self.count += 1;
        let n = count_to_f64(self.count);
        let delta = x - self.mean;
        self.mean += delta / n;
        let delta2 = x - self.mean;
        self.m2 = delta.mul_add(delta2, self.m2);
    }

    /// Returns the running arithmetic mean of all values consumed so far.
    ///
    /// # Returns
    ///
    /// The mean, or `0.0` if no values have been consumed yet.
    #[must_use]
    pub const fn mean(&self) -> f64 {
        self.mean
    }

    /// Returns the running Bessel-corrected sample variance.
    ///
    /// # Returns
    ///
    /// The sample variance `M2 / (n - 1)`, or `0.0` when fewer than two values
    /// have been consumed (variance is undefined for a single observation, and
    /// `0.0` is the natural, panic-free convention here).
    #[must_use]
    pub fn variance(&self) -> f64 {
        if self.count < 2 {
            return 0.0;
        }
        self.m2 / count_to_f64(self.count - 1)
    }

    /// Returns the running sample standard deviation (the square root of
    /// [`Self::variance`]).
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Returns the number of values consumed so far.
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }
}

impl Default for RunningMoments {
    /// Returns an empty accumulator, equivalent to [`RunningMoments::new`].
    fn default() -> Self {
        Self::new()
    }
}

/// Kani formal-verification harnesses for Welford's accumulator.
///
/// Compiled only under `cargo kani` (behind `#[cfg(kani)]`); invisible to normal
/// build/test/clippy. They fold a bounded number of *symbolic finite* updates and
/// prove the invariants hold for every such stream, not the sampled fixtures a
/// `#[cfg(test)]` suite would use.
#[cfg(kani)]
mod verification {
    use super::RunningMoments;

    /// Upper bound on `|x|` for the variance proof.
    ///
    /// A fully symbolic *finite* `f64` (up to `f64::MAX ≈ 1.8e308`) breaks the
    /// non-negativity property: `x·x` and the running sums overflow to `±∞`, and
    /// `∞ − ∞` yields `NaN`, so `variance()` can be `NaN` for extreme-magnitude
    /// inputs — a genuine limitation, not a spurious solver artifact. Bounding
    /// `|x| ≤ 1e150` keeps every intermediate (`x·x ≤ 1e300`, and the few-term
    /// sums) comfortably below `f64::MAX`, isolating the pure sign argument.
    const MAX_ABS: f64 = 1e150;

    /// Draws a symbolic `f64` constrained to be finite and bounded by [`MAX_ABS`].
    ///
    /// Welford's monotone-`M2` argument holds only where the arithmetic stays
    /// finite; the module contract already documents that non-finite inputs
    /// propagate into the statistics unchanged, so the proofs scope to the
    /// non-overflowing finite regime.
    ///
    /// # Returns
    ///
    /// A finite `f64` with `|x| ≤ MAX_ABS`.
    fn any_bounded() -> f64 {
        let x: f64 = kani::any();
        kani::assume(x.is_finite());
        kani::assume(x.abs() <= MAX_ABS);
        x
    }

    /// Proves that after any three symbolic magnitude-bounded (`|x| ≤ MAX_ABS`)
    /// updates the accumulator neither panics nor overflows and reports a
    /// non-negative, non-`NaN` variance. Kani confirms the sign argument survives
    /// `f64` rounding, not just in exact arithmetic.
    ///
    /// The `M2` update adds `delta · delta2`, whose two factors are
    /// `(x − mean_old)` and `(x − mean_new) = delta · (n−1)/n`; they share a sign,
    /// so the exact product is `≥ 0` and the fused multiply-add of two non-negative
    /// reals rounds to a non-negative `f64`. Hence `M2 ≥ 0`, and the Bessel divisor
    /// `n − 1 > 0` once `count ≥ 2`, so `variance() ≥ 0`. (`assert!(v >= 0.0)` also
    /// rejects `NaN`, which is never `≥ 0`.) Three updates suffice to exercise the
    /// `count ≥ 2` variance path; the loop-free unrolling needs no unwind bound.
    #[kani::proof]
    fn moments_variance_non_negative() {
        let mut m = RunningMoments::new();
        m.update(any_bounded());
        m.update(any_bounded());
        m.update(any_bounded());
        let v = m.variance();
        assert!(v >= 0.0, "variance was negative or NaN: {v}");
        assert_eq!(m.count(), 3, "count diverged from the number of updates");
    }

    /// Proves a single symbolic finite update is panic-/overflow-free and that the
    /// one-observation variance convention (`0.0`, undefined for `n < 2`) holds
    /// exactly.
    #[kani::proof]
    fn moments_single_update_variance_zero() {
        let mut m = RunningMoments::new();
        m.update(any_bounded());
        let v = m.variance();
        assert!(
            v == 0.0,
            "single-sample variance must be exactly 0.0, was {v}"
        );
    }
}
