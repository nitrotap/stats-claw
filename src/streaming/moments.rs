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
