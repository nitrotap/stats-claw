//! The P² streaming-quantile estimator (Jain & Chlamtac, 1985).
//
// `indexing_slicing` is allowed in this module only: every index is a
// compile-time constant or a loop bound in `0..5` / `1..4` indexing the
// fixed-length `[_; 5]` marker arrays, so the access is provably in bounds.
// Rewriting each as `.get(..).unwrap_or(..)` would obscure the dense P²
// arithmetic and add unreachable fallbacks, so a scoped allow is the clearer
// choice here (no `unwrap`/`as`/`unsafe` is introduced).
#![allow(clippy::indexing_slicing)]

use super::position_as_f64;

/// Online estimator of a single p-quantile using the P² algorithm.
///
/// Tracks five markers — the running minimum, the p/2, p, and (1+p)/2 quantiles,
/// and the running maximum — and adjusts their heights with a piecewise-parabolic
/// (hence "P²") interpolation as each value arrives. It estimates the quantile in
/// O(1) time and O(1) memory per update without storing or sorting the stream,
/// trading a small approximation error for unbounded scalability.
///
/// # Invariants
///
/// All state lives in fixed-length arrays (five markers), so
/// `size_of::<P2Quantile>()` is constant regardless of how many values have been
/// consumed — the bounded-memory guarantee.
///
/// # Examples
///
/// ```
/// use stats_claw::streaming::P2Quantile;
///
/// let mut q = P2Quantile::new(0.5);
/// for x in 1..=99 {
///     q.update(f64::from(x));
/// }
/// // Median of 1..=99 is 50; P² lands within a couple of integers on this ramp.
/// assert!((q.value() - 50.0).abs() < 2.0);
/// ```
#[derive(Debug, Clone)]
pub struct P2Quantile {
    /// Number of values consumed so far.
    count: u64,
    /// Marker heights (the running quantile estimates), ascending.
    heights: [f64; 5],
    /// Integer marker positions (1-based, as in the original paper).
    positions: [i64; 5],
    /// Desired (real-valued) marker positions.
    desired: [f64; 5],
    /// Per-marker increments to the desired positions on each new value.
    increments: [f64; 5],
}

impl P2Quantile {
    /// Creates an estimator for the `p`-quantile.
    ///
    /// # Arguments
    ///
    /// * `p` — the quantile probability, expected in `[0, 1]` (e.g. `0.5` for the
    ///   median, `0.99` for the 99th percentile). Values outside the range are not
    ///   rejected here — the estimate is simply meaningless, mirroring the
    ///   panic-free convention of the rest of the crate.
    ///
    /// # Returns
    ///
    /// A fresh estimator with no values consumed; [`Self::value`] is `0.0` until
    /// the first [`Self::update`].
    #[must_use]
    pub fn new(p: f64) -> Self {
        Self {
            count: 0,
            heights: [0.0; 5],
            positions: [1, 2, 3, 4, 5],
            desired: [
                1.0,
                2.0_f64.mul_add(p, 1.0),
                4.0_f64.mul_add(p, 1.0),
                2.0_f64.mul_add(p, 3.0),
                5.0,
            ],
            increments: [0.0, p / 2.0, p, f64::midpoint(1.0, p), 1.0],
        }
    }

    /// Folds one observation into the estimator.
    ///
    /// # Arguments
    ///
    /// * `x` — the next value of the stream.
    pub fn update(&mut self, x: f64) {
        if self.count < 5 {
            self.bootstrap(x);
            return;
        }
        let k = self.locate_cell(x);
        for marker in (k + 1)..5 {
            self.positions[marker] += 1;
        }
        for marker in 0..5 {
            self.desired[marker] += self.increments[marker];
        }
        self.adjust_interior();
        self.count += 1;
    }

    /// Returns the current quantile estimate: the middle marker height once five
    /// values are in, the largest seen value during warm-up, or `0.0` if empty.
    #[must_use]
    pub fn value(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }
        if self.count < 5 {
            let last = usize::try_from(self.count).unwrap_or(0).saturating_sub(1);
            return self.heights.get(last).copied().unwrap_or(0.0);
        }
        self.heights[2]
    }

    /// Returns the number of values consumed so far.
    #[must_use]
    pub const fn count(&self) -> u64 {
        self.count
    }

    /// Fills the initial five markers, keeping them sorted, before P² proper runs.
    fn bootstrap(&mut self, x: f64) {
        let slot = usize::try_from(self.count).unwrap_or(0);
        if let Some(h) = self.heights.get_mut(slot) {
            *h = x;
        }
        self.count += 1;
        if self.count == 5 {
            self.heights
                .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }
    }

    /// Finds the marker cell `x` falls into and clamps the running extrema.
    ///
    /// # Returns
    ///
    /// The index `k` in `0..4` such that `heights[k] <= x < heights[k+1]`, after
    /// extending the min/max markers to cover `x`.
    fn locate_cell(&mut self, x: f64) -> usize {
        if x < self.heights[0] {
            self.heights[0] = x;
            return 0;
        }
        for k in 0..4 {
            if x < self.heights[k + 1] {
                return k;
            }
        }
        self.heights[4] = x;
        3
    }

    /// Adjusts the three interior markers toward their desired positions using the
    /// parabolic formula, falling back to linear when the parabola would break the
    /// ascending order.
    fn adjust_interior(&mut self) {
        for i in 1..4 {
            let d = self.desired[i] - position_as_f64(self.positions[i]);
            let gap_hi = self.positions[i + 1] - self.positions[i];
            let gap_lo = self.positions[i] - self.positions[i - 1];
            if (d >= 1.0 && gap_hi > 1) || (d <= -1.0 && gap_lo > 1) {
                let dir = d.signum();
                let candidate = self.parabolic(i, dir);
                self.heights[i] =
                    if self.heights[i - 1] < candidate && candidate < self.heights[i + 1] {
                        candidate
                    } else {
                        self.linear(i, dir)
                    };
                self.positions[i] += i64::from(dir >= 0.0) * 2 - 1;
            }
        }
    }

    /// Piecewise-parabolic prediction (P²) of marker `i`'s new height.
    fn parabolic(&self, i: usize, dir: f64) -> f64 {
        let pos_lo = position_as_f64(self.positions[i - 1]);
        let pos_mid = position_as_f64(self.positions[i]);
        let pos_hi = position_as_f64(self.positions[i + 1]);
        let upper =
            (pos_mid - pos_lo + dir) * (self.heights[i + 1] - self.heights[i]) / (pos_hi - pos_mid);
        let lower =
            (pos_hi - pos_mid - dir) * (self.heights[i] - self.heights[i - 1]) / (pos_mid - pos_lo);
        (dir / (pos_hi - pos_lo)).mul_add(upper + lower, self.heights[i])
    }

    /// Linear fallback prediction of marker `i`'s new height.
    fn linear(&self, i: usize, dir: f64) -> f64 {
        let neighbor = if dir >= 0.0 { i + 1 } else { i - 1 };
        let pos_i = position_as_f64(self.positions[i]);
        let pos_neighbor = position_as_f64(self.positions[neighbor]);
        self.heights[i] + dir * (self.heights[neighbor] - self.heights[i]) / (pos_neighbor - pos_i)
    }
}
