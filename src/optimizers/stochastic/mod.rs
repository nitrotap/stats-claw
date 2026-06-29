//! Derivative-free stochastic optimizers: simulated annealing and a genetic
//! (differential-evolution) optimizer.
//!
//! Both explore a bounded box and use only objective *values* (no gradient),
//! driven by a seeded [`SplitMix64`](crate::rng::SplitMix64) so a fixed seed
//! reproduces the run exactly. They are the stats-claw
//! counterparts of `scipy.optimize.dual_annealing` and
//! `scipy.optimize.differential_evolution`.

pub mod genetic;
pub mod simulated_annealing;

pub use genetic::genetic;
pub use simulated_annealing::simulated_annealing;

/// Maps a unit-interval draw `u ∈ [0, 1)` into the closed box coordinate
/// `[lo, hi]`.
///
/// # Arguments
///
/// * `u` — a uniform draw in `[0, 1)`.
/// * `lo`, `hi` — the lower and upper bounds of the coordinate.
///
/// # Returns
///
/// `lo + u·(hi − lo)`.
pub(crate) fn scale(u: f64, lo: f64, hi: f64) -> f64 {
    u.mul_add(hi - lo, lo)
}

/// Clamps `v` into `[lo, hi]`.
pub(crate) const fn clamp(v: f64, lo: f64, hi: f64) -> f64 {
    v.clamp(lo, hi)
}
