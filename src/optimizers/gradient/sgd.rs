//! Stochastic gradient descent with seeded coordinate-subsampled steps.
//!
//! Pure SGD samples a mini-batch each step; with an analytic [`Objective`] that
//! exposes only a full gradient, this implementation injects the stochasticity by
//! sampling a random subset of gradient coordinates to update each step (a
//! coordinate-wise stochastic step), driven by [`SplitMix64`] so a fixed seed
//! reproduces the run byte-for-byte.

use crate::optimizers::{norm, ConvergenceStatus, Objective, OptimizeResult};
use crate::rng::SplitMix64;

/// Minimizes `obj` by seeded stochastic coordinate gradient steps.
///
/// Each step computes the full gradient, then updates each coordinate only with
/// probability one-half (chosen via `rng`), giving a reproducible stochastic
/// trajectory. Convergence is declared when the full gradient norm falls below
/// `tol`.
///
/// # Arguments
///
/// * `obj` — the objective to minimize.
/// * `x0` — the starting point.
/// * `lr` — the learning rate.
/// * `max_iter` — the iteration budget.
/// * `tol` — the gradient-norm convergence threshold.
/// * `rng` — the seeded PRNG controlling which coordinates update; the same seed
///   reproduces the run exactly.
///
/// # Returns
///
/// An [`OptimizeResult`] reporting the located point, value, iteration count, and
/// status.
///
/// # Examples
///
/// ```
/// use stats_claw::optimizers::gradient::sgd;
/// use stats_claw::optimizers::objectives::Quadratic;
/// use stats_claw::rng::SplitMix64;
///
/// let obj = Quadratic::new(vec![2.0, -1.0]);
/// let mut rng = SplitMix64::new(42);
/// let r = sgd(&obj, &[0.0, 0.0], 0.1, 50_000, 1e-10, &mut rng);
/// // SGD is stochastic — just assert the objective decreased.
/// assert!(r.fx < 1.0, "f(x) was {}", r.fx);
/// ```
#[must_use]
pub fn sgd(
    obj: &impl Objective,
    x0: &[f64],
    lr: f64,
    max_iter: usize,
    tol: f64,
    rng: &mut SplitMix64,
) -> OptimizeResult {
    let mut x = x0.to_vec();
    let mut status = ConvergenceStatus::MaxIterReached;
    let mut iterations = 0;
    for step in 0..max_iter {
        iterations = step + 1;
        let g = obj.grad(&x);
        if norm(&g) < tol {
            status = ConvergenceStatus::Converged;
            break;
        }
        for (xi, gi) in x.iter_mut().zip(&g) {
            if rng.next_f64() < 0.5 {
                *xi -= lr * gi;
            }
        }
    }
    let fx = obj.value(&x);
    OptimizeResult {
        x,
        fx,
        iterations,
        status,
    }
}
