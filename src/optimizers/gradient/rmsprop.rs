//! `RMSProp` optimizer (Tieleman & Hinton 2012) — root-mean-square gradient scaling.

use crate::optimizers::{norm, ConvergenceStatus, Objective, OptimizeResult};

/// Decay rate for the running mean of squared gradients.
const DECAY: f64 = 0.9;
/// Numerical-stability constant added to the denominator.
const EPS: f64 = 1e-8;

/// Minimizes `obj` with the `RMSProp` update.
///
/// Maintains an exponentially decayed average of squared gradients and steps
/// each coordinate by `lr · g / (√v + ε)`. Deterministic given the inputs.
/// Converges when the gradient norm falls below `tol`.
///
/// # Arguments
///
/// * `obj` — the objective to minimize.
/// * `x0` — the starting point.
/// * `lr` — the learning rate.
/// * `max_iter` — the iteration budget.
/// * `tol` — the gradient-norm convergence threshold.
///
/// # Returns
///
/// An [`OptimizeResult`] reporting the located point, value, iterations, status.
///
/// # Examples
///
/// ```
/// use stats_claw::optimizers::gradient::rmsprop;
/// use stats_claw::optimizers::objectives::Quadratic;
/// use stats_claw::optimizers::ConvergenceStatus;
///
/// let obj = Quadratic::new(vec![1.0, 4.0]);
/// let r = rmsprop(&obj, &[0.0, 0.0], 0.05, 20_000, 1e-10);
/// assert!(matches!(r.status, ConvergenceStatus::Converged));
/// assert!((r.x[1] - 4.0).abs() < 1e-4, "x[1] was {}", r.x[1]);
/// ```
#[must_use]
pub fn rmsprop(
    obj: &impl Objective,
    x0: &[f64],
    lr: f64,
    max_iter: usize,
    tol: f64,
) -> OptimizeResult {
    let mut x = x0.to_vec();
    let mut v = vec![0.0; x.len()];
    let mut status = ConvergenceStatus::MaxIterReached;
    let mut iterations = 0;
    for step in 0..max_iter {
        iterations = step + 1;
        let g = obj.grad(&x);
        if norm(&g) < tol {
            status = ConvergenceStatus::Converged;
            break;
        }
        for ((xi, gi), vi) in x.iter_mut().zip(&g).zip(v.iter_mut()) {
            *vi = DECAY.mul_add(*vi, (1.0 - DECAY) * gi * gi);
            *xi -= lr * gi / (vi.sqrt() + EPS);
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
