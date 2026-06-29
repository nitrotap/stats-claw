//! `AdaGrad` optimizer (Duchi, Hazan & Singer 2011) — per-coordinate adaptive rates.

use crate::optimizers::{norm, ConvergenceStatus, Objective, OptimizeResult};

/// Numerical-stability constant added to the denominator.
const EPS: f64 = 1e-8;

/// Minimizes `obj` with the `AdaGrad` update.
///
/// Accumulates the sum of squared gradients per coordinate and steps by
/// `lr · g / (√G + ε)`, so frequently-large-gradient coordinates take smaller
/// steps over time. Deterministic given the inputs. Converges when the gradient
/// norm falls below `tol`.
///
/// # Arguments
///
/// * `obj` — the objective to minimize.
/// * `x0` — the starting point.
/// * `lr` — the base learning rate.
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
/// use stats_claw::optimizers::gradient::adagrad;
/// use stats_claw::optimizers::objectives::Quadratic;
/// use stats_claw::optimizers::ConvergenceStatus;
///
/// let obj = Quadratic::new(vec![3.0, -2.0]);
/// let r = adagrad(&obj, &[0.0, 0.0], 0.5, 50_000, 1e-10);
/// assert!(matches!(r.status, ConvergenceStatus::Converged));
/// assert!((r.x[0] - 3.0).abs() < 1e-4, "x[0] was {}", r.x[0]);
/// ```
#[must_use]
pub fn adagrad(
    obj: &impl Objective,
    x0: &[f64],
    lr: f64,
    max_iter: usize,
    tol: f64,
) -> OptimizeResult {
    let mut x = x0.to_vec();
    let mut acc = vec![0.0; x.len()];
    let mut status = ConvergenceStatus::MaxIterReached;
    let mut iterations = 0;
    for step in 0..max_iter {
        iterations = step + 1;
        let g = obj.grad(&x);
        if norm(&g) < tol {
            status = ConvergenceStatus::Converged;
            break;
        }
        for ((xi, gi), gacc) in x.iter_mut().zip(&g).zip(acc.iter_mut()) {
            *gacc += gi * gi;
            *xi -= lr * gi / (gacc.sqrt() + EPS);
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
