//! Adam optimizer (Kingma & Ba 2014) — adaptive moment estimation.

use crate::optimizers::{norm, ConvergenceStatus, Objective, OptimizeResult};

/// Default exponential-decay rate for the first moment (mean of gradients).
const BETA1: f64 = 0.9;
/// Default exponential-decay rate for the second moment (uncentered variance).
const BETA2: f64 = 0.999;
/// Numerical-stability constant added to the denominator.
const EPS: f64 = 1e-8;

/// Minimizes `obj` with the Adam adaptive-moment update.
///
/// Maintains bias-corrected estimates of the first and second gradient moments
/// and steps each coordinate by `lr · m̂ / (√v̂ + ε)`. Deterministic given the
/// inputs (no randomness). Converges when the gradient norm falls below `tol`.
///
/// # Arguments
///
/// * `obj` — the objective to minimize.
/// * `x0` — the starting point.
/// * `lr` — the base learning rate (step scale).
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
/// use stats_claw::optimizers::gradient::adam;
/// use stats_claw::optimizers::objectives::Quadratic;
/// use stats_claw::optimizers::ConvergenceStatus;
///
/// let obj = Quadratic::new(vec![2.0, -1.0]);
/// let r = adam(&obj, &[0.0, 0.0], 0.01, 10_000, 1e-10);
/// assert!(matches!(r.status, ConvergenceStatus::Converged));
/// assert!((r.x[0] - 2.0).abs() < 1e-4, "x[0] was {}", r.x[0]);
/// ```
// Single-char names (m, v, g, t) are the canonical Adam-paper notation.
#[allow(clippy::many_single_char_names)]
#[must_use]
pub fn adam(
    obj: &impl Objective,
    x0: &[f64],
    lr: f64,
    max_iter: usize,
    tol: f64,
) -> OptimizeResult {
    let mut x = x0.to_vec();
    let n = x.len();
    let mut m = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut status = ConvergenceStatus::MaxIterReached;
    let mut iterations = 0;
    let mut t = 0.0_f64;
    for step in 0..max_iter {
        iterations = step + 1;
        let g = obj.grad(&x);
        if norm(&g) < tol {
            status = ConvergenceStatus::Converged;
            break;
        }
        t += 1.0;
        let bc1 = 1.0 - BETA1.powf(t);
        let bc2 = 1.0 - BETA2.powf(t);
        for (((xi, gi), mi), vi) in x.iter_mut().zip(&g).zip(m.iter_mut()).zip(v.iter_mut()) {
            *mi = BETA1.mul_add(*mi, (1.0 - BETA1) * gi);
            *vi = BETA2.mul_add(*vi, (1.0 - BETA2) * gi * gi);
            let m_hat = *mi / bc1;
            let v_hat = *vi / bc2;
            *xi -= lr * m_hat / (v_hat.sqrt() + EPS);
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
