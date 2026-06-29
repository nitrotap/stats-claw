//! Nonlinear conjugate gradient (Fletcher–Reeves) with backtracking line search.
//!
//! The stats-claw counterpart of `scipy.optimize.minimize(method="CG")`.

use crate::optimizers::{
    dot, line_search, norm, step, ConvergenceStatus, Objective, OptimizeResult,
};

/// Minimizes `obj` by the Fletcher–Reeves nonlinear conjugate-gradient method.
///
/// The search direction starts as steepest descent and is updated by
/// `d ← −g + β·d_prev` with the Fletcher–Reeves coefficient
/// `β = gᵀg / g_prevᵀg_prev`, restarting to steepest descent every `n` steps to
/// preserve conjugacy. Each step length is chosen by an Armijo backtracking line
/// search. Converges when the gradient norm falls below `tol`.
///
/// # Arguments
///
/// * `obj` — the objective to minimize.
/// * `x0` — the starting point.
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
/// use stats_claw::optimizers::gradient::conjugate_gradient;
/// use stats_claw::optimizers::objectives::Quadratic;
/// use stats_claw::optimizers::ConvergenceStatus;
///
/// let obj = Quadratic::new(vec![3.0, -1.0]);
/// let r = conjugate_gradient(&obj, &[0.0, 0.0], 1_000, 1e-10);
/// assert!(matches!(r.status, ConvergenceStatus::Converged));
/// assert!((r.x[0] - 3.0).abs() < 1e-6, "x[0] was {}", r.x[0]);
/// ```
#[must_use]
pub fn conjugate_gradient(
    obj: &impl Objective,
    x0: &[f64],
    max_iter: usize,
    tol: f64,
) -> OptimizeResult {
    let mut x = x0.to_vec();
    let n = x.len().max(1);
    let mut g = obj.grad(&x);
    let mut dir: Vec<f64> = g.iter().map(|gi| -gi).collect();
    let mut status = ConvergenceStatus::MaxIterReached;
    let mut iterations = 0;
    for step_idx in 0..max_iter {
        iterations = step_idx + 1;
        if norm(&g) < tol {
            status = ConvergenceStatus::Converged;
            break;
        }
        let alpha = line_search(obj, &x, &dir, &g);
        x = step(&x, alpha, &dir);
        let g_new = obj.grad(&x);
        let denom = dot(&g, &g);
        let beta = if step_idx % n == n - 1 || denom == 0.0 {
            0.0
        } else {
            dot(&g_new, &g_new) / denom
        };
        dir = g_new
            .iter()
            .zip(&dir)
            .map(|(gi, di)| beta.mul_add(*di, -gi))
            .collect();
        g = g_new;
    }
    let fx = obj.value(&x);
    OptimizeResult {
        x,
        fx,
        iterations,
        status,
    }
}
