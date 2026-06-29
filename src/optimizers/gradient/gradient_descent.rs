//! Vanilla (batch) gradient descent — the worked pattern every learning-rate
//! optimizer follows.

use crate::optimizers::{norm, ConvergenceStatus, Objective, OptimizeResult};

/// Minimizes `obj` by full-gradient steps of fixed learning rate.
///
/// At each step the point moves against the gradient: `x ← x − lr·∇f(x)`. The
/// run stops early (reporting [`ConvergenceStatus::Converged`]) when the gradient
/// norm drops below `tol`, otherwise it runs the full `max_iter` budget and
/// reports [`ConvergenceStatus::MaxIterReached`].
///
/// # Arguments
///
/// * `obj` — the objective to minimize.
/// * `x0` — the starting point; its length is the problem dimension.
/// * `lr` — the learning rate (step size); must be positive and small enough for
///   the problem's curvature or the iterates diverge.
/// * `max_iter` — the maximum number of gradient steps.
/// * `tol` — the gradient-norm convergence threshold.
///
/// # Returns
///
/// An [`OptimizeResult`] with the located point, its objective value, the number
/// of iterations performed, and the convergence status.
///
/// # Examples
///
/// ```
/// use stats_claw::optimizers::gradient::gradient_descent;
/// use stats_claw::optimizers::objectives::Quadratic;
/// use stats_claw::optimizers::ConvergenceStatus;
///
/// let obj = Quadratic::new(vec![3.0, -2.0]);
/// let r = gradient_descent(&obj, &[0.0, 0.0], 0.1, 10_000, 1e-12);
/// assert!(matches!(r.status, ConvergenceStatus::Converged));
/// assert!((r.x[0] - 3.0).abs() < 1e-6);
/// ```
#[must_use]
pub fn gradient_descent(
    obj: &impl Objective,
    x0: &[f64],
    lr: f64,
    max_iter: usize,
    tol: f64,
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
            *xi -= lr * gi;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizers::objectives::Quadratic;

    #[test]
    fn budget_limited_run_reports_max_iter() {
        let obj = Quadratic::new(vec![3.0, -2.0]);
        let r = gradient_descent(&obj, &[0.0, 0.0], 0.1, 1, 1e-12);
        assert_eq!(r.status, ConvergenceStatus::MaxIterReached);
        assert_eq!(r.iterations, 1, "iterations was {}", r.iterations);
    }
}
