//! Newton's method with an inner conjugate-gradient solve (Newton-CG).
//!
//! The stats-claw counterpart of `scipy.optimize.minimize(method="Newton-CG")`.

use crate::optimizers::{
    dot, line_search, matvec, norm, step, ConvergenceStatus, Objective, OptimizeResult,
};

/// Minimizes `obj` by Newton's method, solving `H·p = −g` with inner CG.
///
/// Each outer step approximately solves the Newton system using a linear
/// conjugate-gradient loop on the (finite-difference or analytic) Hessian, then
/// takes an Armijo line-searched step along the resulting direction. Falls back
/// to steepest descent when the Newton direction is not a descent direction.
/// Converges when the gradient norm falls below `tol`.
///
/// # Arguments
///
/// * `obj` — the objective to minimize; supplies value/grad and a Hessian
///   (analytic if overridden, otherwise finite-difference).
/// * `x0` — the starting point.
/// * `max_iter` — the outer-iteration budget.
/// * `tol` — the gradient-norm convergence threshold.
///
/// # Returns
///
/// An [`OptimizeResult`] reporting the located point, value, iterations, status.
///
/// # Examples
///
/// ```
/// use stats_claw::optimizers::second_order::newton;
/// use stats_claw::optimizers::objectives::Quadratic;
/// use stats_claw::optimizers::ConvergenceStatus;
///
/// let obj = Quadratic::new(vec![3.0, -2.0]);
/// let r = newton(&obj, &[0.0, 0.0], 100, 1e-10);
/// assert!(matches!(r.status, ConvergenceStatus::Converged));
/// assert!((r.x[0] - 3.0).abs() < 1e-6, "x[0] was {}", r.x[0]);
/// ```
#[must_use]
pub fn newton(obj: &impl Objective, x0: &[f64], max_iter: usize, tol: f64) -> OptimizeResult {
    let mut x = x0.to_vec();
    let mut status = ConvergenceStatus::MaxIterReached;
    let mut iterations = 0;
    for step_idx in 0..max_iter {
        iterations = step_idx + 1;
        let g = obj.grad(&x);
        if norm(&g) < tol {
            status = ConvergenceStatus::Converged;
            break;
        }
        let hess = obj.hessian(&x);
        let neg_g: Vec<f64> = g.iter().map(|gi| -gi).collect();
        let mut dir = cg_solve(&hess, &neg_g);
        // Guard against a non-descent direction (e.g. indefinite Hessian).
        if dot(&dir, &g) >= 0.0 {
            dir = neg_g;
        }
        let alpha = line_search(obj, &x, &dir, &g);
        x = step(&x, alpha, &dir);
    }
    let fx = obj.value(&x);
    OptimizeResult {
        x,
        fx,
        iterations,
        status,
    }
}

/// Solves `A·p = b` by linear conjugate gradient for a symmetric positive-
/// definite-ish `A`; returns the approximate solution after at most `n` steps.
// Single-char names (a, b, p, r, d) are the canonical linear-CG notation.
#[allow(clippy::many_single_char_names)]
fn cg_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut p = vec![0.0; n];
    let mut r = b.to_vec();
    let mut d = r.clone();
    let mut rs_old = dot(&r, &r);
    for _ in 0..n.max(1) {
        if rs_old.sqrt() < 1e-12 {
            break;
        }
        let ad = matvec(a, &d);
        let denom = dot(&d, &ad);
        if denom.abs() < 1e-30 {
            break;
        }
        let alpha = rs_old / denom;
        for (pi, di) in p.iter_mut().zip(&d) {
            *pi += alpha * di;
        }
        for (ri, adi) in r.iter_mut().zip(&ad) {
            *ri -= alpha * adi;
        }
        let rs_new = dot(&r, &r);
        let beta = rs_new / rs_old;
        for (di, ri) in d.iter_mut().zip(&r) {
            *di = beta.mul_add(*di, *ri);
        }
        rs_old = rs_new;
    }
    p
}
