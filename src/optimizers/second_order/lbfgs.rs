//! Limited-memory BFGS (L-BFGS) with the two-loop recursion and line search.
//!
//! The stats-claw counterpart of `scipy.optimize.minimize(method="L-BFGS-B")`
//! (unconstrained — no bounds are applied here).

use std::collections::VecDeque;

use crate::optimizers::{
    ConvergenceStatus, Objective, OptimizeResult, dot, line_search, norm, step,
};

/// History length (number of `(s, y)` correction pairs) retained.
const HISTORY: usize = 10;

/// Minimizes `obj` by L-BFGS using a history of `m = 10` correction pairs.
///
/// The search direction is `−H·g`, where `H` (the inverse-Hessian approximation)
/// is applied implicitly to the gradient via the two-loop recursion over the
/// stored `(s, y)` pairs; the step length comes from an Armijo backtracking line
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
/// use stats_claw::optimizers::second_order::lbfgs;
/// use stats_claw::optimizers::objectives::Quadratic;
/// use stats_claw::optimizers::ConvergenceStatus;
///
/// let obj = Quadratic::new(vec![3.0, -2.0]);
/// let r = lbfgs(&obj, &[0.0, 0.0], 1_000, 1e-10);
/// assert!(matches!(r.status, ConvergenceStatus::Converged));
/// assert!((r.x[0] - 3.0).abs() < 1e-6, "x[0] was {}", r.x[0]);
/// ```
#[must_use]
pub fn lbfgs(obj: &impl Objective, x0: &[f64], max_iter: usize, tol: f64) -> OptimizeResult {
    let mut x = x0.to_vec();
    let mut g = obj.grad(&x);
    let mut s_hist: VecDeque<Vec<f64>> = VecDeque::new();
    let mut y_hist: VecDeque<Vec<f64>> = VecDeque::new();
    let mut rho_hist: VecDeque<f64> = VecDeque::new();
    let mut status = ConvergenceStatus::MaxIterReached;
    let mut iterations = 0;
    for step_idx in 0..max_iter {
        iterations = step_idx + 1;
        if norm(&g) < tol {
            status = ConvergenceStatus::Converged;
            break;
        }
        let dir = two_loop(&g, &s_hist, &y_hist, &rho_hist);
        let alpha = line_search(obj, &x, &dir, &g);
        let x_new = step(&x, alpha, &dir);
        let g_new = obj.grad(&x_new);
        let s: Vec<f64> = x_new.iter().zip(&x).map(|(a, b)| a - b).collect();
        let y: Vec<f64> = g_new.iter().zip(&g).map(|(a, b)| a - b).collect();
        let sy = dot(&s, &y);
        if sy > 1e-12 {
            if s_hist.len() == HISTORY {
                s_hist.pop_front();
                y_hist.pop_front();
                rho_hist.pop_front();
            }
            rho_hist.push_back(1.0 / sy);
            s_hist.push_back(s);
            y_hist.push_back(y);
        }
        x = x_new;
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

/// Computes the L-BFGS search direction `−H·g` via the two-loop recursion.
// Single-char names (g, q, s, y, a) are the canonical Nocedal & Wright notation.
#[allow(clippy::many_single_char_names)]
fn two_loop(
    g: &[f64],
    s_hist: &VecDeque<Vec<f64>>,
    y_hist: &VecDeque<Vec<f64>>,
    rho_hist: &VecDeque<f64>,
) -> Vec<f64> {
    let mut q = g.to_vec();
    let k = s_hist.len();
    let mut alpha = vec![0.0; k];
    for i in (0..k).rev() {
        let (Some(s), Some(y), Some(&rho)) = (s_hist.get(i), y_hist.get(i), rho_hist.get(i)) else {
            continue;
        };
        let a = rho * dot(s, &q);
        if let Some(slot) = alpha.get_mut(i) {
            *slot = a;
        }
        for (qi, yi) in q.iter_mut().zip(y) {
            *qi -= a * yi;
        }
    }
    // Initial Hessian scaling γ = sᵀy / yᵀy from the most recent pair.
    let gamma = match (s_hist.back(), y_hist.back()) {
        (Some(s), Some(y)) => {
            let yy = dot(y, y);
            if yy > 0.0 { dot(s, y) / yy } else { 1.0 }
        }
        _ => 1.0,
    };
    for qi in &mut q {
        *qi *= gamma;
    }
    for i in 0..k {
        let (Some(s), Some(y), Some(&rho), Some(&ai)) =
            (s_hist.get(i), y_hist.get(i), rho_hist.get(i), alpha.get(i))
        else {
            continue;
        };
        let beta = rho * dot(y, &q);
        for (qi, si) in q.iter_mut().zip(s) {
            *qi += (ai - beta) * si;
        }
    }
    q.iter().map(|qi| -qi).collect()
}
