//! Standard test objectives for the optimizer suite.
//!
//! [`Quadratic`] is a strictly convex bowl with a known global minimizer, used
//! to assert convergence and `scipy.optimize` agreement on an easy problem.
//! [`Rosenbrock`] is the classic non-convex valley with its global minimum at
//! `(1, 1)`, used to stress the gradient-based, quasi-Newton, and derivative-free
//! methods.
//!
//! # Examples
//!
//! ```
//! use stats_claw::optimizers::objectives::Quadratic;
//! use stats_claw::optimizers::Objective;
//!
//! let q = Quadratic::new(vec![3.0, -2.0]);
//! // The minimum value is 0, attained at the center.
//! assert!(q.value(&[3.0, -2.0]).abs() < 1e-15, "f(c) = {}", q.value(&[3.0, -2.0]));
//! // The gradient is zero at the center.
//! use stats_claw::optimizers::norm;
//! assert!(norm(&q.grad(&[3.0, -2.0])) < 1e-15);
//! ```

use super::Objective;

/// Separable quadratic `f(x) = Σ (xᵢ − cᵢ)²`.
///
/// Strictly convex with a single global minimum of `0` at the center `c`. Its
/// gradient is `2(x − c)` and its Hessian is `2·I`.
#[derive(Debug, Clone)]
pub struct Quadratic {
    /// The minimizer `c`; `f` attains its minimum of `0` here.
    center: Vec<f64>,
}

impl Quadratic {
    /// Builds a quadratic with global minimum `0` at `center`.
    ///
    /// # Arguments
    ///
    /// * `center` — the minimizer `c`; its length is the problem dimension.
    ///
    /// # Returns
    ///
    /// A [`Quadratic`] whose [`Objective`] impl minimizes to `center`.
    #[must_use]
    pub const fn new(center: Vec<f64>) -> Self {
        Self { center }
    }
}

impl Objective for Quadratic {
    fn value(&self, x: &[f64]) -> f64 {
        x.iter()
            .zip(&self.center)
            .map(|(a, c)| (a - c) * (a - c))
            .sum()
    }

    fn grad(&self, x: &[f64]) -> Vec<f64> {
        x.iter()
            .zip(&self.center)
            .map(|(a, c)| 2.0 * (a - c))
            .collect()
    }

    fn hessian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let n = x.len();
        (0..n)
            .map(|i| (0..n).map(|j| if i == j { 2.0 } else { 0.0 }).collect())
            .collect()
    }
}

/// The two-dimensional Rosenbrock function
/// `f(x) = (1 − x₀)² + 100·(x₁ − x₀²)²`.
///
/// Non-convex, with a narrow curved valley and a global minimum of `0` at
/// `(1, 1)`. A demanding benchmark for first-order methods.
#[derive(Debug, Clone, Copy)]
pub struct Rosenbrock;

impl Objective for Rosenbrock {
    fn value(&self, x: &[f64]) -> f64 {
        let x0 = *x.first().unwrap_or(&0.0);
        let x1 = *x.get(1).unwrap_or(&0.0);
        let a = 1.0 - x0;
        let b = x1 - x0 * x0;
        b.mul_add(100.0 * b, a * a)
    }

    fn grad(&self, x: &[f64]) -> Vec<f64> {
        let x0 = *x.first().unwrap_or(&0.0);
        let x1 = *x.get(1).unwrap_or(&0.0);
        let d0 = (-400.0 * x0).mul_add(x1 - x0 * x0, -2.0 * (1.0 - x0));
        let d1 = 200.0 * (x1 - x0 * x0);
        vec![d0, d1]
    }

    fn hessian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let x0 = *x.first().unwrap_or(&0.0);
        let x1 = *x.get(1).unwrap_or(&0.0);
        let h00 = (-400.0_f64).mul_add(x1, (1200.0 * x0).mul_add(x0, 2.0));
        let h01 = -400.0 * x0;
        vec![vec![h00, h01], vec![h01, 200.0]]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadratic_minimum_is_zero_at_center() {
        let q = Quadratic::new(vec![3.0, -2.0]);
        assert!(
            q.value(&[3.0, -2.0]).abs() < 1e-15,
            "f(c) = {}",
            q.value(&[3.0, -2.0])
        );
        let g = q.grad(&[3.0, -2.0]);
        assert!(super::super::norm(&g) < 1e-15, "grad at center nonzero");
    }

    #[test]
    fn rosenbrock_minimum_is_zero_at_one_one() {
        let r = Rosenbrock;
        assert!(
            r.value(&[1.0, 1.0]).abs() < 1e-15,
            "f(1,1) = {}",
            r.value(&[1.0, 1.0])
        );
        let g = r.grad(&[1.0, 1.0]);
        assert!(super::super::norm(&g) < 1e-12, "grad at (1,1) nonzero");
    }

    #[test]
    fn rosenbrock_analytic_hessian_matches_finite_difference() {
        // Override agrees with the trait default (finite-difference) within 1e-4.
        let r = Rosenbrock;
        let x = [0.5, 0.3];
        let analytic = r.hessian(&x);
        let fd = Objective::hessian(&FdRosen, &x);
        for (ra, rf) in analytic.iter().zip(&fd) {
            for (a, f) in ra.iter().zip(rf) {
                assert!((a - f).abs() < 1e-4, "hessian mismatch: {a} vs {f}");
            }
        }
    }

    /// Rosenbrock without the Hessian override, to exercise the trait default.
    struct FdRosen;
    impl Objective for FdRosen {
        fn value(&self, x: &[f64]) -> f64 {
            Rosenbrock.value(x)
        }
        fn grad(&self, x: &[f64]) -> Vec<f64> {
            Rosenbrock.grad(x)
        }
    }
}
