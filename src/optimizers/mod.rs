//! Numerical optimizers minimizing an [`Objective`].
//!
//! Every optimizer in this module reduces a scalar objective `f: ℝⁿ → ℝ` and
//! returns an [`OptimizeResult`] reporting the located point, its objective
//! value, the iteration count, and a [`ConvergenceStatus`]. The families are
//! grouped into subfolders: [`gradient`] (first-order learning-rate methods and
//! conjugate gradient), [`second_order`] (Newton and L-BFGS), and [`stochastic`]
//! (simulated annealing and genetic / differential evolution). The shared test
//! objectives live in [`objectives`].
//!
//! ## `scipy.optimize` mapping
//!
//! Each optimizer is paired with the `scipy.optimize` method it is cross-checked
//! against; methods with no faithful counterpart are documented as excluded so
//! the comparison coverage is auditable.
//!
//! | stats-claw              | `scipy.optimize`                  | agreement |
//! |-----------------------|-----------------------------------|-----------|
//! | `gradient_descent`    | none (vanilla GD)                 | excluded  |
//! | `sgd`                 | none                              | excluded  |
//! | `adam`                | none                              | excluded  |
//! | `rmsprop`             | none                              | excluded  |
//! | `adagrad`             | none                              | excluded  |
//! | `conjugate_gradient`  | `minimize(method="CG")`           | compared  |
//! | `newton`              | `minimize(method="Newton-CG")`    | compared  |
//! | `lbfgs`               | `minimize(method="L-BFGS-B")`     | compared  |
//! | `simulated_annealing` | `dual_annealing`                  | optimum   |
//! | `genetic`             | `differential_evolution`          | optimum   |
//!
//! Deterministic optimizers (exempt from the seed-variation check):
//! `gradient_descent`, `adam`, `rmsprop`, `adagrad`, `conjugate_gradient`,
//! `newton`, `lbfgs`. Stochastic optimizers (seed-variation required): `sgd`,
//! `simulated_annealing`, `genetic`.

pub mod gradient;
pub mod objectives;
pub mod second_order;
pub mod stochastic;

/// Outcome of an optimization run: whether the stopping criterion was satisfied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceStatus {
    /// The convergence criterion (e.g. gradient norm below tolerance) was met.
    Converged,
    /// The iteration budget was exhausted before the criterion was met.
    MaxIterReached,
}

/// The result of minimizing an [`Objective`].
///
/// The fields together answer the three questions a caller has after a run:
/// *did it converge, how good is the result, and how much work did it take.*
#[derive(Debug, Clone)]
pub struct OptimizeResult {
    /// The located minimizer (the point at which the run stopped).
    pub x: Vec<f64>,
    /// The objective value `f(x)` at the located point.
    pub fx: f64,
    /// The number of iterations actually performed (always `≥ 0`).
    pub iterations: usize,
    /// Whether the run converged or exhausted its iteration budget.
    pub status: ConvergenceStatus,
}

/// A differentiable scalar objective `f: ℝⁿ → ℝ` to be minimized.
///
/// Implementors must supply [`value`](Objective::value) and
/// [`grad`](Objective::grad). The Hessian defaults to a central finite-difference
/// approximation built from `grad`, so second-order optimizers work for any
/// objective without an analytic Hessian; objectives that have one may override
/// [`hessian`](Objective::hessian) for accuracy.
pub trait Objective {
    /// Evaluates the objective at `x`.
    ///
    /// # Arguments
    ///
    /// * `x` — the point at which to evaluate; any finite coordinates.
    ///
    /// # Returns
    ///
    /// The scalar objective value `f(x)`.
    fn value(&self, x: &[f64]) -> f64;

    /// Evaluates the gradient `∇f(x)`.
    ///
    /// # Arguments
    ///
    /// * `x` — the point at which to evaluate the gradient.
    ///
    /// # Returns
    ///
    /// The gradient vector, the same length as `x`.
    fn grad(&self, x: &[f64]) -> Vec<f64>;

    /// Approximates the Hessian `∇²f(x)` by central differences of the gradient.
    ///
    /// The default uses a step of `√ε ≈ 1.49e-8` per coordinate and symmetrizes
    /// the result so it is exactly symmetric (rounding can otherwise break
    /// symmetry). Objectives with an analytic Hessian should override this.
    ///
    /// # Arguments
    ///
    /// * `x` — the point at which to approximate the Hessian.
    ///
    /// # Returns
    ///
    /// The `n × n` Hessian in row-major order (`n = x.len()`).
    fn hessian(&self, x: &[f64]) -> Vec<Vec<f64>> {
        let n = x.len();
        let h = f64::EPSILON.sqrt();
        let mut hess = vec![vec![0.0; n]; n];
        let mut xp = x.to_vec();
        for j in 0..n {
            let xj = *xp.get(j).unwrap_or(&0.0);
            set(&mut xp, j, xj + h);
            let gp = self.grad(&xp);
            set(&mut xp, j, xj - h);
            let gm = self.grad(&xp);
            set(&mut xp, j, xj);
            for i in 0..n {
                let dgi = gp.get(i).unwrap_or(&0.0) - gm.get(i).unwrap_or(&0.0);
                set_mat(&mut hess, i, j, dgi / (2.0 * h));
            }
        }
        symmetrize(&mut hess);
        hess
    }
}

/// Writes `value` into `v[i]`, ignoring an out-of-range index (cannot occur for
/// the in-bounds indices used here, but keeps the code clear of
/// `indexing_slicing`).
fn set(v: &mut [f64], i: usize, value: f64) {
    if let Some(slot) = v.get_mut(i) {
        *slot = value;
    }
}

/// Writes `value` into `m[i][j]`, ignoring an out-of-range index.
fn set_mat(m: &mut [Vec<f64>], i: usize, j: usize, value: f64) {
    if let Some(row) = m.get_mut(i) {
        if let Some(slot) = row.get_mut(j) {
            *slot = value;
        }
    }
}

/// Averages a square matrix with its transpose in place so it is symmetric.
fn symmetrize(m: &mut [Vec<f64>]) {
    let n = m.len();
    for i in 0..n {
        for j in (i + 1)..n {
            let a = mat(m, i, j);
            let b = mat(m, j, i);
            let avg = 0.5 * (a + b);
            set_mat(m, i, j, avg);
            set_mat(m, j, i, avg);
        }
    }
}

/// Reads `m[i][j]`, returning `0.0` for an out-of-range index.
fn mat(m: &[Vec<f64>], i: usize, j: usize) -> f64 {
    *m.get(i).and_then(|row| row.get(j)).unwrap_or(&0.0)
}

/// Euclidean (L2) norm of a vector.
///
/// # Arguments
///
/// * `v` — the vector whose norm is taken.
///
/// # Returns
///
/// `√Σ vᵢ²`, used as the gradient-norm stopping criterion across optimizers.
#[must_use]
pub fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

/// Dot product of two equal-length vectors (extra elements of the longer one are
/// ignored, which never happens for the matched-length inputs used internally).
///
/// # Arguments
///
/// * `a`, `b` — the vectors to multiply elementwise and sum.
///
/// # Returns
///
/// `Σ aᵢ·bᵢ`.
#[must_use]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Multiplies a square matrix (row-major `Vec<Vec<f64>>`) by a vector.
///
/// # Arguments
///
/// * `m` — an `n × n` matrix.
/// * `v` — an `n`-vector.
///
/// # Returns
///
/// The product `m·v` as an `n`-vector.
#[must_use]
pub fn matvec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| dot(row, v)).collect()
}

/// Backtracking line search satisfying the Armijo sufficient-decrease condition.
///
/// Starting from step `1.0`, halves the step until
/// `f(x + α·d) ≤ f(x) + c·α·gᵀd` holds, used by the line-search optimizers
/// (conjugate gradient, Newton, L-BFGS) to pick a stable step along `d`.
///
/// # Arguments
///
/// * `obj` — the objective being minimized.
/// * `x` — the current point.
/// * `dir` — the search direction (should be a descent direction).
/// * `grad` — the gradient at `x` (so `gᵀd` need not be recomputed).
///
/// # Returns
///
/// The accepted step length `α` (at least `MIN_STEP`, so progress is bounded).
pub(crate) fn line_search(obj: &impl Objective, x: &[f64], dir: &[f64], grad: &[f64]) -> f64 {
    const C: f64 = 1e-4;
    const SHRINK: f64 = 0.5;
    /// Maximum halvings (`0.5^100 ≈ 1e-30`) before accepting the smallest step.
    const MAX_HALVINGS: usize = 100;
    let f0 = obj.value(x);
    let slope = dot(grad, dir);
    let mut alpha = 1.0_f64;
    for _ in 0..MAX_HALVINGS {
        let trial: Vec<f64> = x
            .iter()
            .zip(dir)
            .map(|(xi, di)| alpha.mul_add(*di, *xi))
            .collect();
        if obj.value(&trial) <= (C * alpha).mul_add(slope, f0) {
            return alpha;
        }
        alpha *= SHRINK;
    }
    alpha
}

/// Steps `x` to `x + α·dir`, returning the new point.
///
/// # Arguments
///
/// * `x` — the current point.
/// * `alpha` — the step length.
/// * `dir` — the step direction.
///
/// # Returns
///
/// The point `x + α·dir`.
pub(crate) fn step(x: &[f64], alpha: f64, dir: &[f64]) -> Vec<f64> {
    x.iter()
        .zip(dir)
        .map(|(xi, di)| alpha.mul_add(*di, *xi))
        .collect()
}
