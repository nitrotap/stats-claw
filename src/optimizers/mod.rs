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
/// The fields together answer: *did it converge, how good is the
/// result, and how much work did it take.*
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
    if let Some(row) = m.get_mut(i)
        && let Some(slot) = row.get_mut(j)
    {
        *slot = value;
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

/// Kani formal-verification harnesses for the optimizer step primitives.
///
/// Compiled only under `cargo kani` (behind `#[cfg(kani)]`); invisible to normal
/// build/test/clippy. They prove that the arithmetic every optimizer step is built
/// from — the vector primitives [`norm`], [`dot`], [`matvec`], and [`step`] —
/// neither panics nor overflows for arbitrary *bounded finite* state.
///
/// ## Scope note (honest disclosure)
///
/// The optimizers take the learning rate, tolerance, and iteration budget as free
/// parameters and do **not** validate them — there is no `Result`-returning
/// parameter-validation surface to prove rejects bad input via `Err`. These proofs
/// therefore target the property that *is* present: the per-step vector arithmetic
/// is panic-/overflow-free over magnitude-bounded finite state.
///
/// A whole single [`gradient::gradient_descent`] iteration through a symbolic
/// objective was attempted but **dropped**: the objective's `grad` returns a
/// heap-allocated `Vec<f64>`, and modelling that allocation plus the update loop
/// blew CBMC past its memory budget (≈200k SAT variables, out-of-memory). The step
/// arithmetic it would have exercised is instead covered directly by
/// [`optimizers_step_finite`], whose `α·mul_add(dir, x)` is the *exact* shape of the
/// learning-rate update `xᵢ ← xᵢ − lr·gᵢ` (with `α = −lr`, `dir = g`); together with
/// [`optimizers_norm_finite_non_negative`] (the gradient-norm stopping test) this
/// covers every arithmetic operation a first-order step performs. The learning-rate
/// optimizers (`sgd`, `adam`, `rmsprop`, `adagrad`) and the line-search / second-
/// order methods (`newton`, `lbfgs`, `conjugate_gradient`) share this step shape;
/// their extra per-optimizer accumulator state is not individually proved here.
#[cfg(kani)]
mod verification {
    use super::{dot, matvec, norm, step};

    /// Upper bound on `|xᵢ|` for every symbolic coordinate.
    ///
    /// The task scopes the step proofs to `|x| ≤ 1e6`. At this bound every product
    /// `xᵢ·yⱼ ≤ 1e12` and every three-term sum `≤ 3e12` stays far below
    /// `f64::MAX ≈ 1.8e308`, so no intermediate overflows to `±∞` and the sign /
    /// finiteness arguments hold in `f64` rounding, not just exact arithmetic.
    /// A fully unbounded symbolic `f64` would overflow these sums to `±∞` — a
    /// genuine floating-point limitation, not a solver artifact — so the bound
    /// isolates the panic-/overflow-freedom property from extreme-magnitude
    /// arithmetic that no optimizer is expected to survive.
    const MAX_ABS: f64 = 1e6;

    /// Upper bound on `|α|` (a stand-in for `−lr`) in the [`step`] proof.
    ///
    /// Bounding the scale keeps the update `α·dirᵢ` at most `1e3·1e6 = 1e9`, so the
    /// stepped coordinate `xᵢ + α·dirᵢ` stays finite; an unbounded scale could push
    /// a finite direction past `f64::MAX`, again a real limitation rather than a
    /// spurious failure.
    const MAX_SCALE: f64 = 1e3;

    /// Fixed problem dimension for the vector-primitive proofs.
    ///
    /// Three coordinates exercise the full iterator fold (more than the trivial
    /// one- or two-element cases) while keeping the loop-free unrolling small.
    const DIM: usize = 3;

    /// Draws a symbolic `f64` constrained to be finite and bounded by [`MAX_ABS`].
    ///
    /// # Returns
    ///
    /// A finite `f64` with `|x| ≤ MAX_ABS`.
    fn any_bounded() -> f64 {
        let x: f64 = kani::any();
        kani::assume(x.is_finite());
        kani::assume(x.abs() <= MAX_ABS);
        x
    }

    /// Builds a length-[`DIM`] array of independent bounded-finite coordinates.
    ///
    /// # Returns
    ///
    /// An array `[f64; DIM]` with every entry drawn by [`any_bounded`].
    fn any_vec() -> [f64; DIM] {
        [any_bounded(), any_bounded(), any_bounded()]
    }

    /// Proves the Euclidean norm of a bounded-finite vector is panic-/overflow-free
    /// and yields a finite, non-negative result.
    #[kani::proof]
    fn optimizers_norm_finite_non_negative() {
        let v = any_vec();
        let n = norm(&v);
        assert!(n.is_finite(), "norm produced a non-finite value: {n}");
        assert!(n >= 0.0, "norm produced a negative value: {n}");
    }

    /// Proves the dot product of two bounded-finite vectors is panic-/overflow-free
    /// and finite (the inner arithmetic shared by every gradient step).
    #[kani::proof]
    fn optimizers_dot_finite() {
        let a = any_vec();
        let b = any_vec();
        let d = dot(&a, &b);
        assert!(d.is_finite(), "dot produced a non-finite value: {d}");
    }

    /// Proves the matrix–vector product used by the Newton step is
    /// panic-/overflow-free and finite for a bounded-finite matrix and vector.
    #[kani::proof]
    fn optimizers_matvec_finite() {
        let m: Vec<Vec<f64>> = vec![any_vec().to_vec(), any_vec().to_vec(), any_vec().to_vec()];
        let v = any_vec();
        let y = matvec(&m, &v);
        assert!(
            y.iter().all(|c| c.is_finite()),
            "matvec produced a non-finite component"
        );
        assert!(y.len() == DIM, "matvec changed the vector length");
    }

    /// Proves the `x ← x + α·dir` update is panic-/overflow-free and finite for a
    /// bounded scale and bounded-finite point and direction. With `α = −lr` and
    /// `dir = gradient` this is exactly the learning-rate update
    /// `xᵢ ← xᵢ − lr·gᵢ`, so it stands in for the dropped whole-step proof.
    #[kani::proof]
    fn optimizers_step_finite() {
        let x = any_vec();
        let dir = any_vec();
        let alpha = {
            let a: f64 = kani::any();
            kani::assume(a.is_finite());
            kani::assume(a.abs() <= MAX_SCALE);
            a
        };
        let next = step(&x, alpha, &dir);
        assert!(
            next.iter().all(|c| c.is_finite()),
            "step produced a non-finite coordinate"
        );
    }
}
