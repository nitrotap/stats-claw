//! Dense linear-algebra kernel for the regression normal equations.
//!
//! The estimators in the parent module reduce to solving a small symmetric
//! positive-(semi)definite system `A β = b`. This module provides that solver —
//! Gaussian elimination with partial pivoting — plus the `as`-free count widening
//! the fit relies on. Keeping it separate keeps each file inside the project's
//! 500-line `style.rs` cap and isolates the numerical kernel from the estimator
//! API.

use super::RegressionError;

/// Pivot magnitude below which the normal-equations matrix is treated as singular.
const PIVOT_EPS: f64 = 1e-12;

/// Solves the linear system `A β = b` by Gaussian elimination with partial
/// pivoting, consuming the (square) coefficient matrix `a` and right-hand side `b`.
///
/// Partial pivoting (swapping in the row with the largest sub-column magnitude at
/// each step) keeps the elimination numerically stable for the symmetric
/// normal-equations matrices the regression fit produces.
///
/// # Arguments
///
/// * `a` — the `n × n` coefficient matrix, as `n` row vectors of length `n`.
/// * `b` — the length-`n` right-hand side.
///
/// # Returns
///
/// The solution vector `β` of length `n`.
///
/// # Errors
///
/// Returns [`RegressionError::Singular`] if a pivot is zero to working precision
/// (`< PIVOT_EPS`), i.e. the system has no unique solution.
pub(super) fn solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Result<Vec<f64>, RegressionError> {
    let n = a.len();
    for col in 0..n {
        // Partial pivot: pick the row (at or below `col`) with the largest |pivot|.
        let mut pivot_row = col;
        let mut best = pivot_magnitude(&a, col, col);
        for r in (col + 1)..n {
            let mag = pivot_magnitude(&a, r, col);
            if mag > best {
                best = mag;
                pivot_row = r;
            }
        }
        if best < PIVOT_EPS {
            return Err(RegressionError::Singular);
        }
        a.swap(col, pivot_row);
        b.swap(col, pivot_row);

        let pivot = a
            .get(col)
            .and_then(|row| row.get(col))
            .copied()
            .unwrap_or(0.0);
        // The pivot row is fixed for this column; snapshot the values the lower
        // rows subtract (a clone of the pivot row and its rhs) so the inner loop
        // can mutate row `r` without aliasing the pivot row's borrow.
        let pivot_a = a.get(col).cloned().unwrap_or_default();
        let pivot_b = at(&b, col);
        for r in (col + 1)..n {
            let factor = a
                .get(r)
                .and_then(|row| row.get(col))
                .copied()
                .unwrap_or(0.0)
                / pivot;
            if let Some(target_row) = a.get_mut(r) {
                for k in col..n {
                    let pivot_val = pivot_a.get(k).copied().unwrap_or(0.0);
                    if let Some(target) = target_row.get_mut(k) {
                        *target -= factor * pivot_val;
                    }
                }
            }
            if let Some(bv) = b.get_mut(r) {
                *bv -= factor * pivot_b;
            }
        }
    }
    back_substitute(&a, &b)
}

/// Back-substitutes an upper-triangular system `U β = b` into the solution `β`.
///
/// # Arguments
///
/// * `a` — the upper-triangular coefficient matrix produced by elimination.
/// * `b` — the transformed right-hand side.
///
/// # Returns
///
/// The solution vector `β`.
///
/// # Errors
///
/// Returns [`RegressionError::Singular`] if a diagonal pivot is zero to working
/// precision.
fn back_substitute(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, RegressionError> {
    let n = a.len();
    let mut beta = vec![0.0_f64; n];
    for row in (0..n).rev() {
        let mut acc = at(b, row);
        for k in (row + 1)..n {
            let a_rk = a.get(row).and_then(|r| r.get(k)).copied().unwrap_or(0.0);
            let beta_k = beta.get(k).copied().unwrap_or(0.0);
            acc -= a_rk * beta_k;
        }
        let pivot = a.get(row).and_then(|r| r.get(row)).copied().unwrap_or(0.0);
        if pivot.abs() < PIVOT_EPS {
            return Err(RegressionError::Singular);
        }
        if let Some(beta_row) = beta.get_mut(row) {
            *beta_row = acc / pivot;
        }
    }
    Ok(beta)
}

/// Pivot magnitude `|a[r][c]|`, or `0.0` for an out-of-range index.
fn pivot_magnitude(a: &[Vec<f64>], r: usize, c: usize) -> f64 {
    a.get(r)
        .and_then(|row| row.get(c))
        .copied()
        .unwrap_or(0.0)
        .abs()
}

/// Reads `v[i]`, or `0.0` for an out-of-range index.
fn at(v: &[f64], i: usize) -> f64 {
    v.get(i).copied().unwrap_or(0.0)
}

/// Widens a `usize` count to `f64` without an `as` cast.
///
/// Sample and column counts here are far below `2^53`, so splitting into 32-bit
/// halves and recombining reproduces the value exactly while satisfying the
/// `style.rs` no-`as` guard.
///
/// # Arguments
///
/// * `n` — the count to widen.
///
/// # Returns
///
/// `n` as an `f64` (exact for the sizes this crate handles).
pub(super) fn count_to_f64(n: usize) -> f64 {
    let wide = u64::try_from(n).unwrap_or(u64::MAX);
    let hi = u32::try_from(wide >> 32).unwrap_or(0);
    let lo = u32::try_from(wide & 0xFFFF_FFFF).unwrap_or(0);
    f64::from(hi).mul_add(4_294_967_296.0, f64::from(lo))
}

/// Returns the arithmetic mean of `values`, or `0.0` for an empty slice.
///
/// # Arguments
///
/// * `values` — the numbers to average.
///
/// # Returns
///
/// The mean `Σ values / |values|`, or `0.0` when `values` is empty.
pub(super) fn mean(values: &[f64]) -> f64 {
    let n = count_to_f64(values.len());
    if n > 0.0 {
        values.iter().sum::<f64>() / n
    } else {
        0.0
    }
}

/// Returns the per-column mean of the design matrix `x` (length `n_cols`).
///
/// # Arguments
///
/// * `x` — the design matrix, one inner slice per observation.
/// * `n_cols` — the predictor count (width of every row).
///
/// # Returns
///
/// A length-`n_cols` vector of column means, all `0.0` for an empty `x`.
pub(super) fn column_means(x: &[Vec<f64>], n_cols: usize) -> Vec<f64> {
    let n = count_to_f64(x.len());
    let mut sums = vec![0.0_f64; n_cols];
    for row in x {
        for (s, &v) in sums.iter_mut().zip(row) {
            *s += v;
        }
    }
    if n > 0.0 {
        for s in &mut sums {
            *s /= n;
        }
    }
    sums
}

/// Returns the centred value of column `j` in `row` (`x_{ij} − x̄_j`).
///
/// # Arguments
///
/// * `row` — one observation's predictor values.
/// * `col_means` — the per-column means to subtract.
/// * `j` — the column index to centre.
///
/// # Returns
///
/// `row[j] − col_means[j]`, or `0.0` if either index is out of range (which the
/// caller's shape checks already preclude).
pub(super) fn centered(row: &[f64], col_means: &[f64], j: usize) -> f64 {
    let v = row.get(j).copied().unwrap_or(0.0);
    let m = col_means.get(j).copied().unwrap_or(0.0);
    v - m
}

/// Kani proof harness for the dense Gaussian-elimination solver.
///
/// Compiled only under `cargo kani` (behind `#[cfg(kani)]`); invisible to normal
/// build/test/clippy.
#[cfg(kani)]
mod verification {
    use super::{RegressionError, solve};

    /// Magnitude bound on every matrix and right-hand-side entry: it keeps the
    /// elimination arithmetic finite so a run cannot diverge into `∞`/`NaN` control
    /// flow, isolating the panic-/overflow- and index-safety argument.
    const MAX_ABS: f64 = 1e6;

    /// Draws a symbolic entry constrained to be finite and bounded.
    fn any_entry() -> f64 {
        let x: f64 = kani::any();
        kani::assume(x.is_finite());
        kani::assume(x.abs() <= MAX_ABS);
        x
    }

    /// Proves [`solve`] never panics, overflows, or indexes out of bounds for a fully
    /// symbolic bounded `2×2` system, and returns either a length-2 solution or
    /// [`RegressionError::Singular`].
    ///
    /// Partial pivoting plus the `PIVOT_EPS` guard mean a (near-)singular system is
    /// reported rather than dividing by zero; every array access already goes through
    /// `get`/`get_mut`, and this harness discharges that the derived indices and the
    /// pivot division stay well-defined across the whole bounded coefficient space.
    /// The `#[kani::unwind(4)]` unrolls the fixed `0..2` elimination and
    /// back-substitution loops (`n = 2`, plus one turn to observe each exit).
    #[kani::proof]
    #[kani::unwind(4)]
    fn reg_solve_2x2_total() {
        let a = vec![
            vec![any_entry(), any_entry()],
            vec![any_entry(), any_entry()],
        ];
        let b = vec![any_entry(), any_entry()];
        match solve(a, b) {
            Ok(beta) => assert!(beta.len() == 2, "solution had the wrong length"),
            Err(RegressionError::Singular) => {}
            Err(_) => assert!(false, "solve returned an unexpected error variant"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The solver reproduces the known solution of a small `2 × 2` system.
    #[test]
    fn solves_two_by_two_system() -> Result<(), RegressionError> {
        // [2 1][x]   [5]      x = 1, y = 3  (2·1 + 1·3 = 5, 1·1 + 3·3 = 10)
        // [1 3][y] = [10]
        let a = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
        let b = vec![5.0, 10.0];
        let beta = solve(a, b)?;
        let x = beta.first().copied().unwrap_or(f64::NAN);
        let y = beta.get(1).copied().unwrap_or(f64::NAN);
        assert!((x - 1.0).abs() < 1e-12, "x was {x}");
        assert!((y - 3.0).abs() < 1e-12, "y was {y}");
        Ok(())
    }

    /// A singular matrix is reported, not silently mis-solved.
    #[test]
    fn singular_matrix_is_reported() {
        let a = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
        let b = vec![1.0, 2.0];
        assert_eq!(solve(a, b), Err(RegressionError::Singular));
    }

    /// The `as`-free widening reproduces a representative count exactly.
    #[test]
    fn count_widens_exactly() {
        assert!((count_to_f64(4096) - 4096.0).abs() < 1e-12);
    }
}
