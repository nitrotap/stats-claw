//! Decomposition and embedding algorithms and their shared linear algebra.
//!
//! Each concrete algorithm lives in its own submodule (`pca`, `factor_analysis`,
//! `ica`, `tsne`, `umap`, `lle`) and is re-exported here so callers name
//! `decomposition::<algo>` without the submodule path. This module owns the dense
//! linear-algebra primitives the family shares — mean-centering, the covariance
//! matrix, a symmetric-eigenvalue Jacobi solver, and a reconstruction-error
//! measure — operating on row-major `f64` buffers.
//!
//! Components and embedding axes are defined only up to sign (and, for ICA and
//! spectral methods, up to ordering), so the equivalence suite compares the
//! sign/order-invariant quantities — explained variance, reconstruction error, and
//! trustworthiness — rather than raw component values.

mod factor_analysis;
mod ica;
mod lle;
mod pca;
mod tsne;
mod umap;

pub use factor_analysis::{FactorResult, factor_analysis};
pub use ica::{IcaResult, fast_ica};
pub use lle::lle;
pub use pca::{PcaResult, pca};
pub use tsne::tsne;
pub use umap::umap;

/// Maximum cyclic-Jacobi sweeps before the symmetric eigensolver stops iterating.
const JACOBI_SWEEPS: usize = 100;
/// Off-diagonal magnitude below which a matrix is treated as already diagonal.
const JACOBI_TOL: f64 = 1e-14;

/// Widens a `usize` count to `f64` without an `as` cast.
///
/// Sample and dimension counts here are far below `2^53`, so splitting into 32-bit
/// halves and recombining reproduces the value exactly while satisfying the
/// `style.rs` no-`as` guard.
///
/// # Arguments
///
/// * `n` — the count to widen.
///
/// # Returns
///
/// `n` as an exact `f64`.
#[must_use]
pub fn count_to_f64(n: usize) -> f64 {
    let wide = u64::try_from(n).unwrap_or(u64::MAX);
    let hi = u32::try_from(wide >> 32).unwrap_or(0);
    let lo = u32::try_from(wide & 0xFFFF_FFFF).unwrap_or(0);
    f64::from(hi).mul_add(4_294_967_296.0, f64::from(lo))
}

/// Computes the per-feature mean (centroid) of a row-major data matrix.
///
/// # Arguments
///
/// * `data` — observations; each inner slice is one row of equal dimension `dim`.
/// * `dim` — the number of features per row.
///
/// # Returns
///
/// A length-`dim` vector of column means (a zero vector when `data` is empty).
#[must_use]
pub fn column_means(data: &[Vec<f64>], dim: usize) -> Vec<f64> {
    let mut sum = vec![0.0_f64; dim];
    for row in data {
        for (s, &v) in sum.iter_mut().zip(row) {
            *s += v;
        }
    }
    let n = count_to_f64(data.len());
    if n > 0.0 {
        for s in &mut sum {
            *s /= n;
        }
    }
    sum
}

/// Mean-centers `data`, returning the centered matrix and the column means.
///
/// # Arguments
///
/// * `data` — observations; each inner slice is one row of dimension `dim`.
/// * `dim` — the number of features per row.
///
/// # Returns
///
/// `(centered, means)` where `centered[i][j] = data[i][j] − means[j]`.
#[must_use]
pub fn mean_center(data: &[Vec<f64>], dim: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let means = column_means(data, dim);
    let centered = data
        .iter()
        .map(|row| {
            row.iter()
                .zip(&means)
                .map(|(&v, &m)| v - m)
                .collect::<Vec<f64>>()
        })
        .collect();
    (centered, means)
}

/// Computes the `dim×dim` sample covariance matrix of `centered` (already
/// mean-centered) using the unbiased `n − 1` denominator, matching `scikit-learn`.
///
/// # Arguments
///
/// * `centered` — mean-centered observations, one row per sample.
/// * `dim` — the number of features.
///
/// # Returns
///
/// The covariance matrix as a row-major `dim×dim` buffer. With fewer than two
/// samples the denominator collapses and a zero matrix is returned.
#[must_use]
pub fn covariance(centered: &[Vec<f64>], dim: usize) -> Vec<f64> {
    let mut cov = vec![0.0_f64; dim * dim];
    for row in centered {
        for i in 0..dim {
            let ri = row.get(i).copied().unwrap_or(0.0);
            for j in 0..dim {
                let rj = row.get(j).copied().unwrap_or(0.0);
                if let Some(slot) = cov.get_mut(i * dim + j) {
                    *slot = ri.mul_add(rj, *slot);
                }
            }
        }
    }
    let denom = count_to_f64(centered.len()) - 1.0;
    if denom > 0.0 {
        for c in &mut cov {
            *c /= denom;
        }
    }
    cov
}

/// Reads entry `(i, j)` from a row-major `n×n` buffer (`0.0` if out of range).
#[must_use]
pub fn at(matrix: &[f64], n: usize, i: usize, j: usize) -> f64 {
    matrix.get(i * n + j).copied().unwrap_or(0.0)
}

/// Writes entry `(i, j)` into a row-major `n×n` buffer (no-op if out of range).
pub fn put(matrix: &mut [f64], n: usize, i: usize, j: usize, value: f64) {
    if let Some(slot) = matrix.get_mut(i * n + j) {
        *slot = value;
    }
}

/// Builds the `n×n` identity matrix in a row-major buffer.
#[must_use]
pub fn identity(n: usize) -> Vec<f64> {
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        put(&mut v, n, i, i, 1.0);
    }
    v
}

/// Computes the eigenvalues and eigenvectors of a symmetric `n×n` matrix by the
/// cyclic Jacobi rotation method.
///
/// # Arguments
///
/// * `matrix` — a symmetric matrix in a row-major `n×n` buffer.
/// * `n` — the matrix dimension.
///
/// # Returns
///
/// `(eigenvalues, eigenvectors)` where eigenvector `col` occupies column `col` of
/// the returned row-major `n×n` buffer and pairs with `eigenvalues[col]`.
#[must_use]
pub fn jacobi_eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = matrix.to_vec();
    let mut v = identity(n);
    for _ in 0..JACOBI_SWEEPS {
        let mut off = 0.0_f64;
        for p in 0..n {
            for q in (p + 1)..n {
                off += at(&a, n, p, q).abs();
            }
        }
        if off < JACOBI_TOL {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                rotate(&mut a, &mut v, n, p, q);
            }
        }
    }
    let values = (0..n).map(|i| at(&a, n, i, i)).collect();
    (values, v)
}

/// Applies one Jacobi rotation zeroing the `(p, q)` off-diagonal entry of `a` and
/// accumulating it into the eigenvector matrix `v`.
///
/// The single-character names (`c`, `s`, `t`, `p`, `q`) are the canonical
/// Jacobi-rotation notation from Numerical Recipes; renaming them would obscure the
/// standard formulae rather than clarify them.
#[allow(clippy::many_single_char_names)]
fn rotate(a: &mut [f64], v: &mut [f64], n: usize, p: usize, q: usize) {
    let apq = at(a, n, p, q);
    if apq.abs() < JACOBI_TOL {
        return;
    }
    let app = at(a, n, p, p);
    let aqq = at(a, n, q, q);
    let theta = (aqq - app) / (2.0 * apq);
    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
    let c = 1.0 / (t * t + 1.0).sqrt();
    let s = t * c;
    for i in 0..n {
        let aip = at(a, n, i, p);
        let aiq = at(a, n, i, q);
        put(a, n, i, p, c.mul_add(aip, -(s * aiq)));
        put(a, n, i, q, s.mul_add(aip, c * aiq));
    }
    for i in 0..n {
        let api = at(a, n, p, i);
        let aqi = at(a, n, q, i);
        put(a, n, p, i, c.mul_add(api, -(s * aqi)));
        put(a, n, q, i, s.mul_add(api, c * aqi));
    }
    for i in 0..n {
        let vip = at(v, n, i, p);
        let viq = at(v, n, i, q);
        put(v, n, i, p, c.mul_add(vip, -(s * viq)));
        put(v, n, i, q, s.mul_add(vip, c * viq));
    }
}

/// Returns the eigenvalue indices in descending order of eigenvalue.
///
/// # Arguments
///
/// * `values` — eigenvalues to rank.
///
/// # Returns
///
/// Index permutation `order` with `values[order[0]]` the largest.
#[must_use]
pub fn descending_order(values: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| {
        let va = values.get(a).copied().unwrap_or(f64::NEG_INFINITY);
        let vb = values.get(b).copied().unwrap_or(f64::NEG_INFINITY);
        vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
    });
    order
}

/// Inverts a small symmetric positive-definite `n×n` matrix via its eigen-
/// decomposition: `A⁻¹ = V diag(1/λ) Vᵀ`.
///
/// Used for the tiny `k×k` posterior-covariance solve in factor analysis. Near-zero
/// eigenvalues are floored so the inverse stays finite for a degenerate input.
///
/// # Arguments
///
/// * `matrix` — a symmetric matrix in a row-major `n×n` buffer.
/// * `n` — the matrix dimension.
///
/// # Returns
///
/// The inverse as a row-major `n×n` buffer.
#[must_use]
pub fn symmetric_inverse(matrix: &[f64], n: usize) -> Vec<f64> {
    let (values, vectors) = jacobi_eigen(matrix, n);
    let mut inv = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0_f64;
            for (k, &lambda) in values.iter().enumerate().take(n) {
                let safe = if lambda.abs() < 1e-300 {
                    1e-300
                } else {
                    lambda
                };
                acc += at(&vectors, n, i, k) * at(&vectors, n, j, k) / safe;
            }
            put(&mut inv, n, i, j, acc);
        }
    }
    inv
}

/// Computes the mean squared reconstruction error between two equal-shaped
/// matrices.
///
/// # Arguments
///
/// * `original` — the input rows.
/// * `reconstructed` — the rows recovered from a low-dimensional representation;
///   must match `original` in shape.
///
/// # Returns
///
/// `mean over all entries of (original − reconstructed)²` — `0.0` for empty input.
#[must_use]
pub fn reconstruction_error(original: &[Vec<f64>], reconstructed: &[Vec<f64>]) -> f64 {
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for (orig_row, rec_row) in original.iter().zip(reconstructed) {
        for (&o, &r) in orig_row.iter().zip(rec_row) {
            let d = o - r;
            sum = d.mul_add(d, sum);
            count += 1;
        }
    }
    if count == 0 {
        return 0.0;
    }
    sum / count_to_f64(count)
}

/// Kani proof harness for the decomposition covariance kernel.
///
/// Compiled only under `cargo kani` (behind `#[cfg(kani)]`); invisible to normal
/// build/test/clippy.
#[cfg(kani)]
mod verification {
    use super::covariance;

    /// Magnitude bound on each centered entry: it keeps the squared products finite
    /// so the accumulation cannot diverge into `∞`/`NaN`, isolating the panic-,
    /// overflow-, and index-safety argument for the row-major `i·dim + j` addressing.
    const MAX_ABS: f64 = 1e3;

    /// Proves [`covariance`] never panics, overflows, or indexes out of bounds and
    /// produces a finite `dim×dim` buffer for a symbolic two-sample, two-feature
    /// centered matrix with bounded finite entries.
    ///
    /// This exercises the row-major `i·dim + j` write index against a `dim·dim`
    /// buffer (proving every `cov.get_mut` addresses a real slot) and the `n − 1`
    /// denominator guard, over the whole bounded input box. The `#[kani::unwind(5)]`
    /// unrolls the fixed `0..dim` (`dim = 2`) inner loops, the two-row outer loop, and
    /// the harness's own four-element bounding loop.
    #[kani::proof]
    #[kani::unwind(5)]
    fn decomp_covariance_total() {
        let a: f64 = kani::any();
        let b: f64 = kani::any();
        let c: f64 = kani::any();
        let d: f64 = kani::any();
        for v in [a, b, c, d] {
            kani::assume(v.is_finite());
            kani::assume(v.abs() <= MAX_ABS);
        }
        let centered = vec![vec![a, b], vec![c, d]];
        let cov = covariance(&centered, 2);
        assert!(cov.len() == 4, "covariance buffer was not dim*dim");
        assert!(
            cov.iter().all(|c| c.is_finite()),
            "covariance produced a non-finite entry"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_center_zeroes_the_column_means() {
        let data = vec![vec![1.0, 10.0], vec![3.0, 20.0]];
        let (centered, means) = mean_center(&data, 2);
        assert!((means.first().copied().unwrap_or(0.0) - 2.0).abs() < 1e-12);
        assert!((means.get(1).copied().unwrap_or(0.0) - 15.0).abs() < 1e-12);
        let first = centered
            .first()
            .and_then(|r| r.first())
            .copied()
            .unwrap_or(0.0);
        assert!((first + 1.0).abs() < 1e-12, "centered[0][0] was {first}");
    }

    #[test]
    fn jacobi_recovers_diagonal_eigenvalues() {
        // diag(3, 1): eigenvalues are the diagonal entries.
        let m = vec![3.0, 0.0, 0.0, 1.0];
        let (values, _) = jacobi_eigen(&m, 2);
        assert!(
            values.iter().any(|v| (v - 3.0).abs() < 1e-9),
            "missing eigenvalue 3: {values:?}"
        );
        assert!(
            values.iter().any(|v| (v - 1.0).abs() < 1e-9),
            "missing eigenvalue 1: {values:?}"
        );
    }

    #[test]
    fn covariance_of_centered_two_by_two() {
        // Points (−1,0),(1,0): variance in x is (1+1)/(2−1)=2, none in y.
        let centered = vec![vec![-1.0, 0.0], vec![1.0, 0.0]];
        let cov = covariance(&centered, 2);
        assert!(
            (at(&cov, 2, 0, 0) - 2.0).abs() < 1e-12,
            "var_x = {}",
            at(&cov, 2, 0, 0)
        );
        assert!(
            at(&cov, 2, 1, 1).abs() < 1e-12,
            "var_y = {}",
            at(&cov, 2, 1, 1)
        );
    }

    #[test]
    fn reconstruction_error_is_zero_for_identical() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        assert!(reconstruction_error(&a, &a).abs() < 1e-12);
    }

    #[test]
    fn descending_order_ranks_largest_first() {
        assert_eq!(descending_order(&[1.0, 5.0, 3.0]), vec![1, 2, 0]);
    }
}
