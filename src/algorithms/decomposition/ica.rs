//! Independent component analysis via the symmetric `FastICA` fixed-point iteration.
//!
//! `FastICA` (Hyvärinen & Oja 2000) separates a linear mixture into statistically
//! independent sources. The mixed data is mean-centered and whitened (decorrelated
//! to unit variance) using the covariance eigen-decomposition, then a set of
//! orthonormal unmixing directions is found by the symmetric fixed-point update with
//! the `logcosh` contrast `g(u) = tanh(u)` — the same nonlinearity `scikit-learn`'s
//! `FastICA` uses by default. The recovered sources are the whitened data projected
//! onto those directions, normalized to unit variance (matching `whiten=
//! "unit-variance"`).
//!
//! Neither the order nor the sign of the recovered sources is identifiable, so the
//! equivalence suite matches each reference source to its best-correlated recovered
//! source and checks the magnitude of that correlation rather than raw values.

use crate::algorithms::decomposition::{
    at, count_to_f64, descending_order, jacobi_eigen, mean_center,
};
use crate::rng::SplitMix64;

/// Outcome of a `FastICA` fit.
#[derive(Debug, Clone)]
pub struct IcaResult {
    /// The recovered independent sources, one row per observation and one column per
    /// component, each column zero-mean and unit-variance.
    pub sources: Vec<Vec<f64>>,
}

/// Separates the linear mixture `mixed` into `n_components` independent sources.
///
/// # Arguments
///
/// * `mixed` — observed mixed signals; each inner slice is one observation. Empty
///   input yields an empty result.
/// * `n_components` — number of sources to recover, clamped to the feature dimension.
/// * `max_iter` — maximum fixed-point iterations.
/// * `tol` — convergence tolerance on the unmixing matrix between iterations.
/// * `seed` — RNG seed for the orthogonal initialisation (determinism contract).
///
/// # Returns
///
/// An [`IcaResult`] whose `sources` columns are the recovered unit-variance signals.
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::decomposition::fast_ica;
///
/// // Two simple signals mixed by an invertible matrix.
/// let mixed: Vec<Vec<f64>> = (0..50)
///     .map(|i| {
///         let t = f64::from(i) * 0.3;
///         let s1 = t.sin();
///         let s2 = if i % 2 == 0 { 1.0 } else { -1.0 };
///         vec![s1 + 0.5 * s2, 0.3 * s1 + s2]
///     })
///     .collect();
/// let r = fast_ica(&mixed, 2, 200, 1e-4, 1);
/// assert_eq!(r.sources.len(), 50, "one source row per observation");
/// ```
#[must_use]
pub fn fast_ica(
    mixed: &[Vec<f64>],
    n_components: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> IcaResult {
    let dim = mixed.first().map_or(0, Vec::len);
    let k = n_components.min(dim);
    let n = mixed.len();
    if dim == 0 || k == 0 || n == 0 {
        return IcaResult {
            sources: Vec::new(),
        };
    }
    let whitened = whiten(mixed, dim, k);
    let unmixing = symmetric_fastica(&whitened, k, max_iter, tol, seed);
    let sources = project(&whitened, &unmixing, k);
    IcaResult { sources }
}

/// Whitens the data: mean-center, then project onto the covariance eigenvectors
/// scaled to unit variance, keeping the `k` leading directions.
///
/// Returns the `n × k` whitened matrix whose columns are uncorrelated with unit
/// variance, the standard `FastICA` preprocessing.
fn whiten(mixed: &[Vec<f64>], dim: usize, k: usize) -> Vec<Vec<f64>> {
    let (centered, _means) = mean_center(mixed, dim);
    let mut cov = vec![0.0_f64; dim * dim];
    for row in &centered {
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
    let denom = count_to_f64(centered.len());
    if denom > 0.0 {
        for c in &mut cov {
            *c /= denom;
        }
    }
    let (values, vectors) = jacobi_eigen(&cov, dim);
    let order = descending_order(&values);
    let top: Vec<usize> = order.into_iter().take(k).collect();
    centered
        .iter()
        .map(|row| {
            top.iter()
                .map(|&col| {
                    let eig = values.get(col).copied().unwrap_or(0.0).max(1e-12);
                    let projection: f64 = (0..dim)
                        .map(|f| row.get(f).copied().unwrap_or(0.0) * at(&vectors, dim, f, col))
                        .sum();
                    projection / eig.sqrt()
                })
                .collect()
        })
        .collect()
}

/// Runs the symmetric `FastICA` fixed-point iteration with the `logcosh` contrast.
///
/// Returns the `k × k` orthonormal unmixing matrix acting on the whitened data.
///
/// The short names (`w`, `c`, `g`, `u`, `e`) are the canonical `FastICA` notation —
/// `w` the unmixing matrix, `g` the contrast, `u = wᵀx` the projection; renaming
/// them would obscure the standard update rather than clarify it.
#[allow(clippy::many_single_char_names)]
fn symmetric_fastica(
    whitened: &[Vec<f64>],
    k: usize,
    max_iter: usize,
    tol: f64,
    seed: u64,
) -> Vec<Vec<f64>> {
    let mut rng = SplitMix64::new(seed);
    let mut w: Vec<Vec<f64>> = (0..k)
        .map(|_| (0..k).map(|_| rng.standard_normal()).collect())
        .collect();
    symmetric_decorrelate(&mut w, k);

    let n = count_to_f64(whitened.len()).max(1.0);
    for _ in 0..max_iter {
        let mut next = vec![vec![0.0_f64; k]; k];
        for (c, wc) in w.iter().enumerate() {
            // For component c: E[x g(wᵀx)] − E[g'(wᵀx)] w, with g = tanh.
            let mut expectation = vec![0.0_f64; k];
            let mut mean_gprime = 0.0_f64;
            for row in whitened {
                let u: f64 = row.iter().zip(wc).map(|(&x, &wi)| x * wi).sum();
                let g = u.tanh();
                let gprime = g.mul_add(-g, 1.0);
                for (e, &x) in expectation.iter_mut().zip(row) {
                    *e = x.mul_add(g, *e);
                }
                mean_gprime += gprime;
            }
            mean_gprime /= n;
            if let Some(target) = next.get_mut(c) {
                for (slot, (&e, &wi)) in target.iter_mut().zip(expectation.iter().zip(wc)) {
                    *slot = (-mean_gprime).mul_add(wi, e / n);
                }
            }
        }
        symmetric_decorrelate(&mut next, k);
        let delta = max_abs_change(&w, &next, k);
        w = next;
        if delta < tol {
            break;
        }
    }
    w
}

/// Symmetric decorrelation `W ← (W Wᵀ)^{-1/2} W` via the eigen-decomposition of
/// `W Wᵀ`, keeping the rows orthonormal without privileging any component.
fn symmetric_decorrelate(w: &mut [Vec<f64>], k: usize) {
    let mut gram = vec![0.0_f64; k * k];
    for a in 0..k {
        for b in 0..k {
            let dot: f64 = w.get(a).zip(w.get(b)).map_or(0.0, |(wa, wb)| {
                wa.iter().zip(wb).map(|(&x, &y)| x * y).sum()
            });
            if let Some(slot) = gram.get_mut(a * k + b) {
                *slot = dot;
            }
        }
    }
    let (values, vectors) = jacobi_eigen(&gram, k);
    // inv_sqrt = V diag(1/sqrt(λ)) Vᵀ
    let mut inv_sqrt = vec![0.0_f64; k * k];
    for i in 0..k {
        for j in 0..k {
            let mut acc = 0.0_f64;
            for (e, &lambda) in values.iter().enumerate().take(k) {
                let safe = lambda.max(1e-12).sqrt();
                acc += at(&vectors, k, i, e) * at(&vectors, k, j, e) / safe;
            }
            put(&mut inv_sqrt, k, i, j, acc);
        }
    }
    let original: Vec<Vec<f64>> = w.iter().map(Clone::clone).collect();
    for (a, row) in w.iter_mut().enumerate().take(k) {
        for (b, slot) in row.iter_mut().enumerate().take(k) {
            *slot = (0..k)
                .map(|m| {
                    at(&inv_sqrt, k, a, m)
                        * original
                            .get(m)
                            .and_then(|r| r.get(b))
                            .copied()
                            .unwrap_or(0.0)
                })
                .sum();
        }
    }
}

/// Writes entry `(i, j)` into a row-major `n×n` buffer (no-op if out of range).
fn put(matrix: &mut [f64], n: usize, i: usize, j: usize, value: f64) {
    if let Some(slot) = matrix.get_mut(i * n + j) {
        *slot = value;
    }
}

/// Largest absolute entrywise change between two `k×k` unmixing matrices.
fn max_abs_change(a: &[Vec<f64>], b: &[Vec<f64>], k: usize) -> f64 {
    let mut worst = 0.0_f64;
    for row in 0..k {
        for col in 0..k {
            let av = a.get(row).and_then(|r| r.get(col)).copied().unwrap_or(0.0);
            let bv = b.get(row).and_then(|r| r.get(col)).copied().unwrap_or(0.0);
            // Sign-insensitive: a flipped row is the same direction.
            worst = worst.max((av.abs() - bv.abs()).abs());
        }
    }
    worst
}

/// Projects the whitened data onto the unmixing directions, normalizing each source
/// column to zero mean and unit variance (matching `whiten="unit-variance"`).
fn project(whitened: &[Vec<f64>], unmixing: &[Vec<f64>], k: usize) -> Vec<Vec<f64>> {
    let mut sources: Vec<Vec<f64>> = whitened
        .iter()
        .map(|row| {
            (0..k)
                .map(|c| {
                    unmixing
                        .get(c)
                        .map_or(0.0, |w| row.iter().zip(w).map(|(&x, &wi)| x * wi).sum())
                })
                .collect()
        })
        .collect();
    let n = count_to_f64(sources.len()).max(1.0);
    for c in 0..k {
        let mean: f64 = sources
            .iter()
            .map(|r| r.get(c).copied().unwrap_or(0.0))
            .sum::<f64>()
            / n;
        let var: f64 = sources
            .iter()
            .map(|r| {
                let d = r.get(c).copied().unwrap_or(0.0) - mean;
                d * d
            })
            .sum::<f64>()
            / n;
        let std = var.sqrt().max(1e-12);
        for row in &mut sources {
            if let Some(slot) = row.get_mut(c) {
                *slot = (*slot - mean) / std;
            }
        }
    }
    sources
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a deterministic 2-source mixture for the recovery tests.
    fn mixture() -> Vec<Vec<f64>> {
        (0..200)
            .map(|i| {
                let t = f64::from(i) * 0.2;
                let s1 = (2.0 * t).sin();
                let s2 = if (3.0 * t).sin() >= 0.0 { 1.0 } else { -1.0 };
                vec![0.6_f64.mul_add(s2, s1), 0.4_f64.mul_add(s1, 1.2 * s2)]
            })
            .collect()
    }

    #[test]
    fn recovers_two_unit_variance_sources() {
        let r = fast_ica(&mixture(), 2, 300, 1e-5, 3);
        assert_eq!(r.sources.len(), 200, "source row count");
        // Each recovered column has (near) unit variance by construction.
        let n = count_to_f64(r.sources.len());
        for c in 0..2 {
            let var: f64 = r
                .sources
                .iter()
                .map(|row| {
                    let v = row.get(c).copied().unwrap_or(0.0);
                    v * v
                })
                .sum::<f64>()
                / n;
            assert!((var - 1.0).abs() < 1e-6, "source {c} variance was {var}");
        }
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let a = fast_ica(&mixture(), 2, 300, 1e-5, 9).sources;
        let b = fast_ica(&mixture(), 2, 300, 1e-5, 9).sources;
        let first_a = a.first().and_then(|r| r.first()).copied().unwrap_or(0.0);
        let first_b = b.first().and_then(|r| r.first()).copied().unwrap_or(0.0);
        assert_eq!(first_a.to_bits(), first_b.to_bits(), "non-deterministic");
    }

    #[test]
    fn empty_input_is_empty() {
        assert!(fast_ica(&[], 2, 10, 1e-4, 0).sources.is_empty());
    }
}
