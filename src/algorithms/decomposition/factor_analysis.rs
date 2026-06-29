//! Factor analysis via the SVD-based EM of `sklearn.decomposition.FactorAnalysis`.
//!
//! Factor analysis models each observation as `x = W·z + μ + ε` with `z` a
//! low-dimensional latent factor and `ε` diagonal Gaussian noise of variance `Ψ`.
//! The loadings `W` and noise `Ψ` are fit by the deterministic EM that scikit-learn
//! uses: each iteration scales the centered data by `1/√Ψ`, takes the top-`k`
//! singular directions, forms `W = √max(s²−1, 0)·Vᵀ·√Ψ`, and updates `Ψ` to the
//! residual per-feature variance. The right singular vectors and squared singular
//! values come from the eigen-decomposition of `MᵀM`, reusing the family's Jacobi
//! solver.
//!
//! The factor loadings are defined only up to sign and rotation, so the equivalence
//! suite compares the rotation-invariant reconstruction error rather than `W`
//! itself.

use crate::algorithms::decomposition::{
    at, count_to_f64, jacobi_eigen, mean_center, reconstruction_error, symmetric_inverse,
};

/// Lower bound applied to noise variances and `√Ψ` for numerical stability,
/// matching scikit-learn's `SMALL` constant.
const SMALL: f64 = 1e-12;

/// Outcome of a factor-analysis fit.
#[derive(Debug, Clone)]
pub struct FactorResult {
    /// The `k × dim` factor loading matrix `W`, one row per latent factor.
    pub loadings: Vec<Vec<f64>>,
    /// Per-feature diagonal noise variance `Ψ`, length `dim`.
    pub noise_variance: Vec<f64>,
    /// Mean squared error of reconstructing the input from its latent factors.
    pub reconstruction_error: f64,
}

/// Fits a `k`-factor model to `data` by SVD-based EM.
///
/// # Arguments
///
/// * `data` — observations; each inner slice is one row of equal dimension. Empty
///   input yields an empty result.
/// * `n_components` — number of latent factors `k`, clamped to the feature dimension.
/// * `max_iter` — maximum EM iterations.
/// * `tol` — convergence tolerance on the log-likelihood increase between iterations.
///
/// # Returns
///
/// A [`FactorResult`] with the loadings, noise variance, and reconstruction error.
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::decomposition::factor_analysis;
///
/// let data = vec![
///     vec![1.0, 1.1],
///     vec![2.0, 2.2],
///     vec![3.0, 2.9],
///     vec![4.0, 4.1],
/// ];
/// let r = factor_analysis(&data, 1, 1000, 1e-2);
/// // A single factor captures the shared trend, so reconstruction is tight.
/// assert!(r.reconstruction_error < 0.1, "error was {}", r.reconstruction_error);
/// ```
#[must_use]
pub fn factor_analysis(
    data: &[Vec<f64>],
    n_components: usize,
    max_iter: usize,
    tol: f64,
) -> FactorResult {
    let dim = data.first().map_or(0, Vec::len);
    let k = n_components.min(dim);
    let n = data.len();
    if dim == 0 || k == 0 || n == 0 {
        return FactorResult {
            loadings: Vec::new(),
            noise_variance: Vec::new(),
            reconstruction_error: 0.0,
        };
    }
    let (centered, means) = mean_center(data, dim);
    let var = column_variance(&centered, dim);
    let nsqrt = count_to_f64(n).sqrt();
    let llconst = count_to_f64(dim).mul_add((2.0 * std::f64::consts::PI).ln(), count_to_f64(k));

    let mut psi = vec![1.0_f64; dim];
    let mut loadings = vec![vec![0.0_f64; dim]; k];
    let mut old_ll = f64::NEG_INFINITY;

    for _ in 0..max_iter {
        let sqrt_psi: Vec<f64> = psi.iter().map(|&p| p.sqrt() + SMALL).collect();
        let scaled = scale_columns(&centered, &sqrt_psi, nsqrt);
        let (sq_singular, right_vectors, unexplained) = top_k_svd(&scaled, dim, k);

        loadings = build_loadings(&sq_singular, &right_vectors, &sqrt_psi, dim, k);

        let ll = log_likelihood(llconst, &sq_singular, &psi, unexplained, n);
        if (ll - old_ll) < tol {
            break;
        }
        old_ll = ll;
        update_noise(&mut psi, &var, &loadings, dim);
    }

    let reconstructed = reconstruct(&centered, &loadings, &psi, &means, dim, k);
    let error = reconstruction_error(data, &reconstructed);
    FactorResult {
        loadings,
        noise_variance: psi,
        reconstruction_error: error,
    }
}

/// Computes the per-feature population variance of mean-centered rows (the `n`
/// denominator, matching `numpy.var`).
fn column_variance(centered: &[Vec<f64>], dim: usize) -> Vec<f64> {
    let mut var = vec![0.0_f64; dim];
    for row in centered {
        for (v, &x) in var.iter_mut().zip(row) {
            *v = x.mul_add(x, *v);
        }
    }
    let n = count_to_f64(centered.len());
    if n > 0.0 {
        for v in &mut var {
            *v /= n;
        }
    }
    var
}

/// Returns `centered[i][j] / (sqrt_psi[j] * nsqrt)` — the EM-step scaling.
fn scale_columns(centered: &[Vec<f64>], sqrt_psi: &[f64], nsqrt: f64) -> Vec<Vec<f64>> {
    centered
        .iter()
        .map(|row| {
            row.iter()
                .zip(sqrt_psi)
                .map(|(&x, &sp)| x / (sp * nsqrt))
                .collect()
        })
        .collect()
}

/// Computes the top-`k` squared singular values, right singular vectors, and the
/// unexplained variance of the `n × dim` matrix `scaled`, via the eigen-
/// decomposition of `scaledᵀ·scaled`.
///
/// The eigenvalues of the Gram matrix are the squared singular values; the
/// eigenvectors are the right singular vectors. The unexplained variance is the sum
/// of the squared singular values beyond the top `k` (scikit-learn's `unexp_var`),
/// which the log-likelihood requires so the stopping criterion matches exactly.
///
/// Returns `(squared_singular_values, right_vectors, unexplained_variance)` ordered
/// by descending value.
fn top_k_svd(scaled: &[Vec<f64>], dim: usize, k: usize) -> (Vec<f64>, Vec<Vec<f64>>, f64) {
    let mut gram = vec![0.0_f64; dim * dim];
    for row in scaled {
        for i in 0..dim {
            let ri = row.get(i).copied().unwrap_or(0.0);
            for j in 0..dim {
                let rj = row.get(j).copied().unwrap_or(0.0);
                if let Some(slot) = gram.get_mut(i * dim + j) {
                    *slot = ri.mul_add(rj, *slot);
                }
            }
        }
    }
    let (values, vectors) = jacobi_eigen(&gram, dim);
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&a, &b| {
        let va = values.get(a).copied().unwrap_or(f64::NEG_INFINITY);
        let vb = values.get(b).copied().unwrap_or(f64::NEG_INFINITY);
        vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut sq_singular = Vec::with_capacity(k);
    let mut right_vectors = Vec::with_capacity(k);
    for &col in order.iter().take(k) {
        sq_singular.push(values.get(col).copied().unwrap_or(0.0).max(0.0));
        right_vectors.push((0..dim).map(|row| at(&vectors, dim, row, col)).collect());
    }
    let unexplained: f64 = order
        .iter()
        .skip(k)
        .map(|&col| values.get(col).copied().unwrap_or(0.0).max(0.0))
        .sum();
    (sq_singular, right_vectors, unexplained)
}

/// Builds `W = √max(s² − 1, 0) · Vᵀ · √Ψ`, one loading row per component.
fn build_loadings(
    sq_singular: &[f64],
    right_vectors: &[Vec<f64>],
    sqrt_psi: &[f64],
    dim: usize,
    k: usize,
) -> Vec<Vec<f64>> {
    (0..k)
        .map(|c| {
            let scale = (sq_singular.get(c).copied().unwrap_or(0.0) - 1.0)
                .max(0.0)
                .sqrt();
            let vector = right_vectors.get(c);
            (0..dim)
                .map(|j| {
                    let v = vector.and_then(|row| row.get(j)).copied().unwrap_or(0.0);
                    scale * v * sqrt_psi.get(j).copied().unwrap_or(0.0)
                })
                .collect()
        })
        .collect()
}

/// Evaluates scikit-learn's factor-analysis log-likelihood
/// `−n/2 · (llconst + Σ log s² + unexplained + Σ log Ψ)`.
///
/// The exact value (including `unexplained`) is needed so the `(ll − old_ll) < tol`
/// stopping iteration matches scikit-learn's, which determines the final loadings.
fn log_likelihood(
    llconst: f64,
    sq_singular: &[f64],
    psi: &[f64],
    unexplained: f64,
    n: usize,
) -> f64 {
    let log_s: f64 = sq_singular.iter().map(|&s| s.max(SMALL).ln()).sum();
    let log_psi: f64 = psi.iter().map(|&p| p.ln()).sum();
    let ll = llconst + log_s + unexplained + log_psi;
    ll * (-count_to_f64(n) / 2.0)
}

/// Updates the diagonal noise to the residual per-feature variance
/// `Ψ_j = max(var_j − Σ_c W_{c,j}², SMALL)`.
fn update_noise(psi: &mut [f64], var: &[f64], loadings: &[Vec<f64>], dim: usize) {
    for j in 0..dim {
        let explained: f64 = loadings
            .iter()
            .map(|row| {
                let w = row.get(j).copied().unwrap_or(0.0);
                w * w
            })
            .sum();
        if let Some(slot) = psi.get_mut(j) {
            *slot = (var.get(j).copied().unwrap_or(0.0) - explained).max(SMALL);
        }
    }
}

/// Reconstructs the data: latent means `z = X_c·(W/Ψ)ᵀ·(I + (W/Ψ)·Wᵀ)⁻¹`, then
/// `x̂ = z·W + μ`, mirroring scikit-learn's `transform` followed by the linear map.
fn reconstruct(
    centered: &[Vec<f64>],
    loadings: &[Vec<f64>],
    psi: &[f64],
    means: &[f64],
    dim: usize,
    k: usize,
) -> Vec<Vec<f64>> {
    // Wpsi[c][j] = W[c][j] / psi[j]
    let wpsi: Vec<Vec<f64>> = loadings
        .iter()
        .map(|row| {
            row.iter()
                .zip(psi)
                .map(|(&w, &p)| w / p)
                .collect::<Vec<f64>>()
        })
        .collect();
    // gram[a][b] = (Wpsi · Wᵀ)[a][b], plus identity → I + Wpsi·Wᵀ.
    let mut posterior = vec![0.0_f64; k * k];
    for a in 0..k {
        for b in 0..k {
            let dot: f64 = wpsi.get(a).zip(loadings.get(b)).map_or(0.0, |(wa, lb)| {
                wa.iter().zip(lb).map(|(&x, &y)| x * y).sum()
            });
            let eye = if a == b { 1.0 } else { 0.0 };
            if let Some(slot) = posterior.get_mut(a * k + b) {
                *slot = dot + eye;
            }
        }
    }
    let cov_z = symmetric_inverse(&posterior, k);

    centered
        .iter()
        .map(|row| {
            // tmp[c] = row · Wpsi[c]ᵀ
            let tmp: Vec<f64> = (0..k)
                .map(|c| {
                    wpsi.get(c)
                        .map_or(0.0, |w| row.iter().zip(w).map(|(&x, &y)| x * y).sum())
                })
                .collect();
            // z[c] = Σ_a tmp[a] · cov_z[a][c]
            let z: Vec<f64> = (0..k)
                .map(|c| {
                    (0..k)
                        .map(|a| tmp.get(a).copied().unwrap_or(0.0) * at(&cov_z, k, a, c))
                        .sum()
                })
                .collect();
            // x̂[j] = Σ_c z[c] · W[c][j] + mean[j]
            (0..dim)
                .map(|j| {
                    let acc: f64 = (0..k)
                        .map(|c| {
                            z.get(c).copied().unwrap_or(0.0)
                                * loadings
                                    .get(c)
                                    .and_then(|w| w.get(j))
                                    .copied()
                                    .unwrap_or(0.0)
                        })
                        .sum();
                    acc + means.get(j).copied().unwrap_or(0.0)
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_factor_captures_shared_trend() {
        let data = vec![
            vec![1.0, 1.1],
            vec![2.0, 2.2],
            vec![3.0, 2.9],
            vec![4.0, 4.1],
        ];
        let r = factor_analysis(&data, 1, 1000, 1e-2);
        assert!(
            r.reconstruction_error < 0.1,
            "error was {}",
            r.reconstruction_error
        );
    }

    #[test]
    fn empty_input_is_empty() {
        let r = factor_analysis(&[], 2, 100, 1e-2);
        assert!(r.loadings.is_empty(), "loadings not empty");
    }

    #[test]
    fn k_is_clamped_to_dimension() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let r = factor_analysis(&data, 9, 100, 1e-2);
        assert!(r.loadings.len() <= 2, "factor count exceeded dim");
    }
}
