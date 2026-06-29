//! Spectral clustering via the normalized graph Laplacian.
//!
//! The data is turned into an RBF affinity graph `W(i, j) = exp(−γ‖xᵢ − xⱼ‖²)`,
//! whose symmetric normalized form `N = D^{-1/2} W D^{-1/2}` has its `k` leading
//! eigenvectors extracted by Jacobi rotation (equivalently the `k` smallest
//! eigenvectors of the normalized Laplacian `I − N`). Each point's row in that
//! `k`-dimensional spectral embedding is unit-normalized and the embedding is
//! clustered by k-means, mirroring `scikit-learn`'s `SpectralClustering`.
//!
//! The k-means step is seeded for determinism; the eigen step is deterministic, so
//! repeated runs with the same seed match.

use crate::algorithms::clustering::kmeans;
use crate::algorithms::euclidean_sq;

/// Clusters `data` into `k` groups by spectral embedding plus k-means.
///
/// # Arguments
///
/// * `data` — observations; each inner slice is one point of equal dimension.
/// * `k` — number of clusters (and spectral-embedding dimensions).
/// * `gamma` — RBF kernel width parameter; larger values shrink the neighbourhood.
/// * `seed` — seed for the final k-means on the embedding (determinism contract).
///
/// # Returns
///
/// A label vector with contiguous ids `0, 1, …`.
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::clustering::spectral;
///
/// let data = vec![vec![0.0], vec![0.1], vec![9.0], vec![9.1]];
/// let labels = spectral(&data, 2, 1.0, 42);
/// assert_eq!(labels.first(), labels.get(1), "near pair split");
/// assert_ne!(labels.first(), labels.get(2), "far pair merged");
/// ```
#[must_use]
pub fn spectral(data: &[Vec<f64>], k: usize, gamma: f64, seed: u64) -> Vec<usize> {
    let n = data.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let normalized = normalized_affinity(data, gamma);
    let (values, vectors) = jacobi_eigen(&normalized, n);
    let embedding = spectral_embedding(&vectors, &values, n, k.min(n));
    kmeans(&embedding, k, 300, seed).labels
}

/// Builds the symmetric normalized affinity `D^{-1/2} W D^{-1/2}` from the RBF
/// kernel; the leading eigenvectors of this matrix span the spectral embedding.
fn normalized_affinity(data: &[Vec<f64>], gamma: f64) -> Vec<f64> {
    let n = data.len();
    let mut w = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let value = match (data.get(i), data.get(j)) {
                (Some(pi), Some(pj)) => (-gamma * euclidean_sq(pi, pj)).exp(),
                _ => 0.0,
            };
            put(&mut w, n, i, j, value);
        }
    }
    let inv_sqrt_deg: Vec<f64> = (0..n)
        .map(|i| {
            let deg: f64 = (0..n).map(|j| at(&w, n, i, j)).sum();
            if deg > 0.0 { 1.0 / deg.sqrt() } else { 0.0 }
        })
        .collect();
    let mut normalized = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let di = inv_sqrt_deg.get(i).copied().unwrap_or(0.0);
            let dj = inv_sqrt_deg.get(j).copied().unwrap_or(0.0);
            put(&mut normalized, n, i, j, di * at(&w, n, i, j) * dj);
        }
    }
    normalized
}

/// Forms the row-normalized spectral embedding from the `k` eigenvectors with the
/// largest eigenvalues (the leading eigenvectors of the normalized affinity).
fn spectral_embedding(vectors: &[f64], values: &[f64], n: usize, k: usize) -> Vec<Vec<f64>> {
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let va = values.get(b).copied().unwrap_or(f64::NEG_INFINITY);
        let vb = values.get(a).copied().unwrap_or(f64::NEG_INFINITY);
        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
    });
    let top: Vec<usize> = order.into_iter().take(k).collect();
    (0..n)
        .map(|row| {
            let mut coords: Vec<f64> = top
                .iter()
                .map(|&col| vectors.get(row * n + col).copied().unwrap_or(0.0))
                .collect();
            let norm = coords.iter().map(|v| v * v).sum::<f64>().sqrt();
            if norm > 0.0 {
                for c in &mut coords {
                    *c /= norm;
                }
            }
            coords
        })
        .collect()
}

/// Maximum Jacobi sweeps before the eigen-decomposition stops iterating.
const JACOBI_SWEEPS: usize = 100;
/// Off-diagonal magnitude below which the matrix is treated as diagonal.
const JACOBI_TOL: f64 = 1e-12;

/// Computes the eigenvalues and eigenvectors of a symmetric `n×n` matrix by the
/// cyclic Jacobi rotation method.
///
/// Returns `(eigenvalues, eigenvectors)` where eigenvector `col` occupies column
/// `col` of the row-major `n×n` vector buffer.
fn jacobi_eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
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
/// accumulating the rotation into the eigenvector matrix `v`.
///
/// The single-character names (`c`, `s`, `t`, `p`, `q`) are the canonical
/// Jacobi-rotation notation from Numerical Recipes; renaming them would obscure
/// the standard formulae rather than clarify them.
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
        put(a, n, i, p, c * aip - s * aiq);
        put(a, n, i, q, s * aip + c * aiq);
    }
    for i in 0..n {
        let api = at(a, n, p, i);
        let aqi = at(a, n, q, i);
        put(a, n, p, i, c * api - s * aqi);
        put(a, n, q, i, s * api + c * aqi);
    }
    for i in 0..n {
        let vip = at(v, n, i, p);
        let viq = at(v, n, i, q);
        put(v, n, i, p, c * vip - s * viq);
        put(v, n, i, q, s * vip + c * viq);
    }
}

/// Builds the `n×n` identity matrix in a row-major buffer.
fn identity(n: usize) -> Vec<f64> {
    let mut v = vec![0.0_f64; n * n];
    for i in 0..n {
        put(&mut v, n, i, i, 1.0);
    }
    v
}

/// Reads matrix entry `(i, j)` from a row-major `n×n` buffer.
fn at(matrix: &[f64], n: usize, i: usize, j: usize) -> f64 {
    matrix.get(i * n + j).copied().unwrap_or(0.0)
}

/// Writes matrix entry `(i, j)` into a row-major `n×n` buffer.
fn put(matrix: &mut [f64], n: usize, i: usize, j: usize, value: f64) {
    if let Some(slot) = matrix.get_mut(i * n + j) {
        *slot = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn separates_two_far_pairs() {
        let data = vec![vec![0.0], vec![0.1], vec![9.0], vec![9.1]];
        let labels = spectral(&data, 2, 1.0, 42);
        assert_eq!(labels.first(), labels.get(1), "near pair split");
        assert_ne!(labels.first(), labels.get(2), "far pair merged");
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let data = vec![vec![0.0], vec![0.1], vec![9.0], vec![9.1]];
        assert_eq!(spectral(&data, 2, 1.0, 5), spectral(&data, 2, 1.0, 5));
    }

    #[test]
    fn jacobi_recovers_diagonal_eigenvalues() {
        // Diagonal matrix diag(3, 1): eigenvalues are its diagonal entries.
        let m = vec![3.0, 0.0, 0.0, 1.0];
        let (values, _) = jacobi_eigen(&m, 2);
        assert!(values.contains(&3.0) || (values.iter().any(|v| (v - 3.0).abs() < 1e-9)));
    }
}
