//! Principal component analysis (the worked decomposition PATTERN).
//!
//! PCA finds the orthogonal directions of greatest variance: the data is
//! mean-centered, its covariance matrix is formed, and that symmetric matrix is
//! eigen-decomposed by Jacobi rotation. The eigenvectors sorted by descending
//! eigenvalue are the principal components; each eigenvalue is the variance
//! explained along its component, mirroring `sklearn.decomposition.PCA` with the
//! full SVD solver (which uses the same unbiased `n − 1` covariance).
//!
//! Components are sign-ambiguous, so the equivalence suite aligns each component's
//! sign before comparing and otherwise checks the sign-invariant explained-variance
//! ratio and reconstruction error.

use crate::algorithms::decomposition::{
    at, covariance, descending_order, jacobi_eigen, mean_center, reconstruction_error,
};

/// Outcome of a PCA fit.
#[derive(Debug, Clone)]
pub struct PcaResult {
    /// The top `k` principal components, one length-`dim` unit vector per row, in
    /// descending order of explained variance.
    pub components: Vec<Vec<f64>>,
    /// Variance explained by each returned component (the leading eigenvalues of
    /// the covariance matrix).
    pub explained_variance: Vec<f64>,
    /// Each component's explained variance as a fraction of the total variance
    /// across all features.
    pub explained_variance_ratio: Vec<f64>,
    /// Mean squared error of reconstructing the input from the `k` components.
    pub reconstruction_error: f64,
}

/// Fits PCA to `data`, returning the top `n_components` principal components.
///
/// # Arguments
///
/// * `data` — observations; each inner slice is one row of equal dimension. An
///   empty input yields an empty result.
/// * `n_components` — number of components to keep. Clamped to the feature
///   dimension (cannot exceed the number of features).
///
/// # Returns
///
/// A [`PcaResult`] with the components, their explained variance and ratio, and
/// the reconstruction error of projecting onto and back from those components.
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::decomposition::pca;
///
/// // Variance lies entirely along the x-axis.
/// let data = vec![vec![-2.0, 0.0], vec![-1.0, 0.0], vec![1.0, 0.0], vec![2.0, 0.0]];
/// let r = pca(&data, 1);
/// // The leading component points along x and explains all the variance.
/// assert!((r.explained_variance_ratio.first().copied().unwrap_or(0.0) - 1.0).abs() < 1e-9);
/// ```
#[must_use]
pub fn pca(data: &[Vec<f64>], n_components: usize) -> PcaResult {
    let dim = data.first().map_or(0, Vec::len);
    let k = n_components.min(dim);
    if dim == 0 || k == 0 {
        return PcaResult {
            components: Vec::new(),
            explained_variance: Vec::new(),
            explained_variance_ratio: Vec::new(),
            reconstruction_error: 0.0,
        };
    }
    let (centered, _means) = mean_center(data, dim);
    let cov = covariance(&centered, dim);
    let (values, vectors) = jacobi_eigen(&cov, dim);
    let order = descending_order(&values);
    let total: f64 = values.iter().map(|v| v.max(0.0)).sum();

    let mut components = Vec::with_capacity(k);
    let mut explained_variance = Vec::with_capacity(k);
    let mut explained_variance_ratio = Vec::with_capacity(k);
    for &col in order.iter().take(k) {
        let component: Vec<f64> = (0..dim).map(|row| at(&vectors, dim, row, col)).collect();
        let variance = values.get(col).copied().unwrap_or(0.0).max(0.0);
        explained_variance.push(variance);
        explained_variance_ratio.push(if total > 0.0 { variance / total } else { 0.0 });
        components.push(component);
    }

    let reconstructed = reconstruct(&centered, &components, dim);
    let error = reconstruction_error(&centered, &reconstructed);
    PcaResult {
        components,
        explained_variance,
        explained_variance_ratio,
        reconstruction_error: error,
    }
}

/// Reconstructs the centered data from its projection onto `components`.
///
/// For each row `x`, the reconstruction is `Σ_c (x · cᵀ) c` — the projection onto
/// the retained orthonormal components, which is exactly what `sklearn`'s
/// `inverse_transform(transform(x))` computes in the centered space.
fn reconstruct(centered: &[Vec<f64>], components: &[Vec<f64>], dim: usize) -> Vec<Vec<f64>> {
    centered
        .iter()
        .map(|row| {
            let mut recon = vec![0.0_f64; dim];
            for component in components {
                let score: f64 = row.iter().zip(component).map(|(&x, &c)| x * c).sum();
                for (r, &c) in recon.iter_mut().zip(component) {
                    *r = score.mul_add(c, *r);
                }
            }
            recon
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leading_component_captures_x_axis_variance() {
        let data = vec![
            vec![-2.0, 0.0],
            vec![-1.0, 0.0],
            vec![1.0, 0.0],
            vec![2.0, 0.0],
        ];
        let r = pca(&data, 1);
        let ratio = r.explained_variance_ratio.first().copied().unwrap_or(0.0);
        assert!((ratio - 1.0).abs() < 1e-9, "ratio was {ratio}");
        // One component along the x-axis reconstructs the (already y-free) data exactly.
        assert!(
            r.reconstruction_error < 1e-9,
            "recon error was {}",
            r.reconstruction_error
        );
    }

    #[test]
    fn ratios_sum_to_one_when_all_components_kept() {
        let data = vec![
            vec![1.0, 2.0],
            vec![3.0, 1.0],
            vec![2.0, 4.0],
            vec![5.0, 0.0],
        ];
        let r = pca(&data, 2);
        let total: f64 = r.explained_variance_ratio.iter().sum();
        assert!((total - 1.0).abs() < 1e-9, "ratios summed to {total}");
    }

    #[test]
    fn k_is_clamped_to_dimension() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let r = pca(&data, 5);
        assert_eq!(r.components.len(), 2, "components should clamp to dim");
    }
}
