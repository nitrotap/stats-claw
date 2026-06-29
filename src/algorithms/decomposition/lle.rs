//! Locally linear embedding (LLE).
//!
//! LLE (Roweis & Saul 2000) embeds high-dimensional data into a low-dimensional
//! space that preserves each point's local linear reconstruction from its nearest
//! neighbours. It runs in two stages: first each point's neighbour weights are found
//! by solving the constrained least-squares `min ‖xᵢ − Σⱼ wᵢⱼ xⱼ‖²` with `Σⱼ wᵢⱼ = 1`
//! over its `k` nearest neighbours; then the embedding minimizes the same
//! reconstruction error in the low-dimensional space, which reduces to the bottom
//! eigenvectors of `M = (I − W)ᵀ(I − W)` (skipping the trivial constant
//! eigenvector).
//!
//! The embedding is deterministic given the data and neighbour count. Because the
//! axes are sign/rotation-free, the equivalence suite scores it by trustworthiness —
//! how well the embedding preserves the input neighbourhood structure — rather than
//! by raw coordinates.

use crate::algorithms::decomposition::{at, jacobi_eigen, put, symmetric_inverse};

/// Small ridge added to the neighbour Gram matrix for numerical stability when the
/// local neighbourhood is rank-deficient (scikit-learn's regularisation term).
const REG: f64 = 1e-3;

/// Embeds `data` into `n_components` dimensions by locally linear embedding.
///
/// # Arguments
///
/// * `data` — observations; each inner slice is one point of equal dimension. Empty
///   input yields an empty result.
/// * `n_components` — target embedding dimension `k`.
/// * `n_neighbors` — number of nearest neighbours used for local reconstruction;
///   clamped to `data.len() − 1`.
///
/// # Returns
///
/// The embedding: one length-`n_components` row per input point.
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::decomposition::lle;
///
/// // Points along a line embed into 1-D preserving their order/spacing.
/// let data: Vec<Vec<f64>> = (0..10).map(|i| vec![f64::from(i), 0.0]).collect();
/// let embedding = lle(&data, 1, 3);
/// assert_eq!(embedding.len(), 10, "one embedding row per point");
/// ```
#[must_use]
pub fn lle(data: &[Vec<f64>], n_components: usize, n_neighbors: usize) -> Vec<Vec<f64>> {
    let n = data.len();
    if n == 0 || n_components == 0 {
        return Vec::new();
    }
    let k = n_neighbors.min(n.saturating_sub(1)).max(1);
    let neighbors = nearest_neighbors(data, k);
    let weights = reconstruction_weights(data, &neighbors, k);
    let cost = embedding_cost_matrix(&neighbors, &weights, n);
    bottom_embedding(&cost, n, n_components)
}

/// Finds the `k` nearest neighbours (by squared Euclidean distance, excluding self)
/// of every point, returned as a per-point list of indices.
fn nearest_neighbors(data: &[Vec<f64>], k: usize) -> Vec<Vec<usize>> {
    (0..data.len())
        .map(|i| {
            let origin = data.get(i);
            let mut order: Vec<usize> = (0..data.len()).filter(|&j| j != i).collect();
            order.sort_by(|&a, &b| {
                let da = origin
                    .zip(data.get(a))
                    .map_or(f64::INFINITY, |(o, p)| squared_distance(o, p));
                let db = origin
                    .zip(data.get(b))
                    .map_or(f64::INFINITY, |(o, p)| squared_distance(o, p));
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            });
            order.into_iter().take(k).collect()
        })
        .collect()
}

/// Squared Euclidean distance between two equal-length points.
fn squared_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Solves each point's constrained reconstruction weights over its neighbours.
///
/// For point `i` with neighbours `N`, the local Gram matrix `G_{ab} = (xᵢ − x_a)·
/// (xᵢ − x_b)` is regularized and inverted, and the weights are `G⁻¹·1` renormalized
/// to sum to one, the closed-form solution of the constrained least-squares.
fn reconstruction_weights(data: &[Vec<f64>], neighbors: &[Vec<usize>], k: usize) -> Vec<Vec<f64>> {
    neighbors
        .iter()
        .enumerate()
        .map(|(i, neigh)| {
            let origin = data.get(i);
            // Local Gram matrix G of the centered neighbour offsets.
            let mut gram = vec![0.0_f64; k * k];
            let mut trace = 0.0_f64;
            for a in 0..k {
                for b in 0..k {
                    let value = match (origin, neigh.get(a), neigh.get(b)) {
                        (Some(o), Some(&na), Some(&nb)) => {
                            dot_offsets(o, data.get(na), data.get(nb))
                        }
                        _ => 0.0,
                    };
                    if a == b {
                        trace += value;
                    }
                    put(&mut gram, k, a, b, value);
                }
            }
            // Ridge-regularize the diagonal (scaled by the trace, as sklearn does).
            let ridge = REG * if trace > 0.0 { trace } else { 1.0 };
            for d in 0..k {
                let cur = at(&gram, k, d, d);
                put(&mut gram, k, d, d, cur + ridge);
            }
            let inv = symmetric_inverse(&gram, k);
            // w = G⁻¹·1, then normalize to sum 1.
            let mut w: Vec<f64> = (0..k)
                .map(|a| (0..k).map(|b| at(&inv, k, a, b)).sum())
                .collect();
            let total: f64 = w.iter().sum();
            if total.abs() > 1e-12 {
                for wi in &mut w {
                    *wi /= total;
                }
            }
            w
        })
        .collect()
}

/// Dot product of two neighbour offsets `(o − a)·(o − b)`.
fn dot_offsets(o: &[f64], a: Option<&Vec<f64>>, b: Option<&Vec<f64>>) -> f64 {
    match (a, b) {
        (Some(pa), Some(pb)) => o
            .iter()
            .zip(pa)
            .zip(pb)
            .map(|((&oi, &ai), &bi)| (oi - ai) * (oi - bi))
            .sum(),
        _ => 0.0,
    }
}

/// Builds the symmetric embedding-cost matrix `M = (I − W)ᵀ(I − W)`.
///
/// `W` is the sparse weight matrix with `W[i][neighbor] = weight`; `M` penalizes any
/// embedding that fails to reproduce each point from its weighted neighbours.
fn embedding_cost_matrix(neighbors: &[Vec<usize>], weights: &[Vec<f64>], n: usize) -> Vec<f64> {
    // Dense (I − W).
    let mut iw = vec![0.0_f64; n * n];
    for i in 0..n {
        put(&mut iw, n, i, i, 1.0);
        if let (Some(neigh), Some(w)) = (neighbors.get(i), weights.get(i)) {
            for (&j, &wij) in neigh.iter().zip(w) {
                let cur = at(&iw, n, i, j);
                put(&mut iw, n, i, j, cur - wij);
            }
        }
    }
    // M = (I − W)ᵀ(I − W).
    let mut m = vec![0.0_f64; n * n];
    for a in 0..n {
        for b in 0..n {
            let value: f64 = (0..n).map(|r| at(&iw, n, r, a) * at(&iw, n, r, b)).sum();
            put(&mut m, n, a, b, value);
        }
    }
    m
}

/// Returns the embedding from the `n_components` eigenvectors of `cost` with the
/// smallest non-zero eigenvalues (the first eigenvector is the trivial constant).
fn bottom_embedding(cost: &[f64], n: usize, n_components: usize) -> Vec<Vec<f64>> {
    let (values, vectors) = jacobi_eigen(cost, n);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let va = values.get(a).copied().unwrap_or(f64::INFINITY);
        let vb = values.get(b).copied().unwrap_or(f64::INFINITY);
        va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
    });
    // Skip the smallest (near-zero) eigenvector and take the next `n_components`.
    let chosen: Vec<usize> = order.into_iter().skip(1).take(n_components).collect();
    (0..n)
        .map(|row| {
            chosen
                .iter()
                .map(|&col| at(&vectors, n, row, col))
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embeds_a_line_into_one_dimension() {
        let data: Vec<Vec<f64>> = (0..10).map(|i| vec![f64::from(i), 0.0]).collect();
        let embedding = lle(&data, 1, 3);
        assert_eq!(embedding.len(), 10, "embedding row count");
        // Endpoints should sit at opposite extremes of the 1-D embedding.
        let first = embedding
            .first()
            .and_then(|r| r.first())
            .copied()
            .unwrap_or(0.0);
        let last = embedding
            .last()
            .and_then(|r| r.first())
            .copied()
            .unwrap_or(0.0);
        assert!((first - last).abs() > 1e-6, "line endpoints collapsed");
    }

    #[test]
    fn deterministic_for_fixed_inputs() {
        let data: Vec<Vec<f64>> = (0..12)
            .map(|i| vec![f64::from(i), f64::from(i % 3)])
            .collect();
        let a = lle(&data, 2, 4);
        let b = lle(&data, 2, 4);
        let fa = a.first().and_then(|r| r.first()).copied().unwrap_or(0.0);
        let fb = b.first().and_then(|r| r.first()).copied().unwrap_or(0.0);
        assert_eq!(fa.to_bits(), fb.to_bits(), "non-deterministic embedding");
    }

    #[test]
    fn empty_input_is_empty() {
        assert!(lle(&[], 2, 3).is_empty());
    }
}
