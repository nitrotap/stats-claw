//! UMAP-style neighbour-graph embedding.
//!
//! This is a faithful, dependency-free implementation of the Uniform Manifold
//! Approximation and Projection idea (`McInnes`, Healy & Melville 2018): build a fuzzy
//! `k`-nearest-neighbour graph whose edge weights `exp(−(d − ρ)/σ)` use each point's
//! distance to its nearest neighbour `ρ` and a per-point scale `σ`; symmetrize it as
//! a fuzzy union; then lay the graph out in low dimensions by attractive forces
//! along edges and repulsive forces between sampled non-edges (the UMAP
//! cross-entropy SGD), starting from a PCA initialisation.
//!
//! No `umap-learn` reference is available and the layout SGD is stochastic, so the
//! equivalence suite scores the result by trustworthiness against a data-derived
//! target rather than by matching a reference embedding.

use crate::algorithms::decomposition::{mean_center, pca};
use crate::rng::SplitMix64;

/// Layout optimisation epochs.
const EPOCHS: usize = 500;
/// Attractive/repulsive curve parameter `a` (UMAP's default fit for `min_dist≈0.1`).
const CURVE_A: f64 = 1.577;
/// Attractive/repulsive curve parameter `b`.
const CURVE_B: f64 = 0.895;
/// Initial learning rate, decayed linearly to zero over the epochs.
const LEARNING_RATE: f64 = 1.0;
/// Number of negative (repulsive) samples per positive edge per epoch.
const NEGATIVE_SAMPLES: usize = 5;
/// Gradient clamp keeping the SGD stable.
const CLAMP: f64 = 4.0;

/// Embeds `data` into `n_components` dimensions by a UMAP-style layout.
///
/// # Arguments
///
/// * `data` — observations; each inner slice is one point of equal dimension. Empty
///   input yields an empty result.
/// * `n_components` — target embedding dimension (typically 2).
/// * `n_neighbors` — neighbourhood size for the fuzzy graph; clamped to `len − 1`.
/// * `seed` — RNG seed for the negative sampling and init jitter (determinism).
///
/// # Returns
///
/// The embedding: one length-`n_components` row per input point.
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::decomposition::umap;
///
/// let data: Vec<Vec<f64>> = (0..12).map(|i| vec![f64::from(i), 0.0]).collect();
/// let embedding = umap(&data, 2, 4, 1);
/// assert_eq!(embedding.len(), 12, "one embedding row per point");
/// ```
#[must_use]
pub fn umap(
    data: &[Vec<f64>],
    n_components: usize,
    n_neighbors: usize,
    seed: u64,
) -> Vec<Vec<f64>> {
    let n = data.len();
    if n == 0 || n_components == 0 {
        return Vec::new();
    }
    let k = n_neighbors.min(n.saturating_sub(1)).max(1);
    let edges = fuzzy_edges(data, k);
    let mut embedding = pca_init(data, n_components, seed);
    optimize_layout(&mut embedding, &edges, n, n_components, seed);
    embedding
}

/// A symmetric fuzzy graph edge with its membership weight.
struct Edge {
    /// Source point index.
    from: usize,
    /// Target point index.
    to: usize,
    /// Fuzzy membership strength in `(0, 1]`.
    weight: f64,
}

/// Builds the symmetric fuzzy `k`-NN edge list with UMAP membership weights.
fn fuzzy_edges(data: &[Vec<f64>], k: usize) -> Vec<Edge> {
    let n = data.len();
    // Per-point: sorted neighbour (index, distance), nearest first.
    let neighbours: Vec<Vec<(usize, f64)>> = (0..n)
        .map(|i| {
            let mut list: Vec<(usize, f64)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, distance(data.get(i), data.get(j))))
                .collect();
            list.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            list.truncate(k);
            list
        })
        .collect();

    // Directed membership strengths, then fuzzy-union symmetrize.
    let mut strength = vec![0.0_f64; n * n];
    for (i, list) in neighbours.iter().enumerate() {
        let rho = list.first().map_or(0.0, |&(_, d)| d);
        let sigma = smooth_sigma(list, rho, k);
        for &(j, d) in list {
            let value = (-(d - rho).max(0.0) / sigma).exp();
            if let Some(slot) = strength.get_mut(i * n + j) {
                *slot = value;
            }
        }
    }
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let a = strength.get(i * n + j).copied().unwrap_or(0.0);
            let b = strength.get(j * n + i).copied().unwrap_or(0.0);
            // Fuzzy union: a + b − a·b.
            let weight = a + b - a * b;
            if weight > 1e-12 {
                edges.push(Edge {
                    from: i,
                    to: j,
                    weight,
                });
            }
        }
    }
    edges
}

/// Binary-searches the per-point scale `σ` so the membership row sums to `log2(k)`,
/// the UMAP smoothing target.
fn smooth_sigma(list: &[(usize, f64)], rho: f64, k: usize) -> f64 {
    let target = count_to_f64(k).log2().max(1e-3);
    let mut lo = 0.0_f64;
    let mut hi = f64::INFINITY;
    let mut sigma = 1.0_f64;
    for _ in 0..64 {
        let sum: f64 = list
            .iter()
            .map(|&(_, d)| (-(d - rho).max(0.0) / sigma).exp())
            .sum();
        if (sum - target).abs() < 1e-5 {
            break;
        }
        if sum > target {
            hi = sigma;
            sigma = f64::midpoint(lo, hi);
        } else {
            lo = sigma;
            sigma = if hi.is_infinite() {
                sigma * 2.0
            } else {
                f64::midpoint(lo, hi)
            };
        }
    }
    sigma.max(1e-3)
}

/// Euclidean distance between two optional points.
fn distance(a: Option<&Vec<f64>>, b: Option<&Vec<f64>>) -> f64 {
    match (a, b) {
        (Some(pa), Some(pb)) => pa
            .iter()
            .zip(pb)
            .map(|(&x, &y)| {
                let d = x - y;
                d * d
            })
            .sum::<f64>()
            .sqrt(),
        _ => f64::INFINITY,
    }
}

/// Initialises the embedding from the top principal components with a small jitter.
fn pca_init(data: &[Vec<f64>], n_components: usize, seed: u64) -> Vec<Vec<f64>> {
    let dim = data.first().map_or(0, Vec::len);
    let (centered, _means) = mean_center(data, dim);
    let result = pca(data, n_components);
    let mut rng = SplitMix64::new(seed);
    centered
        .iter()
        .map(|row| {
            result
                .components
                .iter()
                .map(|component| {
                    let score: f64 = row.iter().zip(component).map(|(&x, &c)| x * c).sum();
                    1e-3_f64.mul_add(rng.standard_normal(), score)
                })
                .collect()
        })
        .collect()
}

/// Optimises the layout by edge-attractive and negative-sample-repulsive SGD.
fn optimize_layout(embedding: &mut [Vec<f64>], edges: &[Edge], n: usize, dim: usize, seed: u64) {
    let mut rng = SplitMix64::new(seed ^ 0x5DEE_CE66);
    for epoch in 0..EPOCHS {
        let alpha = LEARNING_RATE * (1.0 - count_ratio(epoch, EPOCHS));
        for edge in edges {
            if rng.next_f64() > edge.weight {
                continue;
            }
            attract(embedding, edge.from, edge.to, dim, alpha);
            for _ in 0..NEGATIVE_SAMPLES {
                let target = uniform_index(&mut rng, n);
                if target != edge.from {
                    repel(embedding, edge.from, target, dim, alpha);
                }
            }
        }
    }
}

/// Applies the attractive gradient pulling edge endpoints together.
fn attract(embedding: &mut [Vec<f64>], i: usize, j: usize, dim: usize, alpha: f64) {
    let dist_sq = pair_dist_sq(embedding, i, j, dim);
    // dC/dy for the UMAP attractive term, ≈ −2ab·d^{2(b−1)} / (1 + a·d^{2b}).
    let denom = CURVE_A.mul_add(dist_sq.powf(CURVE_B), 1.0);
    let coeff = -2.0 * CURVE_A * CURVE_B * dist_sq.max(1e-12).powf(CURVE_B - 1.0) / denom;
    apply_force(embedding, i, j, dim, coeff, alpha);
}

/// Applies the repulsive gradient pushing a non-edge pair apart.
fn repel(embedding: &mut [Vec<f64>], i: usize, j: usize, dim: usize, alpha: f64) {
    let dist_sq = pair_dist_sq(embedding, i, j, dim).max(1e-12);
    let denom = CURVE_A.mul_add(dist_sq.powf(CURVE_B), 1.0) * dist_sq;
    let coeff = 2.0 * CURVE_B / denom;
    apply_force(embedding, i, j, dim, coeff, alpha);
}

/// Moves point `i` by `alpha · coeff · (yᵢ − yⱼ)` (and `j` oppositely), clamped.
fn apply_force(embedding: &mut [Vec<f64>], i: usize, j: usize, dim: usize, coeff: f64, alpha: f64) {
    for d in 0..dim {
        let yi = coord(embedding, i, d);
        let yj = coord(embedding, j, d);
        let delta = (coeff * (yi - yj)).clamp(-CLAMP, CLAMP) * alpha;
        set_coord(embedding, i, d, yi + delta);
        set_coord(embedding, j, d, yj - delta);
    }
}

/// Squared Euclidean distance between embedding rows `i` and `j`.
fn pair_dist_sq(embedding: &[Vec<f64>], i: usize, j: usize, dim: usize) -> f64 {
    (0..dim)
        .map(|d| {
            let delta = coord(embedding, i, d) - coord(embedding, j, d);
            delta * delta
        })
        .sum()
}

/// Reads embedding coordinate `(point, dim)`.
fn coord(embedding: &[Vec<f64>], point: usize, dim: usize) -> f64 {
    embedding
        .get(point)
        .and_then(|r| r.get(dim))
        .copied()
        .unwrap_or(0.0)
}

/// Writes embedding coordinate `(point, dim)`.
fn set_coord(embedding: &mut [Vec<f64>], point: usize, dim: usize, value: f64) {
    if let Some(slot) = embedding.get_mut(point).and_then(|r| r.get_mut(dim)) {
        *slot = value;
    }
}

/// Draws a uniform index in `0..n` from the PRNG.
fn uniform_index(rng: &mut SplitMix64, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let scaled = rng.next_f64() * count_to_f64(n);
    let mut idx = 0_usize;
    let mut bound = 1.0_f64;
    while bound <= scaled && idx + 1 < n {
        idx += 1;
        bound += 1.0;
    }
    idx
}

/// Widens a `usize` to `f64` without an `as` cast.
fn count_to_f64(n: usize) -> f64 {
    crate::algorithms::decomposition::count_to_f64(n)
}

/// Fraction `epoch / total` as an `f64` (for the learning-rate decay).
fn count_ratio(epoch: usize, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    count_to_f64(epoch) / count_to_f64(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds two well-separated clusters in 3-D.
    fn two_clusters() -> Vec<Vec<f64>> {
        (0_usize..20)
            .map(|i| {
                let base = if i < 10 { 0.0 } else { 20.0 };
                let jitter = count_to_f64(i % 10) * 0.05;
                vec![base + jitter, jitter, 0.0]
            })
            .collect()
    }

    #[test]
    fn keeps_clusters_separated() {
        let data = two_clusters();
        let embedding = umap(&data, 2, 5, 7);
        assert_eq!(embedding.len(), 20, "embedding row count");
        let within = pair_distance(&embedding, 0, 1);
        let across = pair_distance(&embedding, 0, 15);
        assert!(within < across, "within {within} not < across {across}");
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let data = two_clusters();
        let a = umap(&data, 2, 5, 3);
        let b = umap(&data, 2, 5, 3);
        let fa = a.first().and_then(|r| r.first()).copied().unwrap_or(0.0);
        let fb = b.first().and_then(|r| r.first()).copied().unwrap_or(0.0);
        assert_eq!(fa.to_bits(), fb.to_bits(), "non-deterministic embedding");
    }

    #[test]
    fn empty_input_is_empty() {
        assert!(umap(&[], 2, 5, 0).is_empty());
    }

    /// Euclidean distance between two embedding rows.
    fn pair_distance(embedding: &[Vec<f64>], i: usize, j: usize) -> f64 {
        pair_dist_sq(embedding, i, j, 2).sqrt()
    }
}
