//! t-distributed stochastic neighbour embedding (t-SNE).
//!
//! t-SNE (van der Maaten & Hinton 2008) embeds data so that local neighbourhoods are
//! preserved. High-dimensional pairwise affinities `P` are Gaussians whose
//! bandwidths are tuned per point to a target perplexity (binary search on the
//! conditional entropy); the symmetric joint `P` is then matched by a Student-t
//! low-dimensional affinity `Q` by minimizing the Kullback–Leibler divergence
//! `KL(P‖Q)` with momentum gradient descent. The embedding is initialised from PCA
//! for a deterministic, well-spread start, mirroring `scikit-learn`'s `init="pca"`.
//!
//! t-SNE is a stochastic, non-convex optimizer; its exact coordinates are not
//! reproducible across implementations, so the equivalence suite scores the
//! embedding by trustworthiness — the preservation of input neighbourhoods — rather
//! than by raw values.

use crate::algorithms::decomposition::{count_to_f64, mean_center, pca};
use crate::rng::SplitMix64;

/// Number of gradient-descent iterations.
const ITERATIONS: usize = 1000;
/// Iterations during which `P` is exaggerated to form tight initial clusters.
const EXAGGERATION_ITERS: usize = 100;
/// Early-exaggeration factor applied to `P`.
const EXAGGERATION: f64 = 4.0;
/// Gradient-descent learning rate.
const LEARNING_RATE: f64 = 8.0;
/// Floor applied to probabilities to keep logs and ratios finite.
const EPS: f64 = 1e-12;

/// Embeds `data` into `n_components` dimensions by t-SNE.
///
/// # Arguments
///
/// * `data` — observations; each inner slice is one point of equal dimension. Empty
///   input yields an empty result.
/// * `n_components` — target embedding dimension (typically 2).
/// * `perplexity` — the effective neighbourhood size; the per-point Gaussian
///   bandwidths are tuned so each conditional distribution has this perplexity.
/// * `seed` — RNG seed for the small init jitter (determinism contract).
///
/// # Returns
///
/// The embedding: one length-`n_components` row per input point.
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::decomposition::tsne;
///
/// // Two well-separated clusters in 3-D.
/// let mut data = vec![vec![0.0, 0.0, 0.0]; 6];
/// for (i, row) in data.iter_mut().enumerate() {
///     let base = if i < 3 { 0.0 } else { 10.0 };
///     *row = vec![base + f64::from(i32::try_from(i).unwrap_or(0)) * 0.01, 0.0, 0.0];
/// }
/// let embedding = tsne(&data, 2, 2.0, 1);
/// assert_eq!(embedding.len(), 6, "one embedding row per point");
/// ```
#[must_use]
pub fn tsne(data: &[Vec<f64>], n_components: usize, perplexity: f64, seed: u64) -> Vec<Vec<f64>> {
    let n = data.len();
    if n == 0 || n_components == 0 {
        return Vec::new();
    }
    let affinities = joint_affinities(data, perplexity);
    let mut embedding = pca_init(data, n_components, seed);
    optimize(&mut embedding, &affinities, n, n_components);
    embedding
}

/// Builds the symmetric joint affinity matrix `P` (row-major `n×n`) from per-point
/// Gaussian conditionals tuned to `perplexity`.
fn joint_affinities(data: &[Vec<f64>], perplexity: f64) -> Vec<f64> {
    let n = data.len();
    let dist = squared_distances(data, n);
    let mut p = vec![0.0_f64; n * n];
    for i in 0..n {
        let beta = tune_beta(&dist, n, i, perplexity);
        let mut sum = 0.0_f64;
        for j in 0..n {
            if i != j {
                let value = (-beta * at(&dist, n, i, j)).exp();
                put(&mut p, n, i, j, value);
                sum += value;
            }
        }
        if sum > 0.0 {
            for j in 0..n {
                let v = at(&p, n, i, j);
                put(&mut p, n, i, j, v / sum);
            }
        }
    }
    // Symmetrize: P = (P + Pᵀ) / (2n), floored.
    let norm = 2.0 * count_to_f64(n);
    let mut joint = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let value = (at(&p, n, i, j) + at(&p, n, j, i)) / norm;
            put(&mut joint, n, i, j, value.max(EPS));
        }
    }
    joint
}

/// Binary-searches the Gaussian precision `β` for point `i` so its conditional
/// distribution has the target `perplexity` (equivalently the target entropy).
fn tune_beta(dist: &[f64], n: usize, i: usize, perplexity: f64) -> f64 {
    let target = perplexity.ln();
    let mut beta = 1.0_f64;
    let mut lo = f64::MIN_POSITIVE;
    let mut hi = f64::INFINITY;
    for _ in 0..50 {
        let (entropy, _) = entropy_and_sum(dist, n, i, beta);
        let diff = entropy - target;
        if diff.abs() < 1e-5 {
            break;
        }
        if diff > 0.0 {
            lo = beta;
            beta = if hi.is_infinite() {
                beta * 2.0
            } else {
                f64::midpoint(beta, hi)
            };
        } else {
            hi = beta;
            beta = f64::midpoint(beta, lo);
        }
    }
    beta
}

/// Returns the Shannon entropy (in nats) and normalisation of point `i`'s
/// conditional Gaussian at precision `beta`.
fn entropy_and_sum(dist: &[f64], n: usize, i: usize, beta: f64) -> (f64, f64) {
    let mut sum = 0.0_f64;
    let mut dsum = 0.0_f64;
    for j in 0..n {
        if i != j {
            let d = at(dist, n, i, j);
            let w = (-beta * d).exp();
            sum += w;
            dsum = d.mul_add(w, dsum);
        }
    }
    if sum <= 0.0 {
        return (0.0, 0.0);
    }
    let entropy = beta * dsum / sum + sum.ln();
    (entropy, sum)
}

/// Computes the dense matrix of squared Euclidean distances between all points.
fn squared_distances(data: &[Vec<f64>], n: usize) -> Vec<f64> {
    let mut dist = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = match (data.get(i), data.get(j)) {
                (Some(a), Some(b)) => a
                    .iter()
                    .zip(b)
                    .map(|(&x, &y)| {
                        let delta = x - y;
                        delta * delta
                    })
                    .sum(),
                _ => 0.0,
            };
            put(&mut dist, n, i, j, d);
            put(&mut dist, n, j, i, d);
        }
    }
    dist
}

/// Initialises the embedding from the top principal-component scores, rescaled so
/// each axis has unit standard deviation (a well-spread, deterministic start), plus
/// a tiny seeded jitter so degenerate ties break deterministically.
fn pca_init(data: &[Vec<f64>], n_components: usize, seed: u64) -> Vec<Vec<f64>> {
    let dim = data.first().map_or(0, Vec::len);
    let (centered, _means) = mean_center(data, dim);
    let result = pca(data, n_components);
    let mut rng = SplitMix64::new(seed);
    let mut embedding: Vec<Vec<f64>> = centered
        .iter()
        .map(|row| {
            result
                .components
                .iter()
                .map(|component| {
                    let score: f64 = row.iter().zip(component).map(|(&x, &c)| x * c).sum();
                    1e-4_f64.mul_add(rng.standard_normal(), score)
                })
                .collect()
        })
        .collect();
    rescale_unit_std(&mut embedding, n_components);
    embedding
}

/// Rescales each embedding axis to unit standard deviation in place.
fn rescale_unit_std(embedding: &mut [Vec<f64>], dim: usize) {
    let n = count_to_f64(embedding.len()).max(1.0);
    for d in 0..dim {
        let mean: f64 = embedding
            .iter()
            .map(|r| r.get(d).copied().unwrap_or(0.0))
            .sum::<f64>()
            / n;
        let var: f64 = embedding
            .iter()
            .map(|r| {
                let v = r.get(d).copied().unwrap_or(0.0) - mean;
                v * v
            })
            .sum::<f64>()
            / n;
        let std = var.sqrt().max(1e-12);
        for row in embedding.iter_mut() {
            if let Some(slot) = row.get_mut(d) {
                *slot = (*slot - mean) / std;
            }
        }
    }
}

/// Runs momentum gradient descent on `KL(P‖Q)`, mutating `embedding` in place.
fn optimize(embedding: &mut [Vec<f64>], affinities: &[f64], n: usize, dim: usize) {
    let mut velocity = vec![vec![0.0_f64; dim]; n];
    for iter in 0..ITERATIONS {
        let exaggeration = if iter < EXAGGERATION_ITERS {
            EXAGGERATION
        } else {
            1.0
        };
        let momentum = if iter < EXAGGERATION_ITERS { 0.5 } else { 0.8 };
        let (q_unnorm, q_sum) = student_t_affinities(embedding, n, dim);
        for i in 0..n {
            let mut grad = vec![0.0_f64; dim];
            for j in 0..n {
                if i == j {
                    continue;
                }
                let p = exaggeration * at(affinities, n, i, j);
                let q = (at(&q_unnorm, n, i, j) / q_sum).max(EPS);
                let mult = 4.0 * (p - q) * at(&q_unnorm, n, i, j);
                for d in 0..dim {
                    let yi = coord(embedding, i, d);
                    let yj = coord(embedding, j, d);
                    if let Some(g) = grad.get_mut(d) {
                        *g += mult * (yi - yj);
                    }
                }
            }
            for d in 0..dim {
                let v = LEARNING_RATE.mul_add(
                    -grad.get(d).copied().unwrap_or(0.0),
                    momentum * coord_vel(&velocity, i, d),
                );
                set_vel(&mut velocity, i, d, v);
                let cur = coord(embedding, i, d);
                set_coord(embedding, i, d, cur + v);
            }
        }
    }
}

/// Computes the unnormalised Student-t affinities `(1 + ‖yᵢ − yⱼ‖²)⁻¹` and their sum.
fn student_t_affinities(embedding: &[Vec<f64>], n: usize, dim: usize) -> (Vec<f64>, f64) {
    let mut q = vec![0.0_f64; n * n];
    let mut sum = 0.0_f64;
    for i in 0..n {
        for j in (i + 1)..n {
            let mut d = 0.0_f64;
            for k in 0..dim {
                let delta = coord(embedding, i, k) - coord(embedding, j, k);
                d = delta.mul_add(delta, d);
            }
            let value = 1.0 / (1.0 + d);
            put(&mut q, n, i, j, value);
            put(&mut q, n, j, i, value);
            sum = 2.0f64.mul_add(value, sum);
        }
    }
    (q, sum.max(EPS))
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

/// Reads velocity coordinate `(point, dim)`.
fn coord_vel(velocity: &[Vec<f64>], point: usize, dim: usize) -> f64 {
    velocity
        .get(point)
        .and_then(|r| r.get(dim))
        .copied()
        .unwrap_or(0.0)
}

/// Writes velocity coordinate `(point, dim)`.
fn set_vel(velocity: &mut [Vec<f64>], point: usize, dim: usize, value: f64) {
    if let Some(slot) = velocity.get_mut(point).and_then(|r| r.get_mut(dim)) {
        *slot = value;
    }
}

/// Reads entry `(i, j)` from a row-major `n×n` buffer (`0.0` if out of range).
fn at(matrix: &[f64], n: usize, i: usize, j: usize) -> f64 {
    matrix.get(i * n + j).copied().unwrap_or(0.0)
}

/// Writes entry `(i, j)` into a row-major `n×n` buffer (no-op if out of range).
fn put(matrix: &mut [f64], n: usize, i: usize, j: usize, value: f64) {
    if let Some(slot) = matrix.get_mut(i * n + j) {
        *slot = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds two well-separated clusters for the neighbourhood-preservation tests.
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
        let embedding = tsne(&data, 2, 5.0, 7);
        assert_eq!(embedding.len(), 20, "embedding row count");
        // A within-cluster pair should be closer than a cross-cluster pair.
        let within = pair_distance(&embedding, 0, 1);
        let across = pair_distance(&embedding, 0, 15);
        assert!(within < across, "within {within} not < across {across}");
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let data = two_clusters();
        let a = tsne(&data, 2, 5.0, 3);
        let b = tsne(&data, 2, 5.0, 3);
        let fa = a.first().and_then(|r| r.first()).copied().unwrap_or(0.0);
        let fb = b.first().and_then(|r| r.first()).copied().unwrap_or(0.0);
        assert_eq!(fa.to_bits(), fb.to_bits(), "non-deterministic embedding");
    }

    #[test]
    fn empty_input_is_empty() {
        assert!(tsne(&[], 2, 5.0, 0).is_empty());
    }

    /// Euclidean distance between two embedding rows.
    fn pair_distance(embedding: &[Vec<f64>], left: usize, right: usize) -> f64 {
        match (embedding.get(left), embedding.get(right)) {
            (Some(pa), Some(pb)) => pa
                .iter()
                .zip(pb)
                .map(|(&x, &y)| {
                    let delta = x - y;
                    delta * delta
                })
                .sum::<f64>()
                .sqrt(),
            _ => 0.0,
        }
    }
}
