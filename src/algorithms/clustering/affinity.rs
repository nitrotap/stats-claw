//! Affinity propagation clustering (Frey & Dueck 2007).
//!
//! Points exchange two kinds of real-valued messages over the similarity graph:
//! *responsibility* `r(i, k)` (how well-suited `k` is as `i`'s exemplar versus
//! competitors) and *availability* `a(i, k)` (how appropriate it is for `i` to
//! pick `k`, given other points' support for `k`). Messages are damped and
//! iterated to a fixed point; the points whose self-message `r(k, k) + a(k, k)`
//! is positive become exemplars, and every point joins its best exemplar.
//!
//! Similarity is the negative squared Euclidean distance (the `scikit-learn`
//! default). The algorithm is deterministic — no RNG — so repeated runs match.

use crate::algorithms::euclidean_sq;

/// Clusters `data` by affinity propagation, returning one label per point.
///
/// # Arguments
///
/// * `data` — observations; each inner slice is one point of equal dimension.
/// * `damping` — message damping factor in `[0.5, 1.0)`; higher is more stable.
/// * `preference` — the shared self-similarity `s(k, k)`; lower values yield fewer
///   exemplars (clusters). `scikit-learn` defaults this to the median similarity.
/// * `max_iter` — maximum message-passing iterations.
///
/// # Returns
///
/// A label vector with contiguous ids `0, 1, …` (one cluster per exemplar).
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::clustering::affinity_propagation;
///
/// let data = vec![vec![0.0], vec![0.1], vec![0.2], vec![5.0], vec![5.1], vec![5.2]];
/// let labels = affinity_propagation(&data, 0.9, -1.0, 300);
/// assert_eq!(labels.first(), labels.get(2), "first blob split");
/// assert_ne!(labels.first(), labels.get(3), "blobs merged");
/// ```
#[must_use]
pub fn affinity_propagation(
    data: &[Vec<f64>],
    damping: f64,
    preference: f64,
    max_iter: usize,
) -> Vec<usize> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    let sim = similarity_matrix(data, preference);
    let mut resp = vec![0.0_f64; n * n];
    let mut avail = vec![0.0_f64; n * n];

    for _ in 0..max_iter {
        update_responsibilities(&sim, &avail, &mut resp, n, damping);
        update_availabilities(&resp, &mut avail, n, damping);
    }
    let exemplars = exemplars_of(&resp, &avail, n);
    assign_labels(&sim, &exemplars, n)
}

/// Builds the `n×n` similarity matrix `s(i, j) = −‖xᵢ − xⱼ‖²` with `preference` on
/// the diagonal.
fn similarity_matrix(data: &[Vec<f64>], preference: f64) -> Vec<f64> {
    let n = data.len();
    let mut sim = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let value = if i == j {
                preference
            } else {
                match (data.get(i), data.get(j)) {
                    (Some(pi), Some(pj)) => -euclidean_sq(pi, pj),
                    _ => f64::NEG_INFINITY,
                }
            };
            put(&mut sim, n, i, j, value);
        }
    }
    sim
}

/// Damped responsibility update: `r(i, k) ← s(i, k) − maxₖ'≠k [a(i, k') + s(i, k')]`.
fn update_responsibilities(sim: &[f64], avail: &[f64], resp: &mut [f64], n: usize, damping: f64) {
    for i in 0..n {
        for k in 0..n {
            let mut max1 = f64::NEG_INFINITY;
            let mut max2 = f64::NEG_INFINITY;
            for kp in 0..n {
                let candidate = at(avail, n, i, kp) + at(sim, n, i, kp);
                if candidate > max1 {
                    max2 = max1;
                    max1 = candidate;
                } else if candidate > max2 {
                    max2 = candidate;
                }
            }
            // Subtract the largest competing value, excluding column k itself.
            let competitor = if (at(avail, n, i, k) + at(sim, n, i, k)) >= max1 {
                max2
            } else {
                max1
            };
            let updated = at(sim, n, i, k) - competitor;
            let damped = damping.mul_add(at(resp, n, i, k), (1.0 - damping) * updated);
            put(resp, n, i, k, damped);
        }
    }
}

/// Damped availability update:
/// `a(i, k) ← min(0, r(k, k) + Σ_{i'∉{i,k}} max(0, r(i', k)))` for `i ≠ k`, and
/// `a(k, k) ← Σ_{i'≠k} max(0, r(i', k))`.
fn update_availabilities(resp: &[f64], avail: &mut [f64], n: usize, damping: f64) {
    for k in 0..n {
        let mut pos_sum = 0.0_f64;
        for ip in 0..n {
            if ip != k {
                pos_sum += at(resp, n, ip, k).max(0.0);
            }
        }
        for i in 0..n {
            let updated = if i == k {
                pos_sum
            } else {
                let self_resp = at(resp, n, k, k);
                let exclude = at(resp, n, i, k).max(0.0);
                (self_resp + pos_sum - exclude).min(0.0)
            };
            let damped = damping.mul_add(at(avail, n, i, k), (1.0 - damping) * updated);
            put(avail, n, i, k, damped);
        }
    }
}

/// Returns the exemplar indices: points whose diagonal `r(k, k) + a(k, k) > 0`.
fn exemplars_of(resp: &[f64], avail: &[f64], n: usize) -> Vec<usize> {
    (0..n)
        .filter(|&k| at(resp, n, k, k) + at(avail, n, k, k) > 0.0)
        .collect()
}

/// Assigns every point to its most similar exemplar, then relabels contiguously.
/// Falls back to a single cluster when no exemplar emerged.
fn assign_labels(sim: &[f64], exemplars: &[usize], n: usize) -> Vec<usize> {
    if exemplars.is_empty() {
        return vec![0; n];
    }
    let raw: Vec<usize> = (0..n)
        .map(|i| {
            let mut best = *exemplars.first().unwrap_or(&0);
            let mut best_s = f64::NEG_INFINITY;
            for &e in exemplars {
                let s = at(sim, n, i, e);
                if s > best_s {
                    best_s = s;
                    best = e;
                }
            }
            best
        })
        .collect();
    super::relabel_contiguous(&raw)
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
        let data = vec![
            vec![0.0],
            vec![0.1],
            vec![0.2],
            vec![5.0],
            vec![5.1],
            vec![5.2],
        ];
        // Preference near the median similarity so each tight blob earns one
        // exemplar (a very negative preference would over-split into singletons).
        let labels = affinity_propagation(&data, 0.9, -1.0, 300);
        assert_eq!(labels.first(), labels.get(2), "first blob split");
        assert_ne!(labels.first(), labels.get(3), "blobs merged");
    }

    #[test]
    fn deterministic_for_fixed_inputs() {
        let data = vec![vec![0.0], vec![0.2], vec![9.0], vec![9.2]];
        assert_eq!(
            affinity_propagation(&data, 0.9, -1.0, 200),
            affinity_propagation(&data, 0.9, -1.0, 200)
        );
    }
}
