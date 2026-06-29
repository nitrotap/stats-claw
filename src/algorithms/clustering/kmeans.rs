//! K-means clustering with k-means++ seeding (the worked clustering PATTERN).
//!
//! Centers are initialised by the k-means++ scheme (Arthur & Vassilvitskii 2007)
//! using the framework's deterministic [`SplitMix64`] RNG, then refined by Lloyd's
//! algorithm until assignments stabilise or `max_iter` is reached. Inertia — the
//! sum over points of the squared distance to the assigned center — is the
//! identifiable scalar the equivalence suite compares to `scikit-learn` exactly.

use crate::algorithms::{centroid, count_to_f64, euclidean_sq};
use crate::rng::SplitMix64;

/// Outcome of a k-means run.
#[derive(Debug, Clone)]
pub struct KMeansResult {
    /// Cluster id assigned to each input point, in input order.
    pub labels: Vec<usize>,
    /// Final cluster centers, one length-`dim` vector per cluster.
    pub centers: Vec<Vec<f64>>,
    /// Total within-cluster sum of squared distances (`Σ ‖xᵢ − c_{label(i)}‖²`).
    pub inertia: f64,
}

/// Partitions `data` into `k` clusters via k-means++ seeding and Lloyd iterations.
///
/// The seeding draws from a [`SplitMix64`] seeded with `seed`, so two runs with the
/// same `seed`, `data`, and `k` produce byte-identical output. Empty inputs or
/// `k == 0` yield an empty result rather than panicking.
///
/// # Arguments
///
/// * `data` — observations; each inner slice is one point and all must share a
///   dimension.
/// * `k` — number of clusters to form. Clamped to `data.len()` (cannot exceed the
///   number of points).
/// * `max_iter` — maximum Lloyd iterations before stopping even without convergence.
/// * `seed` — RNG seed for k-means++ initialisation (determinism contract).
///
/// # Returns
///
/// A [`KMeansResult`] with per-point labels, final centers, and total inertia.
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::clustering::kmeans;
///
/// let data = vec![vec![0.0, 0.0], vec![0.1, 0.0], vec![10.0, 10.0], vec![10.1, 10.0]];
/// let r = kmeans(&data, 2, 100, 42);
/// // The two tight pairs land in different clusters.
/// assert_ne!(r.labels.first(), r.labels.get(2));
/// assert!(r.inertia < 0.1, "inertia was {}", r.inertia);
/// ```
#[must_use]
pub fn kmeans(data: &[Vec<f64>], k: usize, max_iter: usize, seed: u64) -> KMeansResult {
    let n = data.len();
    let k = k.min(n);
    if k == 0 {
        return KMeansResult {
            labels: Vec::new(),
            centers: Vec::new(),
            inertia: 0.0,
        };
    }
    let dim = data.first().map_or(0, Vec::len);
    let mut rng = SplitMix64::new(seed);
    let mut centers = kmeans_plus_plus(data, k, &mut rng);
    let mut labels = vec![0_usize; n];

    for _ in 0..max_iter {
        let changed = assign(data, &centers, &mut labels);
        recompute_centers(data, &labels, k, dim, &mut centers);
        if !changed {
            break;
        }
    }
    // One final assignment so labels reflect the last center update.
    assign(data, &centers, &mut labels);
    let inertia = inertia_of(data, &centers, &labels);
    KMeansResult {
        labels,
        centers,
        inertia,
    }
}

/// Seeds `k` centers by the k-means++ probability scheme.
///
/// The first center is a uniformly chosen point; each subsequent center is drawn
/// with probability proportional to its squared distance to the nearest existing
/// center (the D² weighting that gives k-means++ its expected-cost guarantee).
fn kmeans_plus_plus(data: &[Vec<f64>], k: usize, rng: &mut SplitMix64) -> Vec<Vec<f64>> {
    let n = data.len();
    let mut centers: Vec<Vec<f64>> = Vec::with_capacity(k);
    let Some(first) = data.get(uniform_index(rng, n)) else {
        return centers;
    };
    centers.push(first.clone());
    let mut dist_sq: Vec<f64> = data.iter().map(|p| euclidean_sq(p, first)).collect();

    while centers.len() < k {
        let total: f64 = dist_sq.iter().sum();
        let chosen = if total <= 0.0 {
            uniform_index(rng, n)
        } else {
            weighted_index(rng, &dist_sq, total)
        };
        let Some(picked) = data.get(chosen) else {
            break;
        };
        centers.push(picked.clone());
        for (d, p) in dist_sq.iter_mut().zip(data) {
            *d = d.min(euclidean_sq(p, picked));
        }
    }
    centers
}

/// Draws a uniform index in `0..n` from the PRNG (returns `0` for an empty range).
fn uniform_index(rng: &mut SplitMix64, n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let scaled = rng.next_f64() * count_to_f64(n);
    float_to_index(scaled, n)
}

/// Draws an index with probability proportional to `weights` (sum given as `total`).
fn weighted_index(rng: &mut SplitMix64, weights: &[f64], total: f64) -> usize {
    let target = rng.next_f64() * total;
    let mut acc = 0.0_f64;
    for (i, &w) in weights.iter().enumerate() {
        acc += w;
        if acc >= target {
            return i;
        }
    }
    weights.len().saturating_sub(1)
}

/// Converts a non-negative `f64` in `[0, n)` to a clamped `usize` index without an
/// `as` cast, walking an `f64` accumulator to find the integer floor.
fn float_to_index(value: f64, n: usize) -> usize {
    let mut idx = 0_usize;
    let mut bound = 1.0_f64;
    while bound <= value && idx + 1 < n {
        idx += 1;
        bound += 1.0;
    }
    idx
}

/// Assigns every point to its nearest center, returning whether any label changed.
fn assign(data: &[Vec<f64>], centers: &[Vec<f64>], labels: &mut [usize]) -> bool {
    let mut changed = false;
    for (point, label) in data.iter().zip(labels.iter_mut()) {
        let nearest = nearest_center(point, centers);
        if nearest != *label {
            *label = nearest;
            changed = true;
        }
    }
    changed
}

/// Returns the index of the center closest to `point`.
fn nearest_center(point: &[f64], centers: &[Vec<f64>]) -> usize {
    let mut best = 0_usize;
    let mut best_d = f64::INFINITY;
    for (i, c) in centers.iter().enumerate() {
        let d = euclidean_sq(point, c);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

/// Recomputes each center as the mean of its assigned points; an emptied cluster
/// keeps its previous center so it can recover on a later iteration.
fn recompute_centers(
    data: &[Vec<f64>],
    labels: &[usize],
    k: usize,
    dim: usize,
    centers: &mut [Vec<f64>],
) {
    for (cluster, center) in centers.iter_mut().enumerate().take(k) {
        let members: Vec<&[f64]> = data
            .iter()
            .zip(labels)
            .filter(|(_, &l)| l == cluster)
            .map(|(p, _)| p.as_slice())
            .collect();
        if !members.is_empty() {
            *center = centroid(&members, dim);
        }
    }
}

/// Sums the squared distance from every point to its assigned center.
fn inertia_of(data: &[Vec<f64>], centers: &[Vec<f64>], labels: &[usize]) -> f64 {
    data.iter()
        .zip(labels)
        .map(|(p, &l)| centers.get(l).map_or(0.0, |c| euclidean_sq(p, c)))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads a label by index, returning `usize::MAX` for an out-of-range index so
    /// the assertions stay clear of the `indexing_slicing` lint.
    fn label_at(r: &KMeansResult, i: usize) -> usize {
        r.labels.get(i).copied().unwrap_or(usize::MAX)
    }

    #[test]
    fn separates_two_well_separated_pairs() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.1],
            vec![9.0, 9.0],
            vec![9.1, 9.1],
        ];
        let r = kmeans(&data, 2, 100, 1);
        assert_eq!(label_at(&r, 0), label_at(&r, 1), "first pair split");
        assert_eq!(label_at(&r, 2), label_at(&r, 3), "second pair split");
        assert_ne!(label_at(&r, 0), label_at(&r, 2), "pairs merged");
    }

    #[test]
    fn deterministic_for_fixed_seed() {
        let data = vec![vec![0.0], vec![1.0], vec![10.0], vec![11.0]];
        assert_eq!(
            kmeans(&data, 2, 50, 9).labels,
            kmeans(&data, 2, 50, 9).labels
        );
    }

    #[test]
    fn empty_input_is_empty_result() {
        let r = kmeans(&[], 3, 10, 0);
        assert!(r.labels.is_empty(), "labels not empty");
        assert!((r.inertia - 0.0).abs() < 1e-12, "inertia was {}", r.inertia);
    }
}
