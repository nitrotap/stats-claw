//! DBSCAN density-based clustering (Ester, Kriegel, Sander & Xu 1996).
//!
//! A point is a *core* point when at least `min_samples` points (itself included,
//! matching `scikit-learn`) lie within distance `eps`. Clusters grow by breadth-
//! first expansion from core points through their `eps`-neighbourhoods; points
//! reachable from no core point are labelled [`NOISE`]. The set of core-point
//! indices is the identifiable scalar the equivalence suite compares exactly.

use crate::algorithms::clustering::NOISE;
use crate::algorithms::euclidean_sq;

/// Outcome of a DBSCAN run.
#[derive(Debug, Clone)]
pub struct DbscanResult {
    /// Cluster id per input point, in input order; noise points carry [`NOISE`].
    pub labels: Vec<usize>,
    /// Indices (into the input) of the points that qualified as core points.
    pub core_samples: Vec<usize>,
}

/// Clusters `data` by density, returning per-point labels and the core-sample set.
///
/// Deterministic: the result depends only on `data`, `eps`, and `min_samples`
/// (no RNG), so repeated runs are identical. Empty input yields an empty result.
///
/// # Arguments
///
/// * `data` — observations; each inner slice is one point of equal dimension.
/// * `eps` — neighbourhood radius. Two points are neighbours when their Euclidean
///   distance is `≤ eps`.
/// * `min_samples` — minimum neighbourhood size (self-inclusive) for a core point.
///
/// # Returns
///
/// A [`DbscanResult`] whose `labels` use contiguous ids `0, 1, …` for clusters and
/// [`NOISE`] for outliers, and whose `core_samples` lists the core-point indices.
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::clustering::{dbscan, NOISE};
///
/// // Two dense triples plus one far outlier.
/// let data = vec![
///     vec![0.0], vec![0.1], vec![0.2],
///     vec![5.0], vec![5.1], vec![5.2],
///     vec![100.0],
/// ];
/// let r = dbscan(&data, 0.5, 3);
/// assert_eq!(r.labels.last(), Some(&NOISE), "outlier not flagged");
/// ```
#[must_use]
pub fn dbscan(data: &[Vec<f64>], eps: f64, min_samples: usize) -> DbscanResult {
    let n = data.len();
    let eps_sq = eps * eps;
    let neighbours: Vec<Vec<usize>> = (0..n).map(|i| region_query(data, i, eps_sq)).collect();
    let is_core: Vec<bool> = neighbours
        .iter()
        .map(|nb| nb.len() >= min_samples)
        .collect();

    let mut labels = vec![UNVISITED; n];
    let mut next_cluster = 0_usize;
    for seed in 0..n {
        if labels.get(seed).copied() != Some(UNVISITED) || !core_at(&is_core, seed) {
            continue;
        }
        expand_cluster(seed, next_cluster, &neighbours, &is_core, &mut labels);
        next_cluster += 1;
    }

    let labels: Vec<usize> = labels
        .into_iter()
        .map(|l| if l == UNVISITED { NOISE } else { l })
        .collect();
    let core_samples = (0..n).filter(|&i| core_at(&is_core, i)).collect();
    DbscanResult {
        labels,
        core_samples,
    }
}

/// Internal marker for a point not yet assigned to a cluster or to noise.
const UNVISITED: usize = usize::MAX - 1;

/// Returns the indices of every point within squared distance `eps_sq` of point
/// `centre` (the point itself included, per the `scikit-learn` convention).
fn region_query(data: &[Vec<f64>], centre: usize, eps_sq: f64) -> Vec<usize> {
    let Some(origin) = data.get(centre) else {
        return Vec::new();
    };
    data.iter()
        .enumerate()
        .filter(|(_, p)| euclidean_sq(origin, p) <= eps_sq)
        .map(|(i, _)| i)
        .collect()
}

/// Reads the core flag for index `i`, defaulting to `false` out of range.
fn core_at(is_core: &[bool], i: usize) -> bool {
    is_core.get(i).copied().unwrap_or(false)
}

/// Grows cluster `cluster` from core point `seed` by breadth-first expansion over
/// `eps`-neighbourhoods, absorbing reachable points (border points join but do not
/// extend the frontier).
fn expand_cluster(
    seed: usize,
    cluster: usize,
    neighbours: &[Vec<usize>],
    is_core: &[bool],
    labels: &mut [usize],
) {
    let mut queue = std::collections::VecDeque::new();
    queue.push_back(seed);
    if let Some(slot) = labels.get_mut(seed) {
        *slot = cluster;
    }
    while let Some(current) = queue.pop_front() {
        if !core_at(is_core, current) {
            continue;
        }
        let Some(reachable) = neighbours.get(current) else {
            continue;
        };
        for &point in reachable {
            if labels.get(point).copied() == Some(UNVISITED) {
                if let Some(slot) = labels.get_mut(point) {
                    *slot = cluster;
                }
                queue.push_back(point);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_isolated_point_as_noise() {
        let data = vec![
            vec![0.0],
            vec![0.1],
            vec![0.2],
            vec![5.0],
            vec![5.1],
            vec![5.2],
            vec![100.0],
        ];
        let r = dbscan(&data, 0.5, 3);
        assert_eq!(r.labels.last(), Some(&NOISE), "outlier not noise");
        assert_eq!(r.labels.first(), r.labels.get(2), "dense triple split");
    }

    #[test]
    fn core_samples_exclude_noise() {
        let data = vec![vec![0.0], vec![0.1], vec![0.2], vec![100.0]];
        let r = dbscan(&data, 0.5, 3);
        assert!(!r.core_samples.contains(&3), "outlier marked core");
        assert_eq!(r.core_samples.len(), 3, "core count");
    }

    #[test]
    fn deterministic_for_fixed_inputs() {
        let data = vec![vec![0.0], vec![0.1], vec![5.0], vec![5.1]];
        assert_eq!(dbscan(&data, 0.5, 2).labels, dbscan(&data, 0.5, 2).labels);
    }
}
