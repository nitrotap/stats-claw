//! Agglomerative (bottom-up) hierarchical clustering.
//!
//! Every point starts in its own cluster; the two closest clusters are merged
//! repeatedly until `k` remain. Inter-cluster distances are maintained by the
//! Lance–Williams recurrence, so a single update rule covers Ward, single,
//! complete, and average linkage. The result is a flat `k`-cluster labelling
//! compared to `scikit-learn`'s `AgglomerativeClustering` by adjusted Rand index.

use crate::algorithms::{count_to_f64, euclidean_sq};

/// Inter-cluster distance rule driving the agglomerative merges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Linkage {
    /// Minimises the increase in total within-cluster variance (`scikit-learn`'s
    /// default). Operates on squared Euclidean distances.
    Ward,
    /// Distance between the two nearest members of the clusters.
    Single,
    /// Distance between the two farthest members of the clusters.
    Complete,
    /// Mean pairwise distance between members of the two clusters (UPGMA).
    Average,
}

/// Clusters `data` into `k` groups by agglomerative merging under `linkage`.
///
/// Deterministic — ties are broken by the lower cluster index, so repeated runs
/// match. Returns one label per point; an empty input or `k == 0` yields an empty
/// labelling, and `k` larger than the point count leaves every point singleton.
///
/// # Arguments
///
/// * `data` — observations; each inner slice is one point of equal dimension.
/// * `k` — number of flat clusters to cut the dendrogram into.
/// * `linkage` — the inter-cluster distance rule (see [`Linkage`]).
///
/// # Returns
///
/// A label vector with contiguous ids `0, 1, …` (no noise label).
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::clustering::{agglomerative, Linkage};
///
/// let data = vec![vec![0.0], vec![0.1], vec![9.0], vec![9.1]];
/// let labels = agglomerative(&data, 2, Linkage::Ward);
/// assert_eq!(labels.first(), labels.get(1), "near points split");
/// assert_ne!(labels.first(), labels.get(2), "far points merged");
/// ```
#[must_use]
pub fn agglomerative(data: &[Vec<f64>], k: usize, linkage: Linkage) -> Vec<usize> {
    let n = data.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let mut state = State::new(data, linkage);
    while state.active_count() > k.min(n) {
        let Some((a, b)) = state.closest_pair() else {
            break;
        };
        state.merge(a, b, linkage);
    }
    state.flat_labels()
}

/// Mutable agglomeration state: cluster membership, sizes, and the working
/// pairwise-distance matrix updated in place via Lance–Williams.
struct State {
    /// Cluster id assigned to each point (`members[i]` = cluster of point `i`).
    members: Vec<usize>,
    /// Sizes of every cluster slot (0 once a slot has been merged away).
    sizes: Vec<usize>,
    /// Whether each cluster slot is still an active cluster.
    active: Vec<bool>,
    /// Row-major distance matrix between cluster slots (`dist[i*n + j]`).
    dist: Vec<f64>,
    /// Number of original points (= number of slots).
    n: usize,
}

impl State {
    /// Initialises one singleton cluster per point with the base distance matrix.
    /// Ward stores squared distances; the other linkages store plain distances.
    fn new(data: &[Vec<f64>], linkage: Linkage) -> Self {
        let n = data.len();
        let mut dist = vec![0.0_f64; n * n];
        for i in 0..n {
            for j in (i + 1)..n {
                let base = base_distance(data, i, j, linkage);
                set_dist(&mut dist, n, i, j, base);
            }
        }
        Self {
            members: (0..n).collect(),
            sizes: vec![1; n],
            active: vec![true; n],
            dist,
            n,
        }
    }

    /// Number of clusters still active.
    fn active_count(&self) -> usize {
        self.active.iter().filter(|&&a| a).count()
    }

    /// Finds the closest active cluster pair `(a, b)` with `a < b`.
    fn closest_pair(&self) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        let mut best_d = f64::INFINITY;
        for i in 0..self.n {
            if !active_at(&self.active, i) {
                continue;
            }
            for j in (i + 1)..self.n {
                if !active_at(&self.active, j) {
                    continue;
                }
                let d = get_dist(&self.dist, self.n, i, j);
                if d < best_d {
                    best_d = d;
                    best = Some((i, j));
                }
            }
        }
        best
    }

    /// Merges cluster `b` into `a`, updating distances to all other clusters by the
    /// Lance–Williams recurrence for `linkage`.
    fn merge(&mut self, a: usize, b: usize, linkage: Linkage) {
        let size_a = self.sizes.get(a).copied().unwrap_or(0);
        let size_b = self.sizes.get(b).copied().unwrap_or(0);
        for other in 0..self.n {
            if other == a || other == b || !active_at(&self.active, other) {
                continue;
            }
            let to_a = get_dist(&self.dist, self.n, a, other);
            let to_b = get_dist(&self.dist, self.n, b, other);
            let between = get_dist(&self.dist, self.n, a, b);
            let size_i = self.sizes.get(other).copied().unwrap_or(0);
            let updated = lance_williams(linkage, to_a, to_b, between, size_a, size_b, size_i);
            set_dist(&mut self.dist, self.n, a, other, updated);
        }
        if let Some(slot) = self.sizes.get_mut(a) {
            *slot = size_a + size_b;
        }
        if let Some(slot) = self.active.get_mut(b) {
            *slot = false;
        }
        for m in &mut self.members {
            if *m == b {
                *m = a;
            }
        }
    }

    /// Renumbers the surviving cluster ids to a contiguous `0..k` labelling.
    fn flat_labels(&self) -> Vec<usize> {
        super::relabel_contiguous(&self.members)
    }
}

/// Base distance between singleton points `i` and `j` (squared for Ward, plain
/// Euclidean otherwise).
fn base_distance(data: &[Vec<f64>], i: usize, j: usize, linkage: Linkage) -> f64 {
    let (Some(pi), Some(pj)) = (data.get(i), data.get(j)) else {
        return f64::INFINITY;
    };
    let sq = euclidean_sq(pi, pj);
    match linkage {
        Linkage::Ward => sq,
        _ => sq.sqrt(),
    }
}

/// Lance–Williams update for the distance between a merged cluster `a∪b` and a
/// third cluster `i`, given the prior pairwise distances and cluster sizes.
fn lance_williams(
    linkage: Linkage,
    to_a: f64,
    to_b: f64,
    between: f64,
    size_a: usize,
    size_b: usize,
    size_i: usize,
) -> f64 {
    match linkage {
        Linkage::Single => to_a.min(to_b),
        Linkage::Complete => to_a.max(to_b),
        Linkage::Average => {
            let (na, nb) = (count_to_f64(size_a), count_to_f64(size_b));
            na.mul_add(to_a, nb * to_b) / (na + nb)
        }
        Linkage::Ward => {
            let (na, nb, ni) = (
                count_to_f64(size_a),
                count_to_f64(size_b),
                count_to_f64(size_i),
            );
            let total = na + nb + ni;
            let weighted = (na + ni).mul_add(to_a, (nb + ni) * to_b);
            ni.mul_add(-between, weighted) / total
        }
    }
}

/// Reads the active flag for slot `i`, defaulting to `false` out of range.
fn active_at(active: &[bool], i: usize) -> bool {
    active.get(i).copied().unwrap_or(false)
}

/// Reads the symmetric distance between slots `i` and `j`.
fn get_dist(dist: &[f64], n: usize, i: usize, j: usize) -> f64 {
    dist.get(i * n + j).copied().unwrap_or(f64::INFINITY)
}

/// Writes the distance between slots `i` and `j` symmetrically.
fn set_dist(dist: &mut [f64], n: usize, i: usize, j: usize, value: f64) {
    if let Some(slot) = dist.get_mut(i * n + j) {
        *slot = value;
    }
    if let Some(slot) = dist.get_mut(j * n + i) {
        *slot = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ward_separates_two_far_pairs() {
        let data = vec![vec![0.0], vec![0.1], vec![9.0], vec![9.1]];
        let labels = agglomerative(&data, 2, Linkage::Ward);
        assert_eq!(labels.first(), labels.get(1), "near pair split");
        assert_ne!(labels.first(), labels.get(2), "far pair merged");
    }

    #[test]
    fn single_linkage_matches_ward_on_separated_blobs() {
        let data = vec![vec![0.0], vec![0.2], vec![5.0], vec![5.2]];
        let single = agglomerative(&data, 2, Linkage::Single);
        assert_eq!(single.first(), single.get(1), "near pair split");
        assert_ne!(single.first(), single.get(2), "far pair merged");
    }

    #[test]
    fn deterministic_for_fixed_inputs() {
        let data = vec![vec![0.0], vec![1.0], vec![10.0], vec![11.0]];
        assert_eq!(
            agglomerative(&data, 2, Linkage::Average),
            agglomerative(&data, 2, Linkage::Average)
        );
    }
}
