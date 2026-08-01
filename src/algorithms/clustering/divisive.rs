//! Divisive (top-down) hierarchical clustering — the DIANA algorithm.
//!
//! DIANA (`DIvisive ANAlysis`, Kaufman & Rousseeuw 1990, ch. 6) is the exact
//! opposite of [`super::agglomerative`]: instead of merging singletons upward, it
//! begins with every point in one cluster and repeatedly splits the cluster of
//! largest **diameter** (maximum pairwise Euclidean distance) until `k` clusters
//! remain. Each split "splinters" the point of largest average dissimilarity into
//! a new group and then migrates any point that lies, on average, closer to the
//! splinter than to the points left behind.
//!
//! The routine is fully deterministic — every tie (seed choice, point migration,
//! cluster selection) is broken by the lowest point index — so repeated calls on
//! identical data return identical labels. `scikit-learn` has no divisive
//! clusterer; the reference implementation is R's `cluster::diana`.

use crate::algorithms::{count_to_f64, euclidean_sq};
use std::cmp::Ordering;

/// Clusters `data` into `k` groups by the top-down DIANA algorithm.
///
/// Mirrors [`super::agglomerative`]'s contract: labels are contiguous ids
/// `0, 1, …` renumbered by each cluster's lowest member index, an empty input or
/// `k == 0` yields an empty labelling, and `k` at least the point count leaves
/// every point in its own singleton cluster. Distances are Euclidean and every
/// tie is broken by the lowest point index, so the output is deterministic.
///
/// # Arguments
///
/// * `data` — observations; each inner slice is one point of equal dimension.
/// * `k` — number of flat clusters to produce.
///
/// # Returns
///
/// A label vector, one entry per point, with contiguous ids in `0..k` (no noise
/// label). Every produced cluster is non-empty.
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::clustering::divisive;
///
/// let data = vec![vec![0.0], vec![0.5], vec![9.0], vec![9.5]];
/// let labels = divisive(&data, 2);
/// assert_eq!(labels.first(), labels.get(1), "near points share a cluster");
/// assert_ne!(labels.first(), labels.get(2), "far points are separated");
/// ```
#[must_use]
pub fn divisive(data: &[Vec<f64>], k: usize) -> Vec<usize> {
    let n = data.len();
    if n == 0 || k == 0 {
        return Vec::new();
    }
    let target = k.min(n);
    let mut clusters: Vec<Vec<usize>> = vec![(0..n).collect()];
    while clusters.len() < target {
        let Some(idx) = select_cluster(&clusters, data) else {
            break;
        };
        if idx >= clusters.len() {
            break;
        }
        let cluster = clusters.swap_remove(idx);
        let (old, splinter) = split_cluster(data, &cluster);
        if old.is_empty() || splinter.is_empty() {
            clusters.push(cluster);
            break;
        }
        clusters.push(old);
        clusters.push(splinter);
    }
    labels_from_clusters(&clusters, n)
}

/// Euclidean distance between points `i` and `j`, or infinity if either index is
/// out of range (a defensive default that never wins a nearest/farthest search).
fn point_distance(data: &[Vec<f64>], i: usize, j: usize) -> f64 {
    match (data.get(i), data.get(j)) {
        (Some(a), Some(b)) => euclidean_sq(a, b).sqrt(),
        _ => f64::INFINITY,
    }
}

/// Squared diameter of `cluster`: the maximum squared pairwise distance among its
/// members. Squared distances are monotonic in true distance, so they order
/// clusters by diameter without a per-pair `sqrt`.
fn diameter_sq(data: &[Vec<f64>], cluster: &[usize]) -> f64 {
    let mut max = 0.0_f64;
    for (pos, &i) in cluster.iter().enumerate() {
        for &j in cluster.iter().skip(pos + 1) {
            let d = match (data.get(i), data.get(j)) {
                (Some(a), Some(b)) => euclidean_sq(a, b),
                _ => f64::INFINITY,
            };
            if d > max {
                max = d;
            }
        }
    }
    max
}

/// Picks the index of the next cluster to split: the splittable cluster (size at
/// least two) of largest diameter, ties broken by the lowest member index.
///
/// Returns `None` when every cluster is a singleton, which is the stopping
/// condition once the target count equals the point count.
fn select_cluster(clusters: &[Vec<usize>], data: &[Vec<f64>]) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut best_diam = f64::NEG_INFINITY;
    let mut best_min = usize::MAX;
    for (idx, cluster) in clusters.iter().enumerate() {
        if cluster.len() < 2 {
            continue;
        }
        let diam = diameter_sq(data, cluster);
        let min_member = cluster.iter().copied().min().unwrap_or(usize::MAX);
        let ord = diam.total_cmp(&best_diam);
        let take = best.is_none()
            || ord == Ordering::Greater
            || (ord == Ordering::Equal && min_member < best_min);
        if take {
            best = Some(idx);
            best_diam = diam;
            best_min = min_member;
        }
    }
    best
}

/// Splits one cluster (size at least two) into the retained group and the
/// splinter group per DIANA.
///
/// The seed of largest average dissimilarity leaves first, then any point whose
/// average distance to the splinter is strictly smaller than to the points left
/// behind migrates, one at a time, until none remains. Both returned groups are
/// sorted ascending and non-empty.
fn split_cluster(data: &[Vec<f64>], cluster: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let seed = choose_seed(data, cluster);
    let mut old: Vec<usize> = cluster.iter().copied().filter(|&p| p != seed).collect();
    let mut splinter = vec![seed];
    while old.len() >= 2 {
        let Some((mover, delta)) = best_mover(data, &old, &splinter) else {
            break;
        };
        if delta > 0.0 {
            old.retain(|&p| p != mover);
            splinter.push(mover);
        } else {
            break;
        }
    }
    old.sort_unstable();
    splinter.sort_unstable();
    (old, splinter)
}

/// Chooses the seed point: the member with the largest average dissimilarity to
/// the other members, ties broken by the lowest index (iteration is ascending and
/// the comparison is strict, so the earliest maximum is kept).
fn choose_seed(data: &[Vec<f64>], cluster: &[usize]) -> usize {
    let mut best_point = cluster.first().copied().unwrap_or(0);
    let mut best_avg = f64::NEG_INFINITY;
    let denom = count_to_f64(cluster.len().saturating_sub(1));
    for &p in cluster {
        let sum: f64 = cluster
            .iter()
            .filter(|&&q| q != p)
            .map(|&q| point_distance(data, p, q))
            .sum();
        let avg = if denom > 0.0 { sum / denom } else { 0.0 };
        if avg > best_avg {
            best_avg = avg;
            best_point = p;
        }
    }
    best_point
}

/// Finds the point in `old` with the largest migration score `delta`, where
/// `delta = avg distance to the rest of `old`` − `avg distance to `splinter``.
///
/// A positive `delta` means the point sits closer to the splinter and should
/// move. `old` is assumed to hold at least two points (so "the rest" is
/// non-empty) and `splinter` at least one. Ties are broken by the lowest index.
fn best_mover(data: &[Vec<f64>], old: &[usize], splinter: &[usize]) -> Option<(usize, f64)> {
    let rest_denom = count_to_f64(old.len().saturating_sub(1));
    let spl_denom = count_to_f64(splinter.len());
    if rest_denom <= 0.0 || spl_denom <= 0.0 {
        return None;
    }
    let mut best: Option<(usize, f64)> = None;
    for &i in old {
        let rest_sum: f64 = old
            .iter()
            .filter(|&&q| q != i)
            .map(|&q| point_distance(data, i, q))
            .sum();
        let spl_sum: f64 = splinter.iter().map(|&q| point_distance(data, i, q)).sum();
        let delta = (rest_sum / rest_denom) - (spl_sum / spl_denom);
        let take = match best {
            Some((_, bd)) => delta > bd,
            None => true,
        };
        if take {
            best = Some((i, delta));
        }
    }
    best
}

/// Renders the final clusters into a per-point label vector, numbering clusters
/// `0, 1, …` by ascending lowest member index for a canonical, stable labelling.
fn labels_from_clusters(clusters: &[Vec<usize>], n: usize) -> Vec<usize> {
    let mut order: Vec<(usize, usize)> = clusters
        .iter()
        .enumerate()
        .map(|(idx, c)| (c.iter().copied().min().unwrap_or(usize::MAX), idx))
        .collect();
    order.sort_unstable();
    let mut labels = vec![0_usize; n];
    for (label, &(_min, idx)) in order.iter().enumerate() {
        if let Some(cluster) = clusters.get(idx) {
            for &p in cluster {
                if let Some(slot) = labels.get_mut(p) {
                    *slot = label;
                }
            }
        }
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn k1_puts_every_point_in_cluster_zero() {
        let data = vec![vec![0.0], vec![5.0], vec![9.0]];
        let labels = divisive(&data, 1);
        assert_eq!(labels, vec![0, 0, 0], "k=1 labels = {labels:?}");
    }

    #[test]
    fn recovers_two_well_separated_blobs() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![10.0, 10.0],
            vec![10.0, 11.0],
            vec![11.0, 10.0],
        ];
        let labels = divisive(&data, 2);
        assert_eq!(labels, vec![0, 0, 0, 1, 1, 1], "blob labels = {labels:?}");
    }

    #[test]
    fn identical_calls_produce_identical_labels() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.3, 0.1],
            vec![5.0, 5.0],
            vec![5.2, 4.9],
            vec![9.0, 1.0],
        ];
        let first = divisive(&data, 3);
        let second = divisive(&data, 3);
        assert_eq!(first, second, "runs diverged: {first:?} vs {second:?}");
    }

    #[test]
    fn k_at_least_n_makes_every_point_a_singleton() {
        let data = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
        let labels = divisive(&data, 10);
        // Four distinct points, k >= n: each in its own cluster, labelled by order.
        assert_eq!(labels, vec![0, 1, 2, 3], "k>=n labels = {labels:?}");
    }

    #[test]
    fn empty_input_and_zero_k_yield_empty_labels() {
        let empty: Vec<Vec<f64>> = Vec::new();
        assert!(
            divisive(&empty, 3).is_empty(),
            "empty data must give no labels"
        );
        let data = vec![vec![0.0], vec![1.0]];
        assert!(divisive(&data, 0).is_empty(), "k=0 must give no labels");
    }

    #[test]
    fn hand_worked_1d_example_matches_traced_diana_run() {
        // Points (index:value): 0:1 1:2 2:3 3:10 4:11 5:12 6:25 7:26, k=3.
        // DIANA reference is R `cluster::diana` (sklearn has no divisive clusterer);
        // the split below is traced by hand with 1-D Euclidean = |a-b|.
        //
        // Split 1 — whole set (diameter |26-1|=25). Seed = point 7 (value 26), the
        //   largest mean dissimilarity (mean 118/7≈16.86 vs 25:16.0, 1:11.71). Point 6
        //   (25) migrates (mean-to-rest 18.5 vs 1 to splinter), then all remaining
        //   deltas go negative. -> {1,2,3,10,11,12} | {25,26}.
        // Split 2 — cluster {1,2,3,10,11,12} (diameter 11 > the {25,26} diameter 1).
        //   Seed = point 0 (value 1) by the lowest-index tie with point 5 (both mean
        //   6.6). Points 2 (value 2) then 3 (value 3) migrate; the 10/11/12 trio then
        //   has all deltas negative. -> {1,2,3} | {10,11,12}.
        // Final clusters {1,2,3},{10,11,12},{25,26} relabel by lowest index.
        let data = vec![
            vec![1.0],
            vec![2.0],
            vec![3.0],
            vec![10.0],
            vec![11.0],
            vec![12.0],
            vec![25.0],
            vec![26.0],
        ];
        let labels = divisive(&data, 3);
        assert_eq!(
            labels,
            vec![0, 0, 0, 1, 1, 1, 2, 2],
            "hand-worked labels = {labels:?}"
        );
    }

    #[test]
    fn labels_span_zero_to_k_with_no_empty_cluster() {
        let data = vec![
            vec![0.0, 0.0],
            vec![0.4, 0.2],
            vec![8.0, 8.0],
            vec![8.3, 7.7],
            vec![20.0, 1.0],
            vec![20.5, 0.8],
        ];
        let k = 3;
        let labels = divisive(&data, k);
        assert_eq!(labels.len(), data.len(), "one label per point: {labels:?}");
        assert!(
            labels.iter().all(|&l| l < k),
            "labels out of 0..{k}: {labels:?}"
        );
        let mut counts = vec![0_usize; k];
        for &l in &labels {
            if let Some(slot) = counts.get_mut(l) {
                *slot += 1;
            }
        }
        assert!(
            counts.iter().all(|&c| c > 0),
            "an empty cluster: {counts:?}"
        );
    }
}
