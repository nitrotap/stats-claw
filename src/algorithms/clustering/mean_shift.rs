//! Mean-shift clustering with a flat (uniform) kernel.
//!
//! Each point is treated as a seed and iteratively shifted to the mean of the
//! points inside its `bandwidth` window until the shift falls below a tolerance.
//! Converged seeds that land within one bandwidth of each other collapse to a
//! single mode; every original point then takes the label of its nearest mode.
//! This mirrors `scikit-learn`'s `MeanShift` with a flat kernel, so the number of
//! discovered modes is the identifiable scalar the equivalence suite checks.

use crate::algorithms::{centroid, euclidean_sq};

/// Outcome of a mean-shift run.
#[derive(Debug, Clone)]
pub struct MeanShiftResult {
    /// Cluster id per input point, in input order.
    pub labels: Vec<usize>,
    /// Number of distinct modes (clusters) discovered.
    pub n_clusters: usize,
    /// The discovered cluster centers (modes), one per cluster.
    pub centers: Vec<Vec<f64>>,
}

/// Maximum mean-shift iterations per seed before giving up on convergence.
const MAX_ITER: usize = 300;
/// Relative shift (as a fraction of bandwidth) below which a seed has converged.
const SHIFT_TOL: f64 = 1e-3;

/// Clusters `data` by shifting every point toward its local density mode.
///
/// Deterministic — no RNG is used; seeds are processed in input order and modes
/// are merged greedily, so repeated runs match. Empty input or a non-positive
/// bandwidth yields an empty result.
///
/// # Arguments
///
/// * `data` — observations; each inner slice is one point of equal dimension.
/// * `bandwidth` — radius of the flat kernel window; must be `> 0`.
///
/// # Returns
///
/// A [`MeanShiftResult`] with per-point labels, the discovered mode count, and the
/// mode centers.
///
/// # Examples
///
/// ```
/// use stats_claw::algorithms::clustering::mean_shift;
///
/// let data = vec![vec![0.0], vec![0.1], vec![10.0], vec![10.1]];
/// let r = mean_shift(&data, 1.0);
/// assert_eq!(r.n_clusters, 2, "modes were {}", r.n_clusters);
/// ```
#[must_use]
pub fn mean_shift(data: &[Vec<f64>], bandwidth: f64) -> MeanShiftResult {
    if data.is_empty() || bandwidth <= 0.0 {
        return MeanShiftResult {
            labels: Vec::new(),
            n_clusters: 0,
            centers: Vec::new(),
        };
    }
    let band_sq = bandwidth * bandwidth;
    let modes: Vec<Vec<f64>> = data
        .iter()
        .map(|seed| shift_to_mode(seed, data, band_sq))
        .collect();
    let centers = merge_modes(&modes, band_sq);
    let labels: Vec<usize> = data
        .iter()
        .map(|point| nearest_center(point, &centers))
        .collect();
    let n_clusters = centers.len();
    MeanShiftResult {
        labels,
        n_clusters,
        centers,
    }
}

/// Iteratively shifts `seed` to the mean of points within the flat kernel until
/// the move shrinks below the tolerance or the iteration cap is hit.
fn shift_to_mode(seed: &[f64], data: &[Vec<f64>], band_sq: f64) -> Vec<f64> {
    let dim = seed.len();
    let mut current = seed.to_vec();
    let tol_sq = band_sq * SHIFT_TOL * SHIFT_TOL;
    for _ in 0..MAX_ITER {
        let within: Vec<&[f64]> = data
            .iter()
            .filter(|p| euclidean_sq(&current, p) <= band_sq)
            .map(Vec::as_slice)
            .collect();
        if within.is_empty() {
            break;
        }
        let next = centroid(&within, dim);
        let moved = euclidean_sq(&current, &next);
        current = next;
        if moved <= tol_sq {
            break;
        }
    }
    current
}

/// Collapses converged modes into representative centers: a mode joins an existing
/// center when it lies within one bandwidth, otherwise it starts a new center.
fn merge_modes(modes: &[Vec<f64>], band_sq: f64) -> Vec<Vec<f64>> {
    let mut centers: Vec<Vec<f64>> = Vec::new();
    for mode in modes {
        let near = centers.iter().any(|c| euclidean_sq(mode, c) <= band_sq);
        if !near {
            centers.push(mode.clone());
        }
    }
    centers
}

/// Returns the index of the center nearest to `point` (0 when none exist).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_two_modes_in_two_blobs() {
        let data = vec![
            vec![0.0],
            vec![0.1],
            vec![0.2],
            vec![10.0],
            vec![10.1],
            vec![10.2],
        ];
        let r = mean_shift(&data, 1.0);
        assert_eq!(r.n_clusters, 2, "modes were {}", r.n_clusters);
        assert_eq!(r.labels.first(), r.labels.get(2), "first blob split");
        assert_ne!(r.labels.first(), r.labels.get(3), "blobs merged");
    }

    #[test]
    fn deterministic_for_fixed_inputs() {
        let data = vec![vec![0.0], vec![0.1], vec![5.0], vec![5.1]];
        assert_eq!(mean_shift(&data, 1.0).labels, mean_shift(&data, 1.0).labels);
    }

    #[test]
    fn empty_input_is_empty_result() {
        let r = mean_shift(&[], 1.0);
        assert!(r.labels.is_empty(), "labels not empty");
        assert_eq!(r.n_clusters, 0, "clusters not zero");
    }
}
