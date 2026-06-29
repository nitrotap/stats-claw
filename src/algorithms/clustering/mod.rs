//! Clustering algorithms and their shared types.
//!
//! Each concrete algorithm lives in its own submodule (`kmeans`, `dbscan`,
//! `hierarchical`, `mean_shift`, `affinity`, `spectral`, `gmm`) and is re-exported
//! here so callers name `clustering::<algo>` without the submodule path. This module
//! also owns the small types every member shares: the [`NOISE`] sentinel label,
//! cluster result structs, and the contiguous-relabeling helper.
//!
//! All routines return labels as `Vec<usize>`. Labels are arbitrary integers —
//! their absolute values carry no meaning, only the partition they induce — so the
//! equivalence suite compares them to `scikit-learn` by adjusted Rand index, which
//! is invariant to relabeling.

mod affinity;
mod dbscan;
mod gmm;
mod hierarchical;
mod kmeans;
mod mean_shift;
mod spectral;

pub use affinity::affinity_propagation;
pub use dbscan::{DbscanResult, dbscan};
pub use gmm::{GmmResult, gmm_em};
pub use hierarchical::{Linkage, agglomerative};
pub use kmeans::{KMeansResult, kmeans};
pub use mean_shift::{MeanShiftResult, mean_shift};
pub use spectral::spectral;

/// Label assigned to a point that belongs to no cluster (DBSCAN noise / outlier).
///
/// `scikit-learn` uses `-1`; since stats-claw labels are unsigned, the maximum
/// `usize` is reserved as the noise sentinel. The adjusted Rand index treats all
/// noise-labelled points as a single group.
pub const NOISE: usize = usize::MAX;

/// Relabels a partition so its cluster ids are contiguous from `0`, preserving the
/// order in which clusters first appear.
///
/// The [`NOISE`] sentinel is passed through unchanged so noise points stay
/// distinguishable from genuine clusters. This normalizes algorithm output to a
/// canonical form without affecting the partition (and therefore the ARI).
///
/// # Arguments
///
/// * `labels` — raw cluster ids, possibly sparse or containing [`NOISE`].
///
/// # Returns
///
/// A new label vector with clusters renumbered `0, 1, 2, …` by first appearance.
#[must_use]
pub fn relabel_contiguous(labels: &[usize]) -> Vec<usize> {
    let mut mapping: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut next = 0_usize;
    labels
        .iter()
        .map(|&l| {
            if l == NOISE {
                return NOISE;
            }
            *mapping.entry(l).or_insert_with(|| {
                let id = next;
                next += 1;
                id
            })
        })
        .collect()
}

/// Counts the distinct non-noise clusters present in a partition.
///
/// # Arguments
///
/// * `labels` — a label vector, possibly containing [`NOISE`].
///
/// # Returns
///
/// The number of unique labels excluding [`NOISE`].
#[must_use]
pub fn cluster_count(labels: &[usize]) -> usize {
    labels
        .iter()
        .filter(|&&l| l != NOISE)
        .collect::<std::collections::HashSet<_>>()
        .len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relabel_makes_ids_contiguous_by_first_appearance() {
        let got = relabel_contiguous(&[5, 5, 9, 9, 2]);
        assert_eq!(got, vec![0, 0, 1, 1, 2], "relabeled = {got:?}");
    }

    #[test]
    fn relabel_preserves_noise() {
        let got = relabel_contiguous(&[NOISE, 7, NOISE, 7]);
        assert_eq!(got, vec![NOISE, 0, NOISE, 0], "relabeled = {got:?}");
    }

    #[test]
    fn cluster_count_ignores_noise() {
        assert_eq!(cluster_count(&[0, 1, NOISE, 1, NOISE]), 2);
    }
}
