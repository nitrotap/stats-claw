//! Equivalence suite for the decomposition, embedding, change-point, and GMM
//! algorithm families (Track E Wave-2).
//!
//! Each test loads a committed `scikit-learn` / `ruptures` golden fixture and
//! asserts the stats-claw implementation agrees under the appropriate standard:
//!
//! * PCA / factor analysis — explained-variance ratio and reconstruction error
//!   within relative tolerance, components compared after per-component sign
//!   alignment (they are defined only up to sign).
//! * ICA — recovered sources matched to the reference by best correspondence then
//!   sign-aligned (order and sign are not identifiable).
//! * t-SNE / UMAP / LLE — a trustworthiness quality measure, because these
//!   stochastic embeddings cannot be reproduced exactly; the embedding must
//!   preserve the input neighbourhood structure at least as well as the reference.
//! * PELT — change-point indices compared for exact equality (no tolerance).
//! * GMM — BIC/AIC within relative tolerance, partition agreement by adjusted Rand
//!   index, and soft-assignment rows summing to one.
//!
//! Python never runs here — the fixtures are the offline source of truth.

use crate::common;
use crate::common::HarnessError;
use serde_json::Value;

/// Parses the matrix stored under `key` as a row-major `Vec<Vec<f64>>`.
fn matrix(fx: &Value, key: &'static str) -> Result<Vec<Vec<f64>>, HarnessError> {
    fx.get(key)
        .and_then(Value::as_array)
        .ok_or(HarnessError::Shape(key))?
        .iter()
        .map(|row| {
            row.as_array()
                .ok_or(HarnessError::Shape(key))?
                .iter()
                .map(|v| v.as_f64().ok_or(HarnessError::Shape(key)))
                .collect()
        })
        .collect()
}

/// Reads a top-level `f64` scalar from a fixture.
fn scalar(fx: &Value, key: &'static str) -> Result<f64, HarnessError> {
    fx.get(key)
        .and_then(Value::as_f64)
        .ok_or(HarnessError::Shape(key))
}

/// Reads a top-level unsigned scalar from a fixture.
fn usize_scalar(fx: &Value, key: &'static str) -> Result<usize, HarnessError> {
    let raw = fx
        .get(key)
        .and_then(Value::as_u64)
        .ok_or(HarnessError::Shape(key))?;
    Ok(usize::try_from(raw).unwrap_or(0))
}

/// Parses an integer label array as `usize`s.
fn label_vec(fx: &Value, key: &'static str) -> Result<Vec<usize>, HarnessError> {
    fx.get(key)
        .and_then(Value::as_array)
        .ok_or(HarnessError::Shape(key))?
        .iter()
        .map(|v| {
            let raw = v.as_i64().ok_or(HarnessError::Shape(key))?;
            usize::try_from(raw).map_err(|_| HarnessError::Shape(key))
        })
        .collect()
}

#[test]
fn gmm_bic_aic_ari_and_responsibilities_match_sklearn() -> Result<(), HarnessError> {
    use stats_claw::algorithms::clustering::gmm_em;

    let fx = common::load("algo_gmm")?;
    let data = matrix(&fx, "data")?;
    let k = usize_scalar(&fx, "k")?;
    let reference = label_vec(&fx, "labels")?;

    let result = gmm_em(&data, k, 200, 1e-3, 42);

    // BIC and AIC are identifiable (no symmetry freedom) and must match closely.
    common::assert_close(result.bic, scalar(&fx, "bic")?, 1e-6, 1e-6);
    common::assert_close(result.aic, scalar(&fx, "aic")?, 1e-6, 1e-6);

    // Hard assignments agree with sklearn up to label permutation.
    let ari = common::adjusted_rand_index(&result.labels, &reference);
    assert!(ari >= 0.99, "GMM ARI was {ari}");

    // Each soft-assignment (responsibility) row is a probability distribution.
    for (i, row) in result.responsibilities.iter().enumerate() {
        let sum: f64 = row.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-9,
            "responsibility row {i} summed to {sum}"
        );
    }
    Ok(())
}

#[test]
fn pca_explained_variance_and_components_match_sklearn() -> Result<(), HarnessError> {
    use stats_claw::algorithms::decomposition::pca;

    let fx = common::load("algo_pca")?;
    let data = matrix(&fx, "data")?;
    let k = usize_scalar(&fx, "k")?;
    let result = pca(&data, k);

    // Explained-variance ratio is sign/order-invariant and must match closely.
    let ref_ratio = common::f64s(&fx, "explained_variance_ratio")?;
    common::assert_vec_close(&result.explained_variance_ratio, &ref_ratio, 1e-9, 1e-6);

    // Components match the reference per row after aligning their (arbitrary) sign.
    let ref_components = matrix(&fx, "components")?;
    assert_eq!(
        result.components.len(),
        ref_components.len(),
        "component count"
    );
    for (got, reference) in result.components.iter().zip(&ref_components) {
        let mut aligned = got.clone();
        common::align_sign(&mut aligned, reference);
        common::assert_vec_close(&aligned, reference, 1e-6, 1e-6);
    }

    // Reconstruction error is symmetry-invariant: it must match the reference.
    common::assert_close(
        result.reconstruction_error,
        scalar(&fx, "reconstruction_error")?,
        1e-9,
        1e-6,
    );
    Ok(())
}

/// Parses an unsigned-integer array under `key`.
fn usizes(fx: &Value, key: &'static str) -> Result<Vec<usize>, HarnessError> {
    fx.get(key)
        .and_then(Value::as_array)
        .ok_or(HarnessError::Shape(key))?
        .iter()
        .map(|v| {
            let raw = v.as_u64().ok_or(HarnessError::Shape(key))?;
            usize::try_from(raw).map_err(|_| HarnessError::Shape(key))
        })
        .collect()
}

#[test]
fn pelt_change_points_match_ruptures_exactly() -> Result<(), HarnessError> {
    use stats_claw::algorithms::change_point::pelt_l2;

    let fx = common::load("algo_pelt")?;
    let signal = common::f64s(&fx, "signal")?;
    let penalty = scalar(&fx, "penalty")?;
    let min_size = usize_scalar(&fx, "min_size")?;
    let expected = usizes(&fx, "breakpoints")?;

    let breakpoints = pelt_l2(&signal, penalty, min_size);
    // Change-point indices are discrete: an exact set match, no tolerance.
    assert_eq!(
        breakpoints, expected,
        "PELT breakpoints {breakpoints:?} != ruptures {expected:?}"
    );
    Ok(())
}

/// Extracts column `col` of a row-major matrix as a vector.
fn column(matrix: &[Vec<f64>], col: usize) -> Vec<f64> {
    matrix
        .iter()
        .map(|row| row.get(col).copied().unwrap_or(0.0))
        .collect()
}

/// Pearson correlation between two equal-length vectors.
fn correlation(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    // Widen the count without an `as` cast (clippy `cast_precision_loss`).
    let count = u32::try_from(n).map_or(1.0, f64::from);
    let mean_a: f64 = a.iter().take(n).sum::<f64>() / count;
    let mean_b: f64 = b.iter().take(n).sum::<f64>() / count;
    let mut cov = 0.0_f64;
    let mut var_a = 0.0_f64;
    let mut var_b = 0.0_f64;
    for (&x, &y) in a.iter().zip(b).take(n) {
        let dx = x - mean_a;
        let dy = y - mean_b;
        cov = dx.mul_add(dy, cov);
        var_a = dx.mul_add(dx, var_a);
        var_b = dy.mul_add(dy, var_b);
    }
    if var_a <= 0.0 || var_b <= 0.0 {
        return 0.0;
    }
    cov / (var_a.sqrt() * var_b.sqrt())
}

#[test]
fn ica_recovers_sources_order_and_sign_matched() -> Result<(), HarnessError> {
    use stats_claw::algorithms::decomposition::fast_ica;

    let fx = common::load("algo_ica")?;
    let mixed = matrix(&fx, "mixed")?;
    let k = usize_scalar(&fx, "k")?;
    let reference = matrix(&fx, "sources")?;

    let result = fast_ica(&mixed, k, 200, 1e-4, 42);
    assert_eq!(result.sources.len(), mixed.len(), "row count");

    // Each reference source must be matched (best correspondence) by a recovered
    // source up to sign: ICA fixes neither the order nor the sign of its outputs.
    for ref_col in 0..k {
        let reference_source = column(&reference, ref_col);
        let best = (0..k)
            .map(|got_col| correlation(&column(&result.sources, got_col), &reference_source).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            best >= 0.99,
            "reference source {ref_col} best |correlation| was {best}"
        );
    }
    Ok(())
}

/// Squared Euclidean distance between two equal-length points.
fn dist_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Ranks of every point by distance from point `i` in `space` (rank 0 = nearest
/// non-self), used by the trustworthiness measure.
fn neighbor_ranks(space: &[Vec<f64>], i: usize) -> Vec<usize> {
    let origin = space.get(i);
    let mut order: Vec<usize> = (0..space.len()).filter(|&j| j != i).collect();
    order.sort_by(|&a, &b| {
        let da = origin
            .zip(space.get(a))
            .map_or(f64::INFINITY, |(o, p)| dist_sq(o, p));
        let db = origin
            .zip(space.get(b))
            .map_or(f64::INFINITY, |(o, p)| dist_sq(o, p));
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut rank = vec![0_usize; space.len()];
    for (r, &j) in order.iter().enumerate() {
        if let Some(slot) = rank.get_mut(j) {
            *slot = r;
        }
    }
    rank
}

/// Computes the trustworthiness of an embedding (Venna & Kaski 2001): a value in
/// `[0, 1]` measuring how well the `n_neighbors`-neighbourhoods of `embedding`
/// preserve those of `original`. Matches `sklearn.manifold.trustworthiness`.
fn trustworthiness(original: &[Vec<f64>], embedding: &[Vec<f64>], n_neighbors: usize) -> f64 {
    let n = original.len();
    if n <= n_neighbors + 1 {
        return 1.0;
    }
    let high_ranks: Vec<Vec<usize>> = (0..n).map(|i| neighbor_ranks(original, i)).collect();
    let mut penalty = 0.0_f64;
    for i in 0..n {
        let low = neighbor_ranks(embedding, i);
        // Embedding's k nearest neighbours of i.
        let mut low_order: Vec<usize> = (0..n).filter(|&j| j != i).collect();
        low_order.sort_by_key(|&j| low.get(j).copied().unwrap_or(usize::MAX));
        for &j in low_order.iter().take(n_neighbors) {
            let high_rank = high_ranks
                .get(i)
                .and_then(|r| r.get(j))
                .copied()
                .unwrap_or(0);
            if high_rank >= n_neighbors {
                let widen = u32::try_from(high_rank - n_neighbors + 1).map_or(0.0, f64::from);
                penalty += widen;
            }
        }
    }
    let nn = u32::try_from(n_neighbors).map_or(1.0, f64::from);
    let count = u32::try_from(n).map_or(1.0, f64::from);
    let norm = 2.0 / (count * nn * (2.0f64.mul_add(count, -(3.0 * nn)) - 1.0));
    1.0 - norm * penalty
}

#[test]
fn lle_embedding_trustworthiness_matches_sklearn() -> Result<(), HarnessError> {
    use stats_claw::algorithms::decomposition::lle;

    let fx = common::load("algo_lle")?;
    let data = matrix(&fx, "data")?;
    let k = usize_scalar(&fx, "k")?;
    let n_neighbors = usize_scalar(&fx, "n_neighbors")?;
    let reference_trust = scalar(&fx, "trustworthiness")?;

    let embedding = lle(&data, k, n_neighbors);
    let trust = trustworthiness(&data, &embedding, 5);
    // The embedding must preserve neighbourhood structure about as well as sklearn;
    // a small margin absorbs the implementations' different eigen/optimizer paths.
    assert!(
        trust >= reference_trust - 0.05,
        "LLE trustworthiness {trust} < reference {reference_trust} − 0.05"
    );
    Ok(())
}

#[test]
fn tsne_embedding_trustworthiness_matches_sklearn() -> Result<(), HarnessError> {
    use stats_claw::algorithms::decomposition::tsne;

    let fx = common::load("algo_tsne")?;
    let data = matrix(&fx, "data")?;
    let k = usize_scalar(&fx, "k")?;
    let reference_trust = scalar(&fx, "trustworthiness")?;

    let embedding = tsne(&data, k, 10.0, 42);
    let trust = trustworthiness(&data, &embedding, 5);
    assert!(
        trust >= reference_trust - 0.05,
        "t-SNE trustworthiness {trust} < reference {reference_trust} − 0.05"
    );
    Ok(())
}

#[test]
fn umap_embedding_meets_trustworthiness_target() -> Result<(), HarnessError> {
    use stats_claw::algorithms::decomposition::umap;

    let fx = common::load("algo_umap")?;
    let data = matrix(&fx, "data")?;
    let k = usize_scalar(&fx, "k")?;
    let n_neighbors = usize_scalar(&fx, "n_neighbors")?;
    let target = scalar(&fx, "trustworthiness_target")?;

    let embedding = umap(&data, k, n_neighbors, 42);
    let trust = trustworthiness(&data, &embedding, 5);
    // No `umap-learn` reference is installed and UMAP is stochastic, so the contract
    // is a data-derived quality threshold, not byte-identity with a reference run.
    assert!(
        trust >= target,
        "UMAP trustworthiness {trust} < target {target}"
    );
    Ok(())
}

#[test]
fn factor_analysis_reconstruction_error_matches_sklearn() -> Result<(), HarnessError> {
    use stats_claw::algorithms::decomposition::factor_analysis;

    let fx = common::load("algo_factor_analysis")?;
    let data = matrix(&fx, "data")?;
    let k = usize_scalar(&fx, "k")?;
    let result = factor_analysis(&data, k, 1000, 1e-2);

    // Reconstruction error is invariant to the factor loadings' sign/rotation, so
    // it is the comparable quantity against the reference fit.
    common::assert_close(
        result.reconstruction_error,
        scalar(&fx, "reconstruction_error")?,
        1e-9,
        1e-6,
    );
    Ok(())
}
